"""Warp a scanned/photographed page onto its blank reference image.

1. ORB features + ratio test + RANSAC homography, as in the OpenCV tutorial:
   https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
2. A homography assumes a flat page. Curled or bent paper is corrected with a
   thin-plate spline (TPS) fitted to local feature matches, coarse to fine:
   https://docs.opencv.org/4.x/dc/d18/classcv_1_1ThinPlateSplineShapeTransformer.html
   The combined warp is applied to the original scan once, with cv2.remap:
   https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
   Settings were tuned on test_images/ with alignment_test/tps_probe4.py.

The aligned image has the reference's size (1700x2200), so template coordinates
(pixel = PDF point * 200 / 72) apply to it directly.

Each alignment is shown in a window (original | reference | aligned) and
processing waits until 'q' is pressed, like OMRChecker's own image viewer.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
MAX_FEATURES = 5000
RATIO_TEST = 0.75  # keep a match only if clearly better than the second-best
RANSAC_THRESHOLD = 5.0  # max reprojection error in reference pixels
MIN_MATCHES = 10  # findHomography needs >= 4; require more for a usable fit

# Non-flat page correction (TPS)
LOCAL_RADII = (100, 30, 10)  # match search window per pass, reference pixels; coarse to fine
DENSE_FEATURES = 50000  # small table text and grid corners only get features with this many
BLOCK = 200  # reference keypoints are matched block by block; affects speed only
CELL = 60  # one control point per CELL x CELL block: the median of its matches
MIN_PER_CELL = 3  # matches needed for a cell's median to count
NEIGHBORS = 8  # cells compared in the consistency check
MAX_DEVIATION = 8.0  # px a cell's offset may differ from its neighbors' median offset
TPS_REGULARIZATION = 100.0  # must be > 0: with 0 the fit diverged on the test pages
GRID_STEP = 20  # TPS is evaluated on this grid of reference pixels, then interpolated
MIN_CONTROL_POINTS = 10  # fewer than this: keep the previous pass (or the homography alone)

# Alignment window
DISPLAY_HEIGHT = 800  # px height of each of the three views
LABEL_HEIGHT = 40  # px strip above each view for its name
DIVIDER_WIDTH = 4  # px dark line between views
QUIT_KEY = ord("q")


class AlignmentError(Exception):
    pass


@dataclass
class Alignment:
    image: np.ndarray  # grayscale, reference size
    matches: int  # matches passing the ratio test
    inliers: int  # matches consistent with the final homography
    control_points: int  # TPS control points; 0 means only the homography was applied


def align_page(image_path: Path, page_type: str) -> Alignment:
    reference = cv2.imread(str(TEMPLATES_DIR / page_type / "reference.png"), cv2.IMREAD_GRAYSCALE)
    scan = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if reference is None or scan is None:
        raise AlignmentError(f"Could not read '{image_path}' or the {page_type} reference image")
    # Detect features on a copy scaled to the reference height (similar feature scale).
    scale = reference.shape[0] / scan.shape[0]
    small = cv2.resize(scan, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    orb = cv2.ORB_create(MAX_FEATURES)
    scan_kp, scan_des = orb.detectAndCompute(small, None)
    ref_kp, ref_des = orb.detectAndCompute(reference, None)
    if scan_des is None or ref_des is None:
        raise AlignmentError(f"No features found for '{image_path.name}'")

    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(scan_des, ref_des, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO_TEST * p[1].distance]
    if len(good) < MIN_MATCHES:
        raise AlignmentError(f"Only {len(good)} feature matches for '{image_path.name}' against {page_type}")

    # Scan points are mapped back to full-resolution coordinates before fitting,
    # so the warp below samples the original scan rather than the shrunken copy.
    src = np.float32([scan_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) / scale
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESHOLD)
    if homography is None:
        raise AlignmentError(f"No homography found for '{image_path.name}' against {page_type}")

    aligned, control_points = _refine(scan, reference, homography)
    # _show(image_path.name, scan, reference, aligned)
    return Alignment(image=aligned, matches=len(good), inliers=int(mask.sum()), control_points=control_points)


def _refine(scan: np.ndarray, reference: np.ndarray, homography: np.ndarray) -> tuple[np.ndarray, int]:
    """Correct a non-flat page. Returns (aligned image, number of TPS control points).

    The TPS maps reference pixels to pixels of the homography-aligned image. Each pass
    matches features only within `radius` of each other, so repeated text and table
    corners elsewhere on the page are not confused, and starts from the previous
    pass's warp, so the radius can shrink.
    """
    h, w = reference.shape
    coarse = cv2.warpPerspective(scan, homography, (w, h))
    orb = cv2.ORB_create(DENSE_FEATURES)
    ref_kp, ref_des = orb.detectAndCompute(reference, None)
    if ref_des is None:
        return coarse, 0
    ref_pts = np.float32([k.pt for k in ref_kp])

    # Reference grid the TPS is evaluated on, and how to interpolate it to every pixel.
    gw, gh = w // GRID_STEP + 1, h // GRID_STEP + 1
    gx, gy = np.meshgrid(np.linspace(0, w - 1, gw, dtype=np.float32), np.linspace(0, h - 1, gh, dtype=np.float32))
    grid = np.dstack([gx, gy]).reshape(1, -1, 2)
    ux, uy = np.meshgrid(np.arange(w, dtype=np.float32) * (gw - 1) / (w - 1),
                         np.arange(h, dtype=np.float32) * (gh - 1) / (h - 1))

    def to_maps(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Grid of mapped points -> per-pixel (map_x, map_y) for cv2.remap (bilinear, corner-aligned)."""
        g = points.reshape(gh, gw, 2)
        return (cv2.remap(np.ascontiguousarray(g[..., 0]), ux, uy, cv2.INTER_LINEAR),
                cv2.remap(np.ascontiguousarray(g[..., 1]), ux, uy, cv2.INTER_LINEAR))

    tps, control_points, current = None, 0, coarse
    for radius in LOCAL_RADII:
        p, q = _local_matches(orb, current, ref_pts, ref_des, radius)
        cp, cd = _cell_medians(p, q, w)
        if len(cp) >= MIN_CONTROL_POINTS:
            keep = _consistent(cp, cd)
            cp, cd = cp[keep], cd[keep]
        if len(cp) < MIN_CONTROL_POINTS:
            break
        # The feature at reference point cp was found at cp + cd in `current`; the
        # previous TPS gives that position in the homography-aligned image.
        found = (cp + cd).reshape(1, -1, 2)
        target = found if tps is None else tps.applyTransformation(found)[1]
        tps = cv2.createThinPlateSplineShapeTransformer(TPS_REGULARIZATION)
        # estimateTransformation(a, b) then applyTransformation(a) gives b (checked in alignment_test/tps_probe.py).
        tps.estimateTransformation(cp.reshape(1, -1, 2), target.reshape(1, -1, 2),
                                   [cv2.DMatch(i, i, 0) for i in range(len(cp))])
        control_points = len(cp)
        current = cv2.remap(coarse, *to_maps(tps.applyTransformation(grid)[1]), cv2.INTER_LINEAR)

    if tps is None:
        return coarse, 0
    # One resample of the original scan: reference -> TPS -> inverse homography -> scan.
    scan_points = cv2.perspectiveTransform(tps.applyTransformation(grid)[1].reshape(-1, 1, 2), np.linalg.inv(homography))
    return cv2.remap(scan, *to_maps(scan_points), cv2.INTER_LINEAR), control_points


def _local_matches(orb, image: np.ndarray, ref_pts: np.ndarray, ref_des: np.ndarray, radius: int):
    """Match each reference keypoint to `image` keypoints within `radius` (square window).

    Returns (reference points, matching image points), each shaped (N, 2). Reference
    keypoints are the queries, so all candidates of a keypoint are in its block's
    train set and the ratio test sees every one of them.
    """
    kp, des = orb.detectAndCompute(image, None)
    if des is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    pts = np.float32([k.pt for k in kp])
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    h, w = image.shape
    p_out, q_out = [], []
    for y0 in range(0, h, BLOCK):
        for x0 in range(0, w, BLOCK):
            qi = np.flatnonzero((ref_pts[:, 0] >= x0) & (ref_pts[:, 0] < x0 + BLOCK)
                                & (ref_pts[:, 1] >= y0) & (ref_pts[:, 1] < y0 + BLOCK))
            ti = np.flatnonzero((pts[:, 0] >= x0 - radius) & (pts[:, 0] < x0 + BLOCK + radius)
                                & (pts[:, 1] >= y0 - radius) & (pts[:, 1] < y0 + BLOCK + radius))
            if len(qi) == 0 or len(ti) == 0:
                continue
            near = (np.abs(ref_pts[qi, None, :] - pts[None, ti, :]) < radius).all(axis=2).astype(np.uint8)
            for pair in matcher.knnMatch(ref_des[qi], des[ti], k=2, mask=near):
                if len(pair) == 1 or (len(pair) == 2 and pair[0].distance < RATIO_TEST * pair[1].distance):
                    p_out.append(ref_pts[qi[pair[0].queryIdx]])
                    q_out.append(pts[ti[pair[0].trainIdx]])
    return np.float32(p_out).reshape(-1, 2), np.float32(q_out).reshape(-1, 2)


def _cell_medians(p: np.ndarray, q: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """One (reference point, offset) per CELL block: the median of its matches, outvoting wrong ones."""
    offsets = q - p
    cells = (p // CELL).astype(np.int32)
    keys = cells[:, 1] * (width // CELL + 1) + cells[:, 0]
    points, cell_offsets = [], []
    for key in np.unique(keys):
        sel = keys == key
        if sel.sum() >= MIN_PER_CELL:
            points.append(np.median(p[sel], axis=0))
            cell_offsets.append(np.median(offsets[sel], axis=0))
    return np.float32(points).reshape(-1, 2), np.float32(cell_offsets).reshape(-1, 2)


def _consistent(points: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Mask of cells whose offset is within MAX_DEVIATION of their neighbors' median offset."""
    dist = np.linalg.norm(points[:, None] - points[None], axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = np.argsort(dist, axis=1)[:, :NEIGHBORS]
    deviation = np.linalg.norm(offsets - np.median(offsets[nearest], axis=1), axis=1)
    return deviation < MAX_DEVIATION


def _show(name: str, scan: np.ndarray, reference: np.ndarray, aligned: np.ndarray) -> None:
    """Show original | reference | aligned side by side and wait for QUIT_KEY.

    Closing the window with its X button does not continue; the window is shown again.
    """
    views = []
    for label, img in (("Original", scan), ("Reference", reference), ("Aligned", aligned)):
        width = round(img.shape[1] * DISPLAY_HEIGHT / img.shape[0])
        header = np.full((LABEL_HEIGHT, width), 255, np.uint8)
        cv2.putText(header, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
        body = cv2.resize(img, (width, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
        views += [np.vstack([header, body]), np.full((LABEL_HEIGHT + DISPLAY_HEIGHT, DIVIDER_WIDTH), 64, np.uint8)]
    canvas = np.hstack(views[:-1])
    window = f"Alignment: {name} - press q to continue"

    def open_window() -> None:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, canvas.shape[1], canvas.shape[0])
        cv2.imshow(window, canvas)

    open_window()
    while cv2.waitKey(100) & 0xFF != QUIT_KEY:
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            open_window()
    cv2.destroyWindow(window)
