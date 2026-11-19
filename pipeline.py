"""NAMI Basics Participant Evaluation reader (dev_plan.txt, section 3). Runs offline.

Usage:  python pipeline.py [folder | image ...]      (default: the input/ folder)

The output folder is emptied at the start of every run.

Implemented so far:
  Step 1  collect and validate the input images
  Step 2  identify which form page each image shows (fatal on any structural problem)
  Step 3  align each image to its reference page; save the aligned image and its validity mask
  Step 4  read the Likert tables (pages 1 and 2)
  Step 5  read the checkboxes (pages 2 and 3)

Each answer carries a flag: "none", or "<severity>: <reason>[; <reason>...]" with severity
low or high. Detection problems are flagged and processing continues.
"""

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from test_aligner import TEMPLATES_DIR, Alignment, AlignmentError, align_page, match_page

ROOT = Path(__file__).resolve().parent

# --- Configuration (dev_plan.txt, section 4) --------------------------------------------------
DEFAULT_INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"  # emptied at the start of every run
PAGE_TYPES = ("page1", "page2", "page3")  # folders under templates/
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
HEIC_EXTENSIONS = {".heic", ".heif"}
# Test photos: correct page 68-138 inliers, wrong page at most 6 (pages 1 and 2 included).
MIN_PAGE_INLIERS = 25
GEOMETRY_DIR = ROOT / "pdf_extraction"  # geometry_page<N>.json from extract_geometry.py

# Dark pixels (values measured with alignment_tests/explore_likert.py and explore_checkboxes.py)
NORMALIZE_BACKGROUND = True  # divide by a blurred copy so shadows and lighting gradients are not ink
BACKGROUND_SIGMA = 25  # px, Gaussian blur for the background estimate
DARK_THRESHOLD = 170  # normalized value (paper = 255) below which a pixel is ink
DARK_THRESHOLD_RAW = 110  # used instead when NORMALIZE_BACKGROUND is False (test paper ~146-149)

# Likert (test photos: empty cells 0.000, marks 0.024-0.080 at a 12 px inset; 8 px let grid lines in)
LIKERT_COLUMNS = ("Strongly agree", "Agree", "Neutral", "Disagree", "Strongly disagree", "N/A")
LIKERT_MARGIN_PX = 12  # inset from the cell's grid lines
LIKERT_MIN_INNER_PX = 10  # the inset never leaves less than this (page 2's first row is 37 px tall)
MARK_MIN_FRACTION = 0.01  # dark fraction of the inset cell that counts as a mark
NEAR_TIE_RATIO = 0.5  # flag when the runner-up (below MARK_MIN_FRACTION) reaches this share of the top

# Checkboxes (test photos, inside the square: unchecked <= 0.027, checked >= 0.095)
CHECKBOX_INSET_PX = 4  # a box is read inside its printed square, inset so its outline is excluded
CHECK_MIN_DELTA = 0.05  # dark fraction above the blank reference's that counts as checked
CHECKBOX_PAD_PX = 4  # padded box: out-of-frame test and inner edge of the spillover ring
SPILLOVER_RING_PX = 15  # ring outside the padded box searched for marks that miss or overshoot the box
SPILLOVER_MIN_FRACTION = 0.02  # dark fraction of the ring (printed text excluded) that is flagged
PRINTED_DILATE_PX = 3  # reference ink grown by this much is ignored in the ring (label text)
CONDITIONAL_GROUPS = {"24": ("23", "Yes")}  # Q24 is answered only when Q23 is "Yes"


class PipelineError(Exception):
    """A fatal problem in steps 1-2: execution stops."""


# --- Step 1: input collection -------------------------------------------------------------------

def collect_inputs(args: list[str]) -> dict[Path, np.ndarray]:
    """Image paths from folders and/or files, each checked and decoded as grayscale.

    OpenCV's imread decodes by content, but HEIC is not among its formats:
    https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gacbaa02cffc4ec2422dfa2e24412a99e2
    """
    paths = []
    for arg in args or [str(DEFAULT_INPUT_DIR)]:
        path = Path(arg)
        if path.is_dir():
            paths += sorted(p for p in path.iterdir() if p.is_file())
        elif path.is_file():
            paths.append(path)
        else:
            raise PipelineError(f"Input not found: '{path}'")
    if not paths:
        raise PipelineError("No input images given")

    images = {}
    for path in paths:
        ext = path.suffix.lower()
        if ext in HEIC_EXTENSIONS:
            raise PipelineError(f"'{path.name}': HEIC/HEIF images cannot be read by OpenCV; convert to JPEG or PNG")
        if ext not in IMAGE_EXTENSIONS:
            raise PipelineError(f"'{path.name}': unsupported file type; allowed: {', '.join(sorted(IMAGE_EXTENSIONS))}")
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise PipelineError(f"'{path.name}' could not be decoded as an image")
        images[path] = image
    return images


# --- Step 2: page identification ----------------------------------------------------------------

def identify_pages(images: dict[Path, np.ndarray]) -> tuple[dict[str, Path], np.ndarray]:
    """Match every image against every reference page (homography stage only) and assign each
    image to the page with the most RANSAC inliers. Returns ({page type: path}, inlier matrix
    with one row per image and one column per page type)."""
    if len(images) != len(PAGE_TYPES):
        raise PipelineError(f"Expected {len(PAGE_TYPES)} images, one per page; got {len(images)}")
    references = {}
    for page in PAGE_TYPES:
        references[page] = cv2.imread(str(TEMPLATES_DIR / page / "reference.png"), cv2.IMREAD_GRAYSCALE)
        if references[page] is None:
            raise PipelineError(f"Missing reference image for {page}")

    paths = list(images)
    inliers = np.zeros((len(paths), len(PAGE_TYPES)), dtype=int)
    for i, path in enumerate(paths):
        for j, page in enumerate(PAGE_TYPES):
            try:
                inliers[i, j] = match_page(images[path], references[page], path.name, page).inliers
            except AlignmentError:
                pass  # too few features or matches: counts as 0 inliers

    best = inliers.argmax(axis=1)
    for i, path in enumerate(paths):
        if inliers[i, best[i]] < MIN_PAGE_INLIERS:
            raise PipelineError(f"'{path.name}' matches no form page (best: {inliers[i, best[i]]} inliers "
                                f"for {PAGE_TYPES[best[i]]}; at least {MIN_PAGE_INLIERS} needed)")
    for j, page in enumerate(PAGE_TYPES):
        same = [paths[i].name for i in np.flatnonzero(best == j)]
        if len(same) > 1:
            missing = [p for k, p in enumerate(PAGE_TYPES) if k not in best]
            raise PipelineError(f"{', '.join(same)} all show {page}; no image for {', '.join(missing)}")
    return {PAGE_TYPES[best[i]]: path for i, path in enumerate(paths)}, inliers


# --- Step 3: alignment --------------------------------------------------------------------------

def align_pages(pages: dict[str, Path], out_dir: Path) -> dict[str, Alignment]:
    """Full alignment (homography, then TPS) of each image to its identified reference page.
    Saves aligned_<page>.png and valid_<page>.png (white: pixel lies inside the photo) to out_dir."""
    aligned = {}
    for page, path in pages.items():
        try:
            result = align_page(path, page)
        except AlignmentError as error:
            raise PipelineError(f"Alignment of '{path.name}' to {page} failed: {error}") from error
        cv2.imwrite(str(out_dir / f"aligned_{page}.png"), result.image)
        cv2.imwrite(str(out_dir / f"valid_{page}.png"), result.valid * 255)
        aligned[page] = result
    return aligned


# --- Common helper: dark pixels -----------------------------------------------------------------

def dark_mask(gray: np.ndarray) -> np.ndarray:
    """True where a pixel is ink. With NORMALIZE_BACKGROUND the image is first divided by a
    heavily blurred copy of itself, so paper reads ~255 under any lighting.
    https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gae8bdcd9154ed5ca3cbc1766d960f45c1
    https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga6db555d30115642fedae0cda05604874
    """
    if not NORMALIZE_BACKGROUND:
        return gray < DARK_THRESHOLD_RAW
    background = cv2.GaussianBlur(gray, (0, 0), BACKGROUND_SIGMA)
    return cv2.divide(gray, np.maximum(background, 1), scale=255) < DARK_THRESHOLD


def dark_fraction(dark: np.ndarray, valid: np.ndarray, rect) -> tuple[float, bool]:
    """(dark pixels / region pixels inside the photo, whether any of the region is off the photo)."""
    x0, y0, x1, y1 = rect
    d, v = dark[y0:y1, x0:x1], valid[y0:y1, x0:x1]
    return (float(d[v].mean()) if v.any() else 0.0), bool((~v).any())


def make_flag(reasons: list[tuple[str, str]]) -> str:
    """'none', or the highest severity followed by every reason."""
    if not reasons:
        return "none"
    severity = "high" if any(s == "high" for s, _ in reasons) else "low"
    return f"{severity}: " + "; ".join(r for _, r in reasons)


def load_geometry(page: str) -> dict:
    return json.loads((GEOMETRY_DIR / f"geometry_{page}.json").read_text(encoding="utf-8"))


# --- Step 4: Likert tables ----------------------------------------------------------------------

def inset(rect, margin: int, min_inner: int) -> list[int]:
    """Shrink a rect by `margin` on each side, but never below `min_inner` px in either direction."""
    x0, y0, x1, y1 = rect
    mx = max(0, min(margin, (x1 - x0 - min_inner) // 2))
    my = max(0, min(margin, (y1 - y0 - min_inner) // 2))
    return [x0 + mx, y0 + my, x1 - mx, y1 - my]


def read_likert(geometry: dict, dark: np.ndarray, valid: np.ndarray, page_number: int) -> list[dict]:
    results = []
    for table in geometry["likert_tables"]:
        for row in table["rows"]:
            measured = [dark_fraction(dark, valid, inset(c, LIKERT_MARGIN_PX, LIKERT_MIN_INNER_PX)) for c in row["cells"]]
            fractions = [f for f, _ in measured]
            order = sorted(range(len(fractions)), key=lambda j: -fractions[j])
            marked = [j for j in order if fractions[j] >= MARK_MIN_FRACTION]
            reasons = []
            out = [LIKERT_COLUMNS[j] for j, (_, off) in enumerate(measured) if off]
            if out:
                reasons.append(("high", f"out of frame: {', '.join(out)}"))
            if not marked:
                answer = None
                reasons.append(("none", "blank"))
            elif len(marked) > 1:
                answer = [LIKERT_COLUMNS[j] for j in marked]
                reasons.append(("high", "double mark"))
            else:
                answer = LIKERT_COLUMNS[marked[0]]
                top, second = fractions[order[0]], fractions[order[1]]
                if second >= NEAR_TIE_RATIO * top:
                    reasons.append(("low", f"near-tie with {LIKERT_COLUMNS[order[1]]}"))
            flag = make_flag([r for r in reasons if r[0] != "none"])
            if flag == "none" and answer is None:
                flag = "none: blank"
            results.append({"page": page_number, "question": row["question"], "answer": answer, "flag": flag,
                            "fractions": [round(f, 3) for f in fractions]})
    return results


# --- Step 5: checkboxes -------------------------------------------------------------------------

def read_checkboxes(geometry: dict, dark: np.ndarray, valid: np.ndarray, ref_dark: np.ndarray,
                    page_number: int) -> list[dict]:
    """One result per question: every checked label, plus flags for boxes off the photo, marks
    outside the boxes, and a number of checks other than expected."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * PRINTED_DILATE_PX + 1,) * 2)
    printed = cv2.dilate(ref_dark.astype(np.uint8), kernel) > 0
    results = []
    for group in geometry["checkbox_groups"]:
        checked, out, spill, scores = [], [], [], []
        for box in group["boxes"]:
            x0, y0, x1, y1 = box["rect"]
            inner = inset(box["rect"], CHECKBOX_INSET_PX, 1)
            fraction, _ = dark_fraction(dark, valid, inner)
            baseline, _ = dark_fraction(ref_dark, np.ones_like(valid), inner)
            p = CHECKBOX_PAD_PX
            padded = (x0 - p, y0 - p, x1 + p, y1 + p)
            off = bool((~valid[padded[1]:padded[3], padded[0]:padded[2]]).any())
            r = p + SPILLOVER_RING_PX
            ring = np.zeros_like(valid)
            ring[y0 - r:y1 + r, x0 - r:x1 + r] = True
            ring[padded[1]:padded[3], padded[0]:padded[2]] = False
            ring &= valid & ~printed
            ring_fraction = float(dark[ring].mean()) if ring.any() else 0.0
            if off:
                out.append(box["label"])
            elif fraction - baseline >= CHECK_MIN_DELTA:
                checked.append(box["label"])
            if ring_fraction >= SPILLOVER_MIN_FRACTION:
                spill.append(box["label"])
            scores.append(round(fraction - baseline, 3))
        reasons = []
        if out:
            reasons.append(("high", f"out of frame: {', '.join(out)}"))
        if spill:
            # Ink beside the boxes of a question with nothing checked is likely a missed answer.
            severity = "low" if checked else "high"
            reasons.append((severity, f"mark outside box: {', '.join(spill)}"))
        if group["multiple"]:
            if not checked:
                reasons.append(("high", "no box checked"))
        elif group["number"] not in CONDITIONAL_GROUPS and len(checked) != 1:
            reasons.append(("high", f"{len(checked)} boxes checked, expected 1"))
        results.append({"page": page_number, "question": group["question"], "answer": checked,
                        "reasons": reasons, "number": group["number"], "scores": scores})
    return results


def apply_conditional_rules(results: list[dict]) -> None:
    """Q24 is expected only when Q23 is "Yes": flag Q24 if answered after "No" or blank after "Yes"."""
    by_number = {r["number"]: r for r in results if r.get("number")}
    for number, (parent, value) in CONDITIONAL_GROUPS.items():
        child, source = by_number.get(number), by_number.get(parent)
        if not child or not source or len(source["answer"]) != 1:
            continue  # parent unanswered or ambiguous: it is flagged itself
        expected = 1 if source["answer"] == [value] else 0
        if len(child["answer"]) != expected:
            reasons = {1: f"not answered although Q{parent} is {value}",
                       0: f"answered although Q{parent} is {source['answer'][0]}"}
            child["reasons"].append(("high", reasons[expected]))


def read_marks(aligned: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    """Steps 4-5 on {page type: (aligned grayscale image, validity mask)}: Likert rows first,
    then checkbox questions, each with its flag."""
    likert, boxes = [], []
    for number, page in enumerate(PAGE_TYPES, start=1):
        geometry = load_geometry(page)
        image, valid = aligned[page]
        dark = dark_mask(image)
        likert += read_likert(geometry, dark, valid, number)
        if geometry["checkbox_groups"]:
            reference = cv2.imread(str(TEMPLATES_DIR / page / "reference.png"), cv2.IMREAD_GRAYSCALE)
            boxes += read_checkboxes(geometry, dark, valid, dark_mask(reference), number)
    apply_conditional_rules(boxes)
    for r in boxes:
        r["flag"] = make_flag(r.pop("reasons"))
    return likert + boxes


def clean_output(folder: Path) -> None:
    """Create the output folder if needed and delete everything in it."""
    folder.mkdir(exist_ok=True)
    for item in folder.iterdir():
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()


def format_matrix(images: dict[Path, np.ndarray], inliers: np.ndarray) -> str:
    width = max(len(p.name) for p in images)
    rows = [" " * width + "".join(f"{p:>8}" for p in PAGE_TYPES)]
    rows += [f"{p.name:<{width}}" + "".join(f"{n:>8}" for n in row) for p, row in zip(images, inliers)]
    return "\n".join(rows)


def main(args: list[str]) -> None:
    clean_output(OUTPUT_DIR)
    images = collect_inputs(args)
    pages, inliers = identify_pages(images)
    print("RANSAC inliers (rows: images, columns: reference pages)")
    print(format_matrix(images, inliers))
    aligned = align_pages(pages, OUTPUT_DIR)
    for page in PAGE_TYPES:
        a = aligned[page]
        print(f"{page}: {pages[page].name}  matches {a.matches}, inliers {a.inliers}, "
              f"TPS control points {a.control_points}, inside photo {a.valid.mean():.1%}")

    results = read_marks({page: (a.image, a.valid > 0) for page, a in aligned.items()})
    for r in results:
        print(f"p{r['page']} {r['question'][:60]:<60} {str(r['answer']):<28} {r['flag']}")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except PipelineError as error:
        sys.exit(f"Error: {error}")
