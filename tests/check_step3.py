"""Alignment check for pipeline step 3: leftover misalignment against the reference page.

Run from the project root after `python pipeline.py`:  python tests/check_step3.py

The aligned page is cut into PATCH x PATCH tiles. For each tile with enough printed ink in the
reference and lying fully inside the photo (validity mask), cv2.phaseCorrelate measures the
shift between the aligned tile and the reference tile:
https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html#ga552420a2ace9ef3fb053cd630fdb4952
The same is measured for the baseline aligner output in dev_images/test_align_page<N>.png.
"""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PATCH = 200  # px
MIN_INK = 0.02  # fraction of reference pixels darker than INK for a tile to be measured
INK = 128


def residuals(aligned: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> np.ndarray:
    shifts = []
    window = cv2.createHanningWindow((PATCH, PATCH), cv2.CV_64F)
    h, w = reference.shape
    for y in range(0, h - PATCH + 1, PATCH):
        for x in range(0, w - PATCH + 1, PATCH):
            ref = reference[y:y + PATCH, x:x + PATCH]
            if (ref < INK).mean() < MIN_INK or not valid[y:y + PATCH, x:x + PATCH].all():
                continue
            (dx, dy), _ = cv2.phaseCorrelate(ref.astype(np.float64), aligned[y:y + PATCH, x:x + PATCH].astype(np.float64), window)
            shifts.append(np.hypot(dx, dy))
    return np.array(shifts)


def main() -> None:
    print(f"{'':6} {'tiles':>5}  {'new: median / p95 / max px':>28}  {'baseline: median / p95 / max px':>32}")
    for p in (1, 2, 3):
        reference = cv2.imread(str(ROOT / "templates" / f"page{p}" / "reference.png"), cv2.IMREAD_GRAYSCALE)
        valid = cv2.imread(str(ROOT / "output" / f"valid_page{p}.png"), cv2.IMREAD_GRAYSCALE) > 0
        new = residuals(cv2.imread(str(ROOT / "output" / f"aligned_page{p}.png"), cv2.IMREAD_GRAYSCALE), reference, valid)
        old = residuals(cv2.imread(str(ROOT / "dev_images" / f"test_align_page{p}.png"), cv2.IMREAD_GRAYSCALE), reference, valid)
        fmt = lambda s: f"{np.median(s):.2f} / {np.percentile(s, 95):.2f} / {s.max():.2f}"
        print(f"page{p}  {len(new):>5}  {fmt(new):>28}  {fmt(old):>32}")


if __name__ == "__main__":
    main()
