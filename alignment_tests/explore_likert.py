"""Probe for step 4: dark-pixel fraction of every Likert cell on the aligned test pages.

Run after `python pipeline.py` (reads output/aligned_page<N>.png and valid_page<N>.png):
    python alignment_tests/explore_likert.py [RAW_THRESHOLD NORM_THRESHOLD SIGMA [MARGIN]]

Each cell is inset by MARGIN px (default 8). Two ways of counting dark pixels are compared:
  raw   gray value below RAW_THRESHOLD
  norm  gray value divided by a Gaussian-blurred copy (sigma SIGMA px), scaled to 255,
        below NORM_THRESHOLD; this removes shadows and lighting gradients
        https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gae8bdcd9154ed5ca3cbc1766d960f45c1
Only pixels inside the photo (validity mask) are counted. '*' marks the expected answer.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRUTH = {(1, "A"): [2, 1, 4, 3, 0, 5], (1, "B"): [1, 4, 2], (2, "C"): [3, 5, 0, 1, 3]}  # column indices
RAW_T, NORM_T, SIGMA = (int(v) for v in sys.argv[1:4]) if len(sys.argv) > 3 else (110, 170, 25)
MARGIN = int(sys.argv[4]) if len(sys.argv) > 4 else 8

for p in (1, 2):
    g = json.loads((ROOT / "pdf_extraction" / f"geometry_page{p}.json").read_text(encoding="utf-8"))
    img = cv2.imread(str(ROOT / "output" / f"aligned_page{p}.png"), cv2.IMREAD_GRAYSCALE)
    valid = cv2.imread(str(ROOT / "output" / f"valid_page{p}.png"), cv2.IMREAD_GRAYSCALE) > 0
    bg = cv2.GaussianBlur(img, (0, 0), SIGMA)
    norm = cv2.divide(img, np.maximum(bg, 1), scale=255)
    print(f"page{p}: raw median {np.median(img[valid]):.0f} p5 {np.percentile(img[valid], 5):.0f} | "
          f"norm median {np.median(norm[valid]):.0f} p5 {np.percentile(norm[valid], 5):.0f}")
    for t in g["likert_tables"]:
        for r, row in enumerate(t["rows"]):
            exp = TRUTH[(p, t["name"])][r]
            parts = []
            for name, im, thr in ((f"raw<{RAW_T}", img, RAW_T), (f"norm<{NORM_T}", norm, NORM_T)):
                fr = []
                for x0, y0, x1, y1 in row["cells"]:
                    c = im[y0 + MARGIN:y1 - MARGIN, x0 + MARGIN:x1 - MARGIN]
                    v = valid[y0 + MARGIN:y1 - MARGIN, x0 + MARGIN:x1 - MARGIN]
                    fr.append((c[v] < thr).mean() if v.any() else float("nan"))
                parts.append(name + " " + " ".join(("*" if j == exp else " ") + f"{f:.3f}" for j, f in enumerate(fr)))
            print(f"  {t['name']}{r + 1}  " + "  |  ".join(parts))
