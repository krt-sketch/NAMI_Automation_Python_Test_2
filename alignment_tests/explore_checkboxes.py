"""Probe for step 5: dark-pixel measures of every checkbox on the aligned test pages 2 and 3.

Run after `python pipeline.py` (reads output/aligned_page<N>.png and valid_page<N>.png):
    python alignment_tests/explore_checkboxes.py [PAD RING NORM_THRESHOLD [INSET DILATE]]

For each box from pdf_extraction/geometry_page<N>.json:
  box    dark fraction of the printed square padded by PAD px (photo)
  base   the same on the blank reference (the printed square's own outline)
  delta  box - base
  inner  dark fraction inside the printed square, inset by INSET px (excludes the outline)
  masked dark fraction of the padded box, ignoring reference ink dilated by DILATE px
  ring   dark fraction of a RING px wide band just outside the padded box (spillover),
         ignoring reference ink dilated by DILATE px (label text next to the box)
  out    fraction of the padded box lying outside the photo
Dark = background-normalized gray (divided by a sigma-25 Gaussian blur) below NORM_THRESHOLD.
'*' marks boxes that are checked per the user-confirmed ground truth.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PAD, RING, NORM_T = (int(v) for v in sys.argv[1:4]) if len(sys.argv) > 3 else (4, 6, 170)
INSET, DILATE = (int(v) for v in sys.argv[4:6]) if len(sys.argv) > 5 else (4, 3)
SIGMA = 25
CHECKED = {
    "I am willing": {"No"},
    "13.": {"None"}, "14.": {"Yes"}, "15.": {"Student"}, "16.": {"Other"}, "17.": {"Non-binary"},
    "18.": {"Other"}, "19.": {"Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "85+"},
    "20.": {"Active Military", "Veteran", "Family Member (of military service member or veteran)", "None"},
    "22.": {"Urban"}, "23.": set(), "24.": set(), "25.": {"Yes"}, "26.": set(), "27.": {"Yes"},
}


def dark(gray: np.ndarray) -> np.ndarray:
    bg = cv2.GaussianBlur(gray, (0, 0), SIGMA)
    return cv2.divide(gray, np.maximum(bg, 1), scale=255) < NORM_T


def fraction(mask: np.ndarray, rect, valid: np.ndarray | None = None) -> float:
    x0, y0, x1, y1 = rect
    m = mask[y0:y1, x0:x1]
    if valid is None:
        return float(m.mean())
    v = valid[y0:y1, x0:x1]
    return float(m[v].mean()) if v.any() else float("nan")


for p in (2, 3):
    g = json.loads((ROOT / "pdf_extraction" / f"geometry_page{p}.json").read_text(encoding="utf-8"))
    img_dark = dark(cv2.imread(str(ROOT / "output" / f"aligned_page{p}.png"), cv2.IMREAD_GRAYSCALE))
    ref_dark = dark(cv2.imread(str(ROOT / "templates" / f"page{p}" / "reference.png"), cv2.IMREAD_GRAYSCALE))
    valid = cv2.imread(str(ROOT / "output" / f"valid_page{p}.png"), cv2.IMREAD_GRAYSCALE) > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * DILATE + 1, 2 * DILATE + 1))
    printed = cv2.dilate(ref_dark.astype(np.uint8), kernel) > 0
    print(f"== page{p}                {'box':>6} {'base':>6} {'delta':>6} {'inner':>6} {'masked':>6} {'ring':>6} {'out':>5}")
    for group in g["checkbox_groups"]:
        truth = next(v for k, v in CHECKED.items() if group["question"].startswith(k))
        print(f"{group['question'][:70]}")
        for b in group["boxes"]:
            x0, y0, x1, y1 = b["rect"]
            padded = (x0 - PAD, y0 - PAD, x1 + PAD, y1 + PAD)
            outer = (x0 - PAD - RING, y0 - PAD - RING, x1 + PAD + RING, y1 + PAD + RING)
            box = fraction(img_dark, padded, valid)
            base = fraction(ref_dark, padded)
            ring_mask = np.zeros_like(img_dark)
            ring_mask[outer[1]:outer[3], outer[0]:outer[2]] = True
            ring_mask[padded[1]:padded[3], padded[0]:padded[2]] = False
            ring_valid = ring_mask & valid & ~printed  # ignore printed label text next to the box
            ring = float(img_dark[ring_valid].mean()) if ring_valid.any() else float("nan")
            out = 1 - fraction(valid, padded)
            inner = fraction(img_dark, (x0 + INSET, y0 + INSET, x1 - INSET, y1 - INSET), valid)
            keep = ~printed[padded[1]:padded[3], padded[0]:padded[2]] & valid[padded[1]:padded[3], padded[0]:padded[2]]
            masked = float(img_dark[padded[1]:padded[3], padded[0]:padded[2]][keep].mean()) if keep.any() else float("nan")
            mark = "*" if b["label"] in truth else " "
            print(f"  {mark} {b['label'][:22]:<22} {box:6.3f} {base:6.3f} {box - base:6.3f} {inner:6.3f} {masked:6.3f} "
                  f"{ring:6.3f} {out:5.2f}")
