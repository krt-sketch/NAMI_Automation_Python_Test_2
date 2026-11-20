"""NAMI Basics Participant Evaluation reader (dev_plan.txt, section 3). Runs offline.

Usage:  python pipeline.py [folder | image ...]      (default: the input/ folder)

The output folder is emptied at the start of every run.

Implemented so far:
  Step 1  collect and validate the input images
  Step 2  identify which form page each image shows (fatal on any structural problem)
  Step 3  align each image to its reference page; save the aligned image and its validity mask
  Step 4  read the Likert tables (pages 1 and 2)
  Step 5  read the checkboxes (pages 2 and 3)
  Step 6  read the handwriting regions with TrOCR (ocr.py), line by line
  Step 7  write the results, in form order, to RESULTS_PATH as JSON

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
RESULTS_PATH = OUTPUT_DIR / "results.json"  # step 7 output
ROW_TOLERANCE_PX = 20  # result order: items whose tops are this close count as one row
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

# Handwriting (starting values; tuned with ocr_tests/explore_handwriting.py)
TROCR_MODEL_DIR = ROOT / "models" / "trocr-small-handwritten"  # local copy from download_models.py
OCR_MAX_NEW_TOKENS = 48
HANDWRITING_PRINTED_DILATE_PX = 3  # reference ink grown by this much is removed (inpainted) before reading
INPAINT_RADIUS_PX = 3  # neighborhood used to fill removed printed pixels
UNDERLINE_BAND_PX = 10  # leftover underline is searched this far above and below the printed one
UNDERLINE_MIN_RUN_PX = 80  # a horizontal ink run at least this long near the underline is removed
UNDERLINE_THRESHOLD = 230  # normalized value below which underline residue counts (lighter than ink)
LINE_MIN_INK_PX = 2  # a pixel row with at least this much ink belongs to a text line
LINE_MERGE_GAP_PX = 12  # text bands closer than this are one line (ascenders, descenders, dots)
LINE_MIN_HEIGHT_PX = 12  # thinner bands are noise
LINE_MIN_INK_FRACTION = 0.002  # a region with less ink than this share of its area is blank
CROP_PAD_PX = 12  # white margin around each line crop
CROP_MODE = "squash"  # "squash" (as TrOCR was trained) or "square"; see ocr.line_image
OCR_MIN_CONFIDENCE = 0.5  # mean token probability below this is flagged (to be tuned)
OUT_OF_FRAME_INK_PX = 20  # flag a region when ink comes this close to pixels off the photo


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

def normalize(gray: np.ndarray) -> np.ndarray:
    """With NORMALIZE_BACKGROUND: the image divided by a heavily blurred copy of itself, so
    paper reads ~255 under any lighting. Otherwise the image unchanged.
    https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gae8bdcd9154ed5ca3cbc1766d960f45c1
    https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga6db555d30115642fedae0cda05604874
    """
    if not NORMALIZE_BACKGROUND:
        return gray
    background = cv2.GaussianBlur(gray, (0, 0), BACKGROUND_SIGMA)
    return cv2.divide(gray, np.maximum(background, 1), scale=255)


def dark_mask(gray: np.ndarray) -> np.ndarray:
    """True where a pixel is ink."""
    return normalize(gray) < (DARK_THRESHOLD if NORMALIZE_BACKGROUND else DARK_THRESHOLD_RAW)


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
                            "position": (row["cells"][0][1], row["cells"][0][0]),
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
        first = group["boxes"][0]["rect"]
        results.append({"page": page_number, "question": group["question"], "answer": checked,
                        "reasons": reasons, "number": group["number"], "position": (first[1], first[0]),
                        "scores": scores})
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


# --- Step 6: handwriting ------------------------------------------------------------------------

def text_bands(ink: np.ndarray) -> list[tuple[int, int]]:
    """Text lines as (top, bottom) row ranges, from the horizontal projection of the ink."""
    rows = np.flatnonzero(ink.sum(axis=1) >= LINE_MIN_INK_PX)
    bands = []
    for r in rows:
        if bands and r - bands[-1][1] <= LINE_MERGE_GAP_PX:
            bands[-1][1] = r + 1
        else:
            bands.append([r, r + 1])
    return [(a, b) for a, b in bands if b - a >= LINE_MIN_HEIGHT_PX]


def region_lines(h: dict, norm: np.ndarray, valid: np.ndarray, printed: np.ndarray) -> tuple[list, float, bool]:
    """(line crops, ink fraction, ink near the photo edge) for one handwriting region.

    Printed content (reference ink, and any underline left over from misregistration, found
    with the lighter UNDERLINE_THRESHOLD so faint residue goes too) is filled in from its
    surroundings by inpainting, so handwriting that crosses an underline stays connected.
    Gray levels are kept: faint pen strokes have soft edges that thresholding would break.
    https://docs.opencv.org/4.x/d7/d8b/group__photo__inpaint.html
    """
    x0, y0, x1, y1 = h["rect"]
    threshold = DARK_THRESHOLD if NORMALIZE_BACKGROUND else DARK_THRESHOLD_RAW
    img = norm[y0:y1, x0:x1].copy()
    v = valid[y0:y1, x0:x1]
    removed = printed[y0:y1, x0:x1].copy()
    if h["underline_y"] is not None:
        b0 = max(0, h["underline_y"] - y0 - UNDERLINE_BAND_PX)
        b1 = h["underline_y"] - y0 + UNDERLINE_BAND_PX + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (UNDERLINE_MIN_RUN_PX, 1))
        faint = (img[b0:b1] < UNDERLINE_THRESHOLD).astype(np.uint8)
        removed[b0:b1] |= cv2.morphologyEx(faint, cv2.MORPH_OPEN, kernel) > 0
    img = cv2.inpaint(img, removed.astype(np.uint8), INPAINT_RADIUS_PX, cv2.INPAINT_TELEA)
    img[~v] = 255
    ink = img < threshold
    reach = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * OUT_OF_FRAME_INK_PX + 1,) * 2)
    near_edge = bool((cv2.dilate(ink.astype(np.uint8), reach).astype(bool) & ~v).any())
    fraction = float(ink.mean())
    if fraction < LINE_MIN_INK_FRACTION:
        return [], fraction, near_edge
    crops = []
    for top, bottom in text_bands(ink):
        cols = np.flatnonzero(ink[top:bottom].any(axis=0))
        crops.append(img[top:bottom, cols[0]:cols[-1] + 1])
    return crops, fraction, near_edge


def read_text(aligned: dict[str, tuple[np.ndarray, np.ndarray]], reader, out_dir: Path | None) -> list[dict]:
    """Step 6 on {page type: (aligned image, validity mask)}. Line crops are saved to
    out_dir/lines/ when out_dir is given."""
    from ocr import line_image  # imports torch and transformers; only needed for this step

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * HANDWRITING_PRINTED_DILATE_PX + 1,) * 2)
    if out_dir:
        (out_dir / "lines").mkdir(exist_ok=True)
    results = []
    for number, page in enumerate(PAGE_TYPES, start=1):
        geometry = load_geometry(page)
        image, valid = aligned[page]
        reference = cv2.imread(str(TEMPLATES_DIR / page / "reference.png"), cv2.IMREAD_GRAYSCALE)
        printed = cv2.dilate(dark_mask(reference).astype(np.uint8), kernel) > 0
        norm = normalize(image)
        regions = [region_lines(h, norm, valid, printed) for h in geometry["handwriting"]]
        crops = [line_image(c, CROP_MODE, CROP_PAD_PX) for lines, _, _ in regions for c in lines]
        texts = iter(reader.read(crops) if crops else [])
        crop_index = iter(range(len(crops)))
        for k, (h, (lines, fraction, near_edge)) in enumerate(zip(geometry["handwriting"], regions)):
            read = []
            for line in range(len(lines)):
                i = next(crop_index)
                read.append(next(texts))
                if out_dir:
                    cv2.imwrite(str(out_dir / "lines" / f"p{number}_r{k + 1}_l{line + 1}.png"), crops[i])
            reasons = []
            if near_edge:
                reasons.append(("high", "out of frame: writing reaches the photo edge"))
            low = [f"{t!r} {c:.2f}" for t, c in read if c < OCR_MIN_CONFIDENCE]
            if low:
                reasons.append(("low", f"low OCR confidence: {', '.join(low)}"))
            answer = " ".join(t for t, _ in read) or None
            flag = make_flag(reasons)
            if answer is None and flag == "none":
                flag = "none: blank"
            if "write_in_for" in h:  # listed right after its checkbox question
                group = next(g for g in geometry["checkbox_groups"] if g["question"] == h["write_in_for"]["question"])
                position = (group["boxes"][0]["rect"][1], group["boxes"][0]["rect"][0] + 0.5)
            else:
                position = (h["rect"][1], h["rect"][0])
            results.append({"page": number, "question": h["question"], "answer": answer, "flag": flag,
                            "position": position, "confidence": [round(c, 3) for _, c in read],
                            "ink": round(fraction, 4)})
    return results


# --- Step 7: JSON output ------------------------------------------------------------------------

def form_order(results: list[dict]) -> list[dict]:
    """Page by page, top to bottom, left to right. Items whose tops are within ROW_TOLERANCE_PX
    count as one row (the page 3 columns 15-20 start at slightly different heights)."""
    return sorted(results, key=lambda r: (r["page"], round(r["position"][0] / ROW_TOLERANCE_PX), r["position"][1]))


def write_results(results: list[dict], path: Path) -> None:
    """The minimal result file: page, question, answer (text, list of checked labels, or null), flag."""
    entries = [{k: r[k] for k in ("page", "question", "answer", "flag")} for r in form_order(results)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


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

    images = {page: (a.image, a.valid > 0) for page, a in aligned.items()}
    from ocr import HandwritingReader
    reader = HandwritingReader(TROCR_MODEL_DIR, OCR_MAX_NEW_TOKENS)
    results = read_marks(images) + read_text(images, reader, OUTPUT_DIR)
    for r in form_order(results):
        print(f"p{r['page']} {r['question'][:60]:<60} {str(r['answer']):<28} {r['flag']}")
    write_results(results, RESULTS_PATH)
    print(f"Results: {RESULTS_PATH}")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except PipelineError as error:
        sys.exit(f"Error: {error}")
