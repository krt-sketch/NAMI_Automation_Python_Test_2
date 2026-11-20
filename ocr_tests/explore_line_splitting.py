"""Test splitting handwriting lines at word gaps before TrOCR reads them.

Run from the project root after `python pipeline.py` (reads output/aligned_<page>.png and
valid_<page>.png):  python ocr_tests/explore_line_splitting.py [GAP_FACTOR [MAX_ASPECT ...]]

Why: TrOCR resizes every input to 384x384 without keeping its shape (ocr_tests/check_processor.py),
so a long line is squeezed horizontally and stretched vertically. Cutting it into shorter pieces
and reading each piece separately keeps the letters closer to their real shape.

How: in each cleaned line crop (made exactly as the pipeline makes them), columns with no ink
are found. A run of empty columns at least GAP_FACTOR x line height wide is a word gap. Pieces are
then read with the pipeline's model and beam width and joined with spaces. Strategies:
  whole       the line as one image (what the pipeline does now)
  words       every word on its own
  max N       neighboring words grouped while the piece is at most N times wider than tall
              (default N = 4 and 8; lines already that short are not split)

Output: per region the expected text and each strategy's text, mean confidence (mean token
probability, averaged over pieces) and character error rate (CER); a summary; the empty-column
gaps found in every line; and output/line_splits/<page>_<region>.png showing each line with its
cut points in red (output/ is emptied by the next pipeline run).
This script changes nothing in the pipeline; it only calls its functions.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pipeline  # noqa: E402
from explore_handwriting import EXPECTED, cer  # noqa: E402  (same folder)
from ocr import HandwritingReader, line_image  # noqa: E402

GAP_FACTOR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25  # word gap: at least this x line height
MAX_ASPECTS = [float(a) for a in sys.argv[2:]] or [4.0, 8.0]
MIN_GAP_SHOWN = 5  # px; smaller empty-column runs are not listed
SPLITS_DIR = pipeline.OUTPUT_DIR / "line_splits"


def cleaned_lines(aligned: dict) -> list[tuple[int, int, dict, list[np.ndarray]]]:
    """(page number, region index, region, grayscale line crops) as pipeline.read_text makes them."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * pipeline.HANDWRITING_PRINTED_DILATE_PX + 1,) * 2)
    out = []
    for number, page in enumerate(pipeline.PAGE_TYPES, start=1):
        image, valid = aligned[page]
        reference = cv2.imread(str(pipeline.TEMPLATES_DIR / page / "reference.png"), cv2.IMREAD_GRAYSCALE)
        printed = cv2.dilate(pipeline.dark_mask(reference).astype(np.uint8), kernel) > 0
        norm = pipeline.normalize(image)
        for k, h in enumerate(pipeline.load_geometry(page)["handwriting"]):
            lines, _, _ = pipeline.region_lines(h, norm, valid, printed)
            out.append((number, k + 1, h, lines))
    return out


def empty_runs(gray: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) column ranges with no ink between the first and last inked column."""
    threshold = pipeline.DARK_THRESHOLD if pipeline.NORMALIZE_BACKGROUND else pipeline.DARK_THRESHOLD_RAW
    inked = np.flatnonzero((gray < threshold).any(axis=0))
    runs = []
    for a, b in zip(inked, inked[1:]):
        if b - a > 1:
            runs.append((int(a) + 1, int(b)))
    return runs


def words(gray: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """(word column spans, gaps used as word breaks)."""
    min_gap = GAP_FACTOR * gray.shape[0]
    gaps = [g for g in empty_runs(gray) if g[1] - g[0] >= min_gap]
    edges = [0] + [x for g in gaps for x in g] + [gray.shape[1]]
    return [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)], gaps


def group(spans: list[tuple[int, int]], height: int, max_aspect: float | None) -> list[tuple[int, int]]:
    """Merge neighboring word spans while a piece stays at most max_aspect x height wide
    (None: every word alone)."""
    if max_aspect is None:
        return spans
    pieces = [list(spans[0])]
    for a, b in spans[1:]:
        if b - pieces[-1][0] <= max_aspect * height:
            pieces[-1][1] = b
        else:
            pieces.append([a, b])
    return [tuple(p) for p in pieces]


def strategies() -> dict:
    out = {"whole": "whole", "words": None}
    out.update({f"max {a:g}": a for a in MAX_ASPECTS})
    return out


def read_region(reader, lines: list[np.ndarray], how) -> tuple[str | None, float | None]:
    texts, confidences = [], []
    for gray in lines:
        if how == "whole":
            pieces = [(0, gray.shape[1])]
        else:
            pieces = group(words(gray)[0], gray.shape[0], how)
        images = [line_image(gray[:, a:b], pipeline.CROP_MODE, pipeline.CROP_PAD_PX) for a, b in pieces]
        for text, confidence in reader.read(images):
            texts.append(text)
            confidences.append(confidence)
    if not texts:
        return None, None
    return " ".join(texts), float(np.mean(confidences))


def save_splits(number: int, k: int, lines: list[np.ndarray]) -> None:
    """Each line with word breaks in red; grouped piece edges (first MAX_ASPECT) in blue."""
    rows = []
    for gray in lines:
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        spans, gaps = words(gray)
        for a, b in gaps:
            img[:, a:b] = (img[:, a:b] * 0.6 + np.array([0, 0, 255]) * 0.4).astype(np.uint8)
        for a, _ in group(spans, gray.shape[0], MAX_ASPECTS[0])[1:]:
            cv2.line(img, (a, 0), (a, img.shape[0] - 1), (255, 0, 0), 2)
        rows.append(cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(200, 200, 200)))
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    cv2.imwrite(str(SPLITS_DIR / f"p{number}_r{k}.png"), np.vstack(rows))


def main() -> None:
    aligned = {}
    for page in pipeline.PAGE_TYPES:
        image = cv2.imread(str(pipeline.OUTPUT_DIR / f"aligned_{page}.png"), cv2.IMREAD_GRAYSCALE)
        valid = cv2.imread(str(pipeline.OUTPUT_DIR / f"valid_{page}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or valid is None:
            sys.exit(f"Missing output for {page}: run `python pipeline.py` first")
        aligned[page] = (image, valid > 0)
    SPLITS_DIR.mkdir(exist_ok=True)
    reader = HandwritingReader(pipeline.TROCR_MODEL_DIR, pipeline.OCR_MAX_NEW_TOKENS, pipeline.OCR_NUM_BEAMS)
    regions = cleaned_lines(aligned)
    print(f"Model {pipeline.TROCR_MODEL_DIR.name}, beam width {pipeline.OCR_NUM_BEAMS}, "
          f"word gap >= {GAP_FACTOR} x line height, crop mode {pipeline.CROP_MODE}")

    print("\n=== Lines and the gaps between inked columns (px; * = used as a word break) ===")
    for number, k, h, lines in regions:
        for i, gray in enumerate(lines):
            spans, gaps = words(gray)
            shown = [f"{b - a}{'*' if (a, b) in gaps else ''}" for a, b in empty_runs(gray) if b - a >= MIN_GAP_SHOWN]
            print(f"p{number} r{k} line {i + 1}: {gray.shape[1]}x{gray.shape[0]} px (aspect {gray.shape[1] / gray.shape[0]:.1f}), "
                  f"{len(spans)} words; gaps {', '.join(shown) or 'none'}  [{h['question'][:40]}]")
        if lines:
            save_splits(number, k, lines)

    results, times = {}, {}
    for name, how in strategies().items():
        start = time.perf_counter()
        results[name] = [read_region(reader, lines, how) for _, _, _, lines in regions]
        times[name] = time.perf_counter() - start

    print("\n=== Recognized text per region ===")
    for i, (number, k, h, lines) in enumerate(regions):
        if not lines:
            continue
        want = next(v for key, v in EXPECTED.items() if h["question"].startswith(key))
        print(f"\np{number} r{k} {h['question'][:70]}")
        print(f"   {'expected':<8} {want!r}")
        for name in results:
            text, confidence = results[name][i]
            conf = f"{confidence:.3f}" if confidence is not None else "-"
            print(f"   {name:<8} {text!r}  conf {conf}  CER {cer(text, want):.2f}")

    print("\n=== Summary (all 13 regions; blank regions count as correct when read as blank) ===")
    for name in results:
        errors, exact = [], 0
        for (number, k, h, _), (text, _) in zip(regions, results[name]):
            want = next(v for key, v in EXPECTED.items() if h["question"].startswith(key))
            errors.append(cer(text, want))
            exact += (text or "") == (want or "")
        print(f"   {name:<8} mean CER {np.mean(errors):.3f}, exact {exact}/{len(errors)}, time {times[name]:.1f} s")
    print(f"\nLine pictures with cut points: {SPLITS_DIR}")


if __name__ == "__main__":
    main()
