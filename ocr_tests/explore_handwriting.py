"""Compare TrOCR models and crop modes on the handwriting regions of the aligned test pages.

Run from the project root after `python pipeline.py` (reads output/aligned_<page>.png and
valid_<page>.png):  python ocr_tests/explore_handwriting.py [model ...]

Uses pipeline.read_text (step 6) with each model in models/ and each CROP_MODE, and prints
every region's text, its mean token probability per line, the character error rate (CER,
edit distance / expected length) against the user-confirmed readings, and the run time.
Line crops of the first configuration are saved to output/lines/ for inspection.
"""

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pipeline  # noqa: E402
from ocr import HandwritingReader  # noqa: E402

FINED = "I'm just here so I don't get fined."
EXPECTED = {  # question prefix: confirmed reading (None = blank)
    "State where": "Missouri", "Date you": "Nov. 6", "Location of": "St. Louis", "Please list": "Chamise",
    "1.": FINED, "2.": FINED, "3.": FINED, "4.": FINED, "5.": FINED,
    "21.": "No", "15.": None, "16.": None, "17.": None,
}


def cer(got: str | None, want: str | None) -> float:
    got, want = got or "", want or ""
    if not want:
        return 0.0 if not got else 1.0
    prev = list(range(len(want) + 1))
    for i, a in enumerate(got, 1):
        cur = [i]
        for j, b in enumerate(want, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / len(want)


def main() -> None:
    models = sys.argv[1:] or ["trocr-small-handwritten", "trocr-base-handwritten"]
    aligned = {}
    for page in pipeline.PAGE_TYPES:
        image = cv2.imread(str(pipeline.OUTPUT_DIR / f"aligned_{page}.png"), cv2.IMREAD_GRAYSCALE)
        valid = cv2.imread(str(pipeline.OUTPUT_DIR / f"valid_{page}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or valid is None:
            sys.exit(f"Missing output for {page}: run `python pipeline.py` first")
        aligned[page] = (image, valid > 0)

    first = True
    for model in models:
        start = time.perf_counter()
        reader = HandwritingReader(ROOT / "models" / model, pipeline.OCR_MAX_NEW_TOKENS)
        load = time.perf_counter() - start
        for mode in ("squash", "square"):
            pipeline.CROP_MODE = mode
            start = time.perf_counter()
            results = pipeline.read_text(aligned, reader, pipeline.OUTPUT_DIR if first else None)
            elapsed = time.perf_counter() - start
            first = False
            errors = []
            print(f"\n=== {model}, crop {mode}: load {load:.1f} s, read {elapsed:.1f} s")
            for r in results:
                want = next(v for k, v in EXPECTED.items() if r["question"].startswith(k))
                errors.append(cer(r["answer"], want))
                print(f"  p{r['page']} {r['question'][:28]:<28} CER {errors[-1]:4.2f}  conf {r['confidence']}  "
                      f"{r['answer']!r}  [{r['flag']}]")
            print(f"  mean CER {sum(errors) / len(errors):.3f}")


if __name__ == "__main__":
    main()
