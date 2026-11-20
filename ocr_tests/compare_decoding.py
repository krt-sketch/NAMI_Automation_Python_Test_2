"""Compare greedy decoding with beam search for TrOCR on the handwriting of the test document.

Run from the project root after `python pipeline.py` (reads output/aligned_<page>.png and
valid_<page>.png):  python ocr_tests/compare_decoding.py [BEAM_WIDTH ...]   (default: 4)

Greedy decoding picks the single most likely next token at every step. Beam search keeps the
BEAM_WIDTH most likely partial transcriptions at every step and returns the line that is most
likely as a whole. https://huggingface.co/docs/transformers/generation_strategies

Output:
  1. Model size (parameters in the image encoder and the text decoder).
  2. Per handwriting region: the expected text, then the text and confidence (mean token
     probability) from greedy decoding and from each beam width, with the character error
     rate (CER) and a marker where the beam result differs from greedy.
  3. Summary per method: mean CER, exact matches, time.
  4. Per line: the top 3 candidate lines beam search considered, each with its line score
     (exp of the length-normalized log probability, i.e. the geometric mean token probability).
Uses the pipeline's model (pipeline.TROCR_MODEL_DIR) and crop settings.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pipeline  # noqa: E402
from explore_handwriting import EXPECTED, cer  # noqa: E402  (same folder)
from ocr import HandwritingReader, line_image  # noqa: E402

CANDIDATES = 3


def load_aligned() -> dict:
    aligned = {}
    for page in pipeline.PAGE_TYPES:
        image = cv2.imread(str(pipeline.OUTPUT_DIR / f"aligned_{page}.png"), cv2.IMREAD_GRAYSCALE)
        valid = cv2.imread(str(pipeline.OUTPUT_DIR / f"valid_{page}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or valid is None:
            sys.exit(f"Missing output for {page}: run `python pipeline.py` first")
        aligned[page] = (image, valid > 0)
    return aligned


def line_crops(aligned: dict) -> list[tuple[str, np.ndarray]]:
    """(region question, RGB line image) for every handwriting line, as pipeline.read_text makes them."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * pipeline.HANDWRITING_PRINTED_DILATE_PX + 1,) * 2)
    out = []
    for page in pipeline.PAGE_TYPES:
        image, valid = aligned[page]
        reference = cv2.imread(str(pipeline.TEMPLATES_DIR / page / "reference.png"), cv2.IMREAD_GRAYSCALE)
        printed = cv2.dilate(pipeline.dark_mask(reference).astype(np.uint8), kernel) > 0
        norm = pipeline.normalize(image)
        for h in pipeline.load_geometry(page)["handwriting"]:
            lines, _, _ = pipeline.region_lines(h, norm, valid, printed)
            out += [(h["question"], line_image(c, pipeline.CROP_MODE, pipeline.CROP_PAD_PX)) for c in lines]
    return out


def main() -> None:
    widths = [int(w) for w in sys.argv[1:]] or [4]
    aligned = load_aligned()

    readers = {"greedy": HandwritingReader(pipeline.TROCR_MODEL_DIR, pipeline.OCR_MAX_NEW_TOKENS, 1)}
    for w in widths:
        readers[f"beam {w}"] = HandwritingReader(pipeline.TROCR_MODEL_DIR, pipeline.OCR_MAX_NEW_TOKENS, w)
    model = readers["greedy"].model
    count = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    print(f"Model {pipeline.TROCR_MODEL_DIR.name}: {count(model):.1f} M parameters "
          f"(image encoder {count(model.encoder):.1f} M, text decoder {count(model.decoder):.1f} M)")

    runs = {}
    for name, reader in readers.items():
        start = time.perf_counter()
        runs[name] = pipeline.read_text(aligned, reader, None)
        runs[name + " time"] = time.perf_counter() - start

    print("\n=== Recognized text per region (confidence = mean token probability) ===")
    for i, r in enumerate(runs["greedy"]):
        want = next(v for k, v in EXPECTED.items() if r["question"].startswith(k))
        print(f"\np{r['page']} {r['question'][:70]}")
        print(f"   {'expected':<9} {want!r}")
        for name in readers:
            got = runs[name][i]
            changed = "  <- differs from greedy" if name != "greedy" and got["answer"] != r["answer"] else ""
            print(f"   {name:<9} {got['answer']!r}  conf {got['confidence']}  CER {cer(got['answer'], want):.2f}{changed}")

    print("\n=== Summary ===")
    for name in readers:
        errors, exact = [], 0
        for r in runs[name]:
            want = next(v for k, v in EXPECTED.items() if r["question"].startswith(k))
            errors.append(cer(r["answer"], want))
            exact += (r["answer"] or "") == (want or "")
        print(f"   {name:<9} mean CER {np.mean(errors):.3f}, exact {exact}/{len(errors)}, "
              f"time {runs[name + ' time']:.1f} s")

    print(f"\n=== Top {CANDIDATES} candidates beam search considered (beam {widths[0]}) ===")
    reader = readers[f"beam {widths[0]}"]
    for question, img in line_crops(aligned):
        pixel_values = reader.processor(images=[img], return_tensors="pt").pixel_values
        with torch.no_grad():
            out = reader.model.generate(pixel_values, max_new_tokens=pipeline.OCR_MAX_NEW_TOKENS,
                                        num_beams=max(widths[0], CANDIDATES), num_return_sequences=CANDIDATES,
                                        use_cache=True, output_scores=True, return_dict_in_generate=True)
        texts = reader.processor.batch_decode(out.sequences, skip_special_tokens=True)
        print(f"\n{question[:70]}")
        for text, score in zip(texts, out.sequences_scores):
            print(f"   {float(score.exp()):.3f}  {text.strip()!r}")


if __name__ == "__main__":
    main()
