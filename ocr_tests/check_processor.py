"""Check how the local TrOCR processors prepare an image, fully offline.

Run from the project root:  python ocr_tests/check_processor.py

Prints, per model in models/: the image processor and tokenizer classes, the configured size,
and the pixel_values shape produced for a wide 60x600 line image (shows whether the aspect
ratio is kept), plus whether generation can return per-token scores for a confidence value.
Model docs: https://huggingface.co/docs/transformers/model_doc/trocr
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from transformers import GenerationMixin, TrOCRProcessor  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

for name in ("trocr-small-handwritten", "trocr-base-handwritten"):
    processor = TrOCRProcessor.from_pretrained(MODELS_DIR / name, local_files_only=True)
    wide = np.full((60, 600, 3), 255, np.uint8)
    pixel_values = processor(images=wide, return_tensors="pt").pixel_values
    size = processor.image_processor.size
    print(f"{name}: {type(processor.image_processor).__name__}, {type(processor.tokenizer).__name__}, "
          f"size {size.height}x{size.width}, 60x600 input -> pixel_values {tuple(pixel_values.shape)}")
print("GenerationMixin.compute_transition_scores available:", hasattr(GenerationMixin, "compute_transition_scores"))
