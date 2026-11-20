"""Handwriting recognition with TrOCR, loaded from a local folder and run fully offline.

TrOCR reads ONE text-line image per input and returns a string; it does not find text.
Its processors resize every image to 384x384 without keeping the aspect ratio
(checked with ocr_tests/check_processor.py).
  https://huggingface.co/docs/transformers/model_doc/trocr
  https://huggingface.co/microsoft/trocr-small-handwritten
Confidence is the mean probability of the generated tokens, from the generation scores:
  https://huggingface.co/docs/transformers/main_classes/text_generation
  (GenerationMixin.compute_transition_scores)
All settings are passed in by the caller; this module has no configuration of its own.
"""

import os

# No network access: the Hugging Face libraries must only read local files.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # noqa: E402


class HandwritingReader:
    def __init__(self, model_dir: Path, max_new_tokens: int, num_beams: int = 1, batch_size: int = 8):
        """num_beams 1 is greedy decoding (the most likely next token at every step); more than 1
        is beam search, which keeps that many partial transcriptions and returns the line that is
        most likely as a whole: https://huggingface.co/docs/transformers/generation_strategies"""
        if not model_dir.is_dir():
            raise FileNotFoundError(f"TrOCR model folder not found: '{model_dir}' (run download_models.py once)")
        self.processor = TrOCRProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir, local_files_only=True).eval()
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.batch_size = batch_size

    def read(self, lines: list[np.ndarray]) -> list[tuple[str, float]]:
        """(text, mean token probability) for each RGB line image."""
        results = []
        for start in range(0, len(lines), self.batch_size):
            batch = lines[start:start + self.batch_size]
            pixel_values = self.processor(images=batch, return_tensors="pt").pixel_values
            with torch.no_grad():
                out = self.model.generate(pixel_values, max_new_tokens=self.max_new_tokens, num_beams=self.num_beams,
                                          use_cache=True, output_scores=True, return_dict_in_generate=True)
            if self.num_beams > 1:
                # Beam scores are already log-probabilities; beam_indices trace each final line back
                # through the beams it came from.
                log_probs = self.model.compute_transition_scores(out.sequences, out.scores, out.beam_indices,
                                                                 normalize_logits=False)
            else:
                log_probs = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            tokens = out.sequences[:, 1:1 + log_probs.shape[1]]  # the first token is the decoder start token
            texts = self.processor.batch_decode(out.sequences, skip_special_tokens=True)
            for text, lp, tok in zip(texts, log_probs, tokens):
                generated = tok != self.model.generation_config.pad_token_id
                confidence = float(lp[generated].exp().mean()) if generated.any() else 0.0
                results.append((text.strip(), confidence))
        return results


def line_image(gray: np.ndarray, mode: str, pad: int) -> np.ndarray:
    """A grayscale line crop (white paper, dark ink) as the RGB image TrOCR expects.

    mode "squash": add `pad` px of white and let the processor stretch it to 384x384, as TrOCR's
    training lines were. mode "square": pad with white to a square first, so the ink keeps its
    aspect ratio (it then fills less of the 384x384 input).
    """
    img = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    if mode == "square":
        h, w = img.shape
        extra_h, extra_w = max(0, w - h), max(0, h - w)
        img = cv2.copyMakeBorder(img, extra_h // 2, extra_h - extra_h // 2, extra_w // 2, extra_w - extra_w // 2,
                                 cv2.BORDER_CONSTANT, value=255)
    elif mode != "squash":
        raise ValueError(f"Unknown crop mode '{mode}'")
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
