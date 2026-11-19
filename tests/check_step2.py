"""Checks for pipeline steps 1-2 (input collection, page identification) on the test photos.

Run from the project root:  python tests/check_step2.py
Each case builds its inputs in a temporary folder and checks the result or the error raised.
"""

import re
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pipeline  # noqa: E402

PHOTOS = {"page1": ROOT / "dev_images" / "test_read_page1.jpg",
          "page2": ROOT / "dev_images" / "test_read_page2.jpg",
          "page3": ROOT / "dev_images" / "test_read_page3.jpg"}


def build(folder: Path, files: dict) -> None:
    for name, src in files.items():
        if isinstance(src, Path):
            shutil.copy(src, folder / name)
        elif isinstance(src, bytes):
            (folder / name).write_bytes(src)
        else:
            cv2.imwrite(str(folder / name), src)


def run(files: dict, expect_error: str | None = None, expect_pages: dict | None = None) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        build(folder, files)
        try:
            pages, _ = pipeline.identify_pages(pipeline.collect_inputs([str(folder)]))
        except pipeline.PipelineError as error:
            if expect_error and re.search(expect_error, str(error)):
                return f"raised as expected: {error}"
            raise AssertionError(f"unexpected error: {error}")
        if expect_error:
            raise AssertionError(f"expected an error matching '{expect_error}', got {pages}")
        got = {page: path.name for page, path in pages.items()}
        assert got == expect_pages, f"wrong assignment {got}"
        return f"assigned {got}"


def main() -> None:
    noise = np.random.default_rng(0).integers(0, 256, (4096, 3072), dtype=np.uint8)
    blank = np.full((4096, 3072), 255, np.uint8)
    cases = {
        "shuffled names, correct assignment": dict(
            files={"a.jpg": PHOTOS["page3"], "b.jpg": PHOTOS["page1"], "c.jpg": PHOTOS["page2"]},
            expect_pages={"page1": "b.jpg", "page2": "c.jpg", "page3": "a.jpg"}),
        "missing page (2 images)": dict(
            files={"a.jpg": PHOTOS["page1"], "b.jpg": PHOTOS["page2"]}, expect_error=r"Expected 3 images"),
        "extra image (4 images)": dict(
            files={**{f"{p}.jpg": f for p, f in PHOTOS.items()}, "d.jpg": PHOTOS["page1"]},
            expect_error=r"Expected 3 images"),
        "duplicate page": dict(
            files={"a.jpg": PHOTOS["page1"], "b.jpg": PHOTOS["page1"], "c.jpg": PHOTOS["page3"]},
            expect_error=r"all show page1; no image for page2"),
        "unmatched image (noise)": dict(
            files={"a.jpg": PHOTOS["page1"], "b.jpg": PHOTOS["page2"], "c.png": noise},
            expect_error=r"'c.png' matches no form page"),
        "unmatched image (blank)": dict(
            files={"a.jpg": PHOTOS["page1"], "b.jpg": PHOTOS["page2"], "c.png": blank},
            expect_error=r"'c.png' matches no form page"),
        "HEIC rejected": dict(
            files={"a.jpg": PHOTOS["page1"], "b.jpg": PHOTOS["page2"], "c.heic": b"not decoded"},
            expect_error=r"HEIC"),
        "unsupported extension": dict(
            files={"a.jpg": PHOTOS["page1"], "b.jpg": PHOTOS["page2"], "c.txt": b"text"},
            expect_error=r"unsupported file type"),
        "undecodable image": dict(
            files={"a.jpg": PHOTOS["page1"], "b.jpg": PHOTOS["page2"], "c.jpg": b"not a jpeg"},
            expect_error=r"could not be decoded"),
    }
    failures = 0
    for name, case in cases.items():
        try:
            print(f"PASS  {name}: {run(**case)}")
        except AssertionError as error:
            failures += 1
            print(f"FAIL  {name}: {error}")
    try:
        pipeline.collect_inputs([str(ROOT / "no_such_folder")])
        print("FAIL  missing input path: no error")
        failures += 1
    except pipeline.PipelineError as error:
        print(f"PASS  missing input path: raised as expected: {error}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
