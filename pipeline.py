"""NAMI Basics Participant Evaluation reader (dev_plan.txt, section 3). Runs offline.

Usage:  python pipeline.py [folder | image ...]      (default: the input/ folder)

The output folder is emptied at the start of every run.

Implemented so far:
  Step 1  collect and validate the input images
  Step 2  identify which form page each image shows (fatal on any structural problem)
  Step 3  align each image to its reference page; save the aligned image and its validity mask
"""

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
PAGE_TYPES = ("page1", "page2", "page3")  # folders under templates/
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
HEIC_EXTENSIONS = {".heic", ".heif"}
# Test photos: correct page 68-138 inliers, wrong page at most 6 (pages 1 and 2 included).
MIN_PAGE_INLIERS = 25


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


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except PipelineError as error:
        sys.exit(f"Error: {error}")
