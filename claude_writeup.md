# NAMI Form Reader: Exploration Write-up

This document records what was built, what was measured, and what is still open in the exploration
described in `dev_plan.txt`. The goal is to read photographed NAMI Basics Participant Evaluation
forms locally and offline: Likert answers, checkboxes, and handwriting, written to JSON.

**Main limitation:** every number below comes from **one filled-in document** (three phone photos
of one respondent's handwriting). The results show that the approach works. They do not show how
accurate it is in general.

## 1. Summary

| Part | Status on the test document |
|---|---|
| Page identification | Correct. Each photo scores 68-138 inlier matches against its own page and at most 6 against the others. |
| Alignment | Typical leftover misalignment 0.4 px; locally 5-7 px in page 1's lower table |
| Likert tables (14 rows) | 14/14 correct, no false flags |
| Checkboxes (15 questions, 59 boxes) | 15/15 correct. All expected rule and spillover flags fire. |
| Handwriting (13 regions) | Usable but weak: mean character error rate (CER) 0.069; 3 of 4 page 1 fields exact |
| Offline operation | Passes with every network connection blocked |
| Full run time | About 36 s on CPU (Windows 11, Python 3.14) |

## 2. How to run

```powershell
# One-time setup (Python 3.14 via the Python install manager)
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_models.py        # only network use: TrOCR weights into models/

# Read a form: three images, one per page, in any order and with any names
.\.venv\Scripts\python.exe pipeline.py               # default: the input/ folder
.\.venv\Scripts\python.exe pipeline.py path\to\folder
.\.venv\Scripts\python.exe pipeline.py page_a.jpg page_b.jpg page_c.jpg
```

The output is `output/results.json`. The `output/` folder is emptied at the start of every run. It
also holds the aligned pages, the validity masks, and the image pieces given to the handwriting model.

Checks (run from the project root; `pipeline.py` first, where noted):

| Script | Checks |
|---|---|
| `tests/check_step2.py` | Input checks and page identification, including every error case |
| `tests/check_step3.py` | Leftover misalignment per 200 px tile, compared with the earlier aligner output (after `pipeline.py`) |
| `tests/check_step4.py` | Likert and checkbox readings against the confirmed answers, plus synthetic double-mark, blank, and off-photo rows (after `pipeline.py`) |
| `tests/check_offline.py` | Blocks every network connection in the process, then runs the whole pipeline |

## 3. Pipeline

| Step | Code | What it does |
|---|---|---|
| 1. Inputs | `pipeline.collect_inputs` | Takes a folder or a list of files. Allowed: .jpg/.jpeg/.png/.tif/.tiff/.bmp. HEIC is rejected with a message. Each file must decode. |
| 2. Page identification | `pipeline.identify_pages`, `test_aligner.match_page` | Matches every image against all 3 reference pages (ORB features, RANSAC homography). The page with the most inlier matches wins. Fatal: not exactly 3 images, an image matching no page (< 25 inliers), two images on one page. |
| 3. Alignment | `test_aligner.align_page` | Homography, then thin-plate-spline correction for paper curl (constants unchanged). Also returns a validity mask of the pixels that came from inside the photo. |
| 4. Likert | `pipeline.read_likert` | Share of dark pixels in each cell, inset 12 px from the grid lines. |
| 5. Checkboxes | `pipeline.read_checkboxes` | Share of dark pixels inside each printed square, compared with the blank form, plus a ring around the box for marks that miss it. |
| 6. Handwriting | `pipeline.read_text`, `ocr.py` | Removes printed content, finds text lines, splits long lines, reads them with TrOCR. |
| 7. Output | `pipeline.write_results` | JSON in form order: `page`, `question`, `answer`, `flag`. |

Template geometry (cell, box, and handwriting-area positions, with question and label text) was taken from the vector PDF
by `pdf_extraction/extract_geometry.py`. That script also saves overlay images for checking the result by eye.

### Flags

Each answer has a flag: `none`, or `low:`/`high:` followed by every reason. Severity is set for this
exploration and is expected to change.

- **high:** a double mark (Likert); a box, cell, or writing off the photo; a number of checked boxes
  other than expected (only Q13 allows several); the Q23/Q24 rule (Q24 answered after Q23 = No, or
  blank after Q23 = Yes); ink beside the boxes of a question with nothing checked.
- **low:** a near-tie (Likert); ink beside a box of an answered question (spillover); a mean
  handwriting-model token probability below 0.5.
- **none: blank:** a skipped Likert row or empty handwriting area. Skipping is allowed, so it is not an error.

## 4. Results on the test document

### Page identification

Inlier matches (rows: photos; columns: reference pages):

|  | page1 | page2 | page3 |
|---|---|---|---|
| image1.jpg | 6 | **107** | 3 |
| image2.jpg | **138** | 6 | 5 |
| image3.jpg | 3 | 2 | **68** |

Pages 1 and 2 share the same Likert headers but are clearly separated. The cutoff of 25 is about 4x the
worst wrong-page score and well under the weakest right one. Page 3 scores lowest because its right edge
is cropped.

### Handwriting

| Region | Confirmed | Read |
|---|---|---|
| State | Missouri | Missouri |
| Date | Nov. 6 | Nov. 6 |
| Location | St. Louis | St. Louis . |
| Teacher | Chamise | Chamise |
| Page 2, Q1-Q5 | I'm just here so I don't get fined. | Q1-3, Q5: I'm just here so I don't get fined . / Q4: …I don't **set** fined . |
| Q21 zip | No | no |
| Q15-17 "Other" write-ins | (blank) | (blank) |

How the mean CER improved, step by step:

| Change | Mean CER |
|---|---|
| First version (printed lines whited out, which cut through the letters) | 0.202 |
| Printed content inpainted, gray levels kept | 0.107 |
| Beam search, width 2 (`ocr_tests/compare_decoding.py`) | 0.080 |
| Long lines split into pieces ≤ 8x as wide as tall (`ocr_tests/explore_line_splitting.py`) | 0.069 |

Other results:

- **Base model:** `trocr-base-handwritten` was 3x slower and not more accurate (`ocr_tests/explore_handwriting.py`).
- **Square padding:** padding crops to a square before the model's 384x384 resize was clearly worse than
  stretching them. Stretching is how the model was trained.

## 5. Decisions and changes from the plan

All are based on measurements. The probe scripts are kept in `alignment_tests/` and `ocr_tests/`.

- **Geometry:** the existing `templates/*/layout.json` was not enough. Page 3's checkboxes are Wingdings
  characters inside text lines, and page 1's answer lines are typed `_` characters. Per-character boxes
  from PyMuPDF (`rawdict`) solved both. The exact printed square comes from the reference PNG.
- **Likert inset 12 px, not 8:** at 8 px, grid lines leaked into page 1's lower table, where alignment is off
  5-7 px. At 12 px every empty cell measures 0.000 and every mark at least 0.027.
- **Checkboxes judged inside the printed square:** the plan's padded box minus a baseline did not separate
  checked from unchecked (+0.004 vs −0.003). Inside the square (inset 4 px): unchecked ≤ 0.027, checked ≥ 0.095.
- **Spillover ring 15 px, printed text ignored:** at 6 px it missed the marks drawn on Q24's labels. Without
  ignoring printed text, labels such as "Rural" counted as marks.
- **Background normalization on:** each image is divided by a blurred copy (sigma 25 px), and ink is
  normalized value < 170.
- **Output folder:** run results go to `output/`, which is emptied at the start of each run. The plan had
  a temporary folder deleted at the end.
- **OpenCV 5.1:** the newest OpenCV. Its aligned pages differ from the earlier baseline images, but the
  refactored aligner is bit-identical to the original code under the same OpenCV.
- **TrOCR settings:** `trocr-small-handwritten` (61.6 M parameters, 246 MB), beam width 2, line splitting at
  8:1. Beam widths 3-8 read the same as 2 but were slower.
- **Extra packages:** Pillow and sentencepiece are needed by transformers 5.x for these models.

## 6. Limitations and open issues

1. **One test document.** All thresholds and model choices were set on 5 long handwriting lines, 4 short
   fields, 14 Likert rows, and 59 boxes from one respondent with one pen. More forms (different people,
   pens, lighting, and scans as well as photos) are the most important next step.
2. **Handwriting is not reliable enough to use unchecked.** Wrong words still occur ("set fined"), and
   names and short fields depend heavily on how they are written.
3. **The confidence score does not catch wrong words.** It was 0.95 or higher on wrong readings such as
   "get tired". TrOCR's text decoder prefers common English, so it is confident about plausible wrong words.
   In the beam search tests the correct reading was usually the 2nd or 3rd candidate. The gap between the
   top two candidates is a possible review signal, but it is untested.
4. **Model license.** `trocr-base-handwritten` is marked MIT. `trocr-small-handwritten` states no license
   and is assumed MIT for this exploration. Confirm before any distribution.
5. **Grayscale only:** light-colored ink loses contrast.
6. **Local misalignment:** about 5-7 px in page 1's lower table with OpenCV 5.1. The detection margins cover
   this on the test document.
7. **Not read, as decided:** the free text beside Q25 and below Q27. Handwriting on page 3 is read only
   beside the "Other" boxes of Q15-17 and on the Q21 zip line.
8. **Offline check:** done by blocking network connections inside Python. A run with Windows networking
   turned off has not been done.
9. **Phone portability not evaluated:** the handwriting step uses PyTorch and transformers. Running it on a
   phone would need a model export or another runtime. Neither was tried.

## 7. Possible next steps

- Collect and label more forms, then re-check the thresholds and OCR settings with the existing probe scripts.
- Remove the space TrOCR puts before punctuation ("fined ." → "fined.").
- For structured fields, choose among beam candidates using known values: US state names, a date pattern,
  digits for the zip code.
- Test the gap between the top two beam candidates as a "needs review" flag.
- Evaluate a phone-ready runtime for the handwriting model.

## 8. Repository layout

| Path | Contents |
|---|---|
| `pipeline.py` | The pipeline, with all settings in one configuration block at the top |
| `test_aligner.py` | Page alignment (split into `match_page` and `align_page`; constants unchanged) |
| `ocr.py` | TrOCR wrapper: offline loading, beam search, confidence |
| `download_models.py` | One-time model download into `models/` (git-ignored) |
| `requirements.txt` | Tested package versions |
| `pdf_extraction/` | Geometry extraction script, `geometry_page<N>.json`, overlay images |
| `templates/page<N>/` | Blank reference images (plus the original `layout.json`) |
| `alignment_tests/`, `ocr_tests/` | Probe scripts used to measure thresholds and compare OCR options |
| `tests/` | Checks for steps 1-7 and offline operation |
| `dev_images/`, `input/` | Reference PDF and pages, the test photos, earlier aligner output |

Development history: one branch per step, `init_dev_step_1` … `init_dev_step_7`, each built on the previous one.
