"""Checks for pipeline steps 4-5 (Likert and checkbox reading) against the user-confirmed answers.

Run from the project root after `python pipeline.py` (uses output/aligned_<page>.png and
output/valid_<page>.png, so no re-alignment):  python tests/check_step4.py
"""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pipeline  # noqa: E402

LIKERT = {  # page: answers in row order
    1: ["Neutral", "Agree", "Strongly disagree", "Disagree", "Strongly agree", "N/A",
        "Agree", "Strongly disagree", "Neutral"],
    2: ["Disagree", "N/A", "Strongly agree", "Agree", "Disagree"],
}
CHECKBOXES = {  # question prefix: (checked labels, expected flag severity)
    "I am willing": (["No"], "low"),
    "13.": (["None"], "low"),
    "14.": (["Yes"], "low"),
    "15.": (["Student"], "none"),
    "16.": (["Other"], "low"),
    "17.": (["Non-binary"], "none"),
    "18.": (["Other"], "low"),
    "19.": (["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "85+"], "high"),
    "20.": (["Active Military", "Veteran", "Family Member (of military service member or veteran)", "None"], "high"),
    "22.": (["Urban"], "low"),
    "23.": ([], "high"),
    "24.": ([], "high"),
    "25.": (["Yes"], "none"),
    "26.": ([], "high"),
    "27.": (["Yes"], "none"),
}


def main() -> None:
    aligned = {}
    for page in pipeline.PAGE_TYPES:
        image = cv2.imread(str(pipeline.OUTPUT_DIR / f"aligned_{page}.png"), cv2.IMREAD_GRAYSCALE)
        valid = cv2.imread(str(pipeline.OUTPUT_DIR / f"valid_{page}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or valid is None:
            sys.exit(f"Missing output for {page}: run `python pipeline.py` first")
        aligned[page] = (image, valid > 0)
    results = pipeline.read_marks(aligned)

    failures = 0
    likert = [r for r in results if isinstance(r["answer"], str) or r["answer"] is None or "fractions" in r]
    for page, expected in LIKERT.items():
        got = [r for r in likert if r["page"] == page]
        for r, want in zip(got, expected):
            ok = r["answer"] == want and r["flag"] == "none"
            failures += not ok
            print(f"{'PASS' if ok else 'FAIL'}  p{page} {r['question'][:50]:<50} {r['answer']!s:<18} {r['flag']}")
        if len(got) != len(expected):
            failures += 1
            print(f"FAIL  page {page}: {len(got)} Likert rows, expected {len(expected)}")
    for r in results:
        if "fractions" in r:
            continue
        want, severity = next(v for k, v in CHECKBOXES.items() if r["question"].startswith(k))
        ok = r["answer"] == want and r["flag"].split(":")[0] == severity
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  p{r['page']} {r['question'][:50]:<50} {r['answer']!s:<18} {r['flag']}")
    failures += synthetic_likert_flags(aligned)
    sys.exit(1 if failures else 0)


def synthetic_likert_flags(aligned: dict) -> int:
    """The test photos have no double mark, blank row, or off-photo cell: make each on page 2."""
    image, valid = aligned["page2"][0].copy(), aligned["page2"][1].copy()
    rows = pipeline.load_geometry("page2")["likert_tables"][0]["rows"]
    x0, y0, x1, y1 = rows[0]["cells"][5]  # row 1 (Disagree): add an X in N/A
    cv2.line(image, (x0 + 10, y0 + 6), (x1 - 10, y1 - 6), 0, 3)
    cv2.line(image, (x0 + 10, y1 - 6), (x1 - 10, y0 + 6), 0, 3)
    for c in rows[1]["cells"]:  # row 2: erase the mark
        image[c[1]:c[3], c[0]:c[2]] = 255
    c = rows[2]["cells"][1]  # row 3: Agree cell partly off the photo
    valid[c[1]:c[3], c[0]:c[0] + 20] = False
    aligned = {**aligned, "page2": (image, valid)}
    got = [r for r in pipeline.read_marks(aligned) if r["page"] == 2 and "fractions" in r][:3]
    expected = [(["Disagree", "N/A"], "high: double mark"), (None, "none: blank"),
                ("Strongly agree", "high: out of frame: Agree")]
    failures = 0
    for name, r, (answer, flag) in zip(("double mark", "blank row", "out of frame"), got, expected):
        ok = sorted(r["answer"]) == sorted(answer) if isinstance(answer, list) else r["answer"] == answer
        ok = ok and r["flag"] == flag
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  synthetic {name}: {r['answer']} / {r['flag']}")
    return failures


if __name__ == "__main__":
    main()
