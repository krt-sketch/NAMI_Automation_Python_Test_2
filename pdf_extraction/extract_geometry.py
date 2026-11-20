"""Phase A: template geometry from the vector PDF (dev_plan.txt, section 2).

Reads dev_images/ref_full_document.pdf and templates/page<N>/reference.png. Writes, into
this directory only:
  geometry_page<N>.json               Likert cells, checkboxes, handwriting regions
  overlay_reference_page<N>.png       the geometry drawn on the blank reference
  overlay_aligned_test_page<N>.png    the geometry drawn on dev_images/test_align_page<N>.png

Coordinates are reference pixels, px = pt * 200 / 72 (as in test_aligner.py).
A rect is [x0, y0, x1, y1] with x1, y1 exclusive, so image[y0:y1, x0:x1] is its content.

PyMuPDF sources:
  https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_drawings
  https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_text
  https://pymupdf.readthedocs.io/en/latest/app1.html (rawdict: per-character bboxes)
"""

import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
PDF_PATH = ROOT / "dev_images" / "ref_full_document.pdf"
REFERENCE = ROOT / "templates" / "page{}" / "reference.png"
ALIGNED_TEST = ROOT / "dev_images" / "test_align_page{}.png"
SCALE = 200 / 72

LIKERT_COLUMNS = ["Strongly agree", "Agree", "Neutral", "Disagree", "Strongly disagree", "N/A"]
LIKERT_TABLES = {1: ["A", "B"], 2: ["C"]}  # table names per page, top to bottom
FREE_TEXT_PAGES = {2}  # pages whose numbered questions below the table have blank answer space
BOX_GLYPHS = {"", "", "□"}  # Wingdings boxes (page 3) and U+25A1 white square (page 2)
MULTI_SELECT = {"13"}  # the only question that allows more than one checked box
WRITE_IN_OTHER = {"15", "16", "17"}  # page 3 questions whose "Other" box has handwriting beside it
QUESTION_NAMES = {"Gender": "17. Gender"}  # the PDF prints this heading without its number

RULE_MAX_THICKNESS = 6  # px; black filled rectangles thinner than this are printed lines
INK = 128  # reference pixels darker than this are printed ink
PARAGRAPH_GAP = 5  # px between text lines of the same paragraph
CONTINUATION_X = 20  # px; a wrapped checkbox label starts this close to the label above it
CONTINUATION_GAP = 10  # px between a checkbox label line and its wrapped continuation
HEADING_X = 20  # px a column heading may start to the right of its checkboxes
HANDWRITING_ABOVE = 20  # px above a printed text line that handwriting on it may reach
HANDWRITING_BELOW = 15  # px below it
ANSWER_INSET = 5  # px kept clear of the question text around a blank answer area
WRITE_IN_GAP = 5  # px between a printed label or the next column and a write-in area


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    with pymupdf.open(PDF_PATH) as doc:
        for page in doc:
            number = page.number + 1
            reference = cv2.imread(str(REFERENCE).format(number), cv2.IMREAD_GRAYSCALE)
            size = (round(page.rect.height * SCALE), round(page.rect.width * SCALE))
            if reference is None or reference.shape != size:
                raise SystemExit(f"page {number}: reference.png is missing or not {size[1]}x{size[0]} px")
            geometry = page_geometry(page, number, reference)
            path = OUT_DIR / f"geometry_page{number}.json"
            path.write_text(json.dumps(geometry, indent=1, ensure_ascii=False), encoding="utf-8")
            report(geometry)
            draw_overlays(geometry, reference, number)


def page_geometry(page, number: int, reference: np.ndarray) -> dict:
    lines = text_lines(page)
    rules = black_rules(page)
    tables = likert_tables(lines, rules, LIKERT_TABLES.get(number, []))
    groups, box_lines = checkbox_groups(lines, reference)
    handwriting = underscore_fields(lines, reference)
    if number in FREE_TEXT_PAGES:
        bottom = max(row["cells"][0][3] for t in tables for row in t["rows"])
        handwriting += answer_areas(lines, bottom)
    handwriting += write_ins(groups, box_lines, rules)
    for h in handwriting:
        x0, y0, x1, y1 = h["rect"]
        h["printed_ink"] = round(float((reference[y0:y1, x0:x1] < INK).mean()), 4)
    return {
        "page": number,
        "source": PDF_PATH.relative_to(ROOT).as_posix(),
        "units": "reference pixels (pt * 200 / 72); rect = [x0, y0, x1, y1], x1 and y1 exclusive",
        "likert_tables": tables,
        "checkbox_groups": groups,
        "handwriting": handwriting,
    }


# --- PDF reading ----------------------------------------------------------------------------

def px(rect) -> list[float]:
    return [v * SCALE for v in rect]


def clean(text: str) -> str:
    return " ".join(text.split())


def text_lines(page) -> list[dict]:
    """Text lines as PyMuPDF splits them, with per-character boxes, sorted top to bottom."""
    out = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            chars = [(c["c"], px(c["bbox"])) for span in line["spans"] for c in span["chars"]]
            if chars:
                out.append({"text": "".join(c for c, _ in chars), "bbox": px(line["bbox"]), "chars": chars})
    return sorted(out, key=lambda l: (l["bbox"][1], l["bbox"][0]))


def black_rules(page) -> list[list[float]]:
    """Thin black filled rectangles: table borders and printed underlines."""
    out = []
    for d in page.get_drawings():
        x0, y0, x1, y1 = px(d["rect"])
        if d.get("fill") == (0.0, 0.0, 0.0) and min(x1 - x0, y1 - y0) < RULE_MAX_THICKNESS:
            out.append([x0, y0, x1, y1])
    return out


def bands(intervals) -> list[list[float]]:
    """Merge overlapping or touching (start, end) intervals."""
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1] + 0.5:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def inner(x0: float, y0: float, x1: float, y1: float) -> list[int]:
    """Whole pixels strictly inside a rect given in fractional pixels."""
    return [math.ceil(x0), math.ceil(y0), math.floor(x1), math.floor(y1)]


def outer(x0: float, y0: float, x1: float, y1: float) -> list[int]:
    return [math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)]


def center_y(line: dict) -> float:
    return (line["bbox"][1] + line["bbox"][3]) / 2


# --- Likert tables ----------------------------------------------------------------------------

def likert_tables(lines: list[dict], rules: list, names: list[str]) -> list[dict]:
    """Rows are the spaces between horizontal rules; a row with text in the label column is a
    question, an empty one is a header or a gap that ends the table."""
    if not names:
        return []
    rows_y = bands([(r[1], r[3]) for r in rules if r[2] - r[0] > r[3] - r[1]])
    cols_x = bands([(r[0], r[2]) for r in rules if r[3] - r[1] >= r[2] - r[0]])
    if len(cols_x) != len(LIKERT_COLUMNS) + 2:
        raise SystemExit(f"Expected {len(LIKERT_COLUMNS) + 2} vertical table lines, found {len(cols_x)}")
    label_x0, label_x1 = cols_x[0][1], cols_x[1][0]

    def inside(x0, y0, x1, y1):
        return [l for l in lines if l["bbox"][0] >= x0 and l["bbox"][2] <= x1 and y0 <= center_y(l) <= y1]

    tables, rows, header = [], [], None
    for (_, top), (bottom, _) in zip(rows_y, rows_y[1:]):
        label = clean(" ".join(l["text"] for l in inside(label_x0, top, label_x1, bottom)))
        cells = [inner(cols_x[j][1], top, cols_x[j + 1][0], bottom) for j in range(1, len(cols_x) - 1)]
        if label:
            if not rows:
                columns = [clean(" ".join(l["text"] for l in inside(c[0], header[1], c[2], header[3]))) for c in cells]
                if [c.lower() for c in columns] != [c.lower() for c in LIKERT_COLUMNS]:
                    raise SystemExit(f"Table header {columns} does not match {LIKERT_COLUMNS}")
            rows.append({"question": label, "cells": cells})
        else:
            if rows:
                tables.append(rows)
                rows = []
            header = cells[0]  # the last empty row before a table is its header
    if rows:
        tables.append(rows)
    if len(tables) != len(names):
        raise SystemExit(f"Expected {len(names)} Likert tables, found {len(tables)}")
    return [{"name": n, "columns": LIKERT_COLUMNS, "rows": r} for n, r in zip(names, tables)]


# --- Checkboxes -------------------------------------------------------------------------------

def checkbox_groups(lines: list[dict], reference: np.ndarray) -> tuple[list[dict], dict]:
    """Box positions come from the box glyphs in the PDF text, labels from the text after each
    glyph (plus wrapped lines below it), and the question from the text before the glyph on its
    line or else the nearest heading above it in the same column."""
    glyph_lines = [l for l in lines if any(c in BOX_GLYPHS for c, _ in l["chars"])]
    boxes = []
    for line in glyph_lines:
        chars = line["chars"]
        idx = [i for i, (c, _) in enumerate(chars) if c in BOX_GLYPHS]
        for k, i in enumerate(idx):
            end = idx[k + 1] if k + 1 < len(idx) else len(chars)
            label_chars = [ch for ch in chars[i + 1:end] if not ch[0].isspace()]
            boxes.append({
                "glyph": chars[i][1],
                "label": clean("".join(c for c, _ in chars[i + 1:end])),
                "label_x0": label_chars[0][1][0],
                "label_x1": label_chars[-1][1][2],
                "bottom": line["bbox"][3],
                "line": line,
                "prefix": clean("".join(c for c, _ in chars[:idx[0]])),
                "alone": len(idx) == 1,
            })

    # Wrapped labels: a line without a box that starts under a box's label, directly below it.
    continuation = []
    for line in lines:
        if line in glyph_lines:
            continue
        for b in boxes:
            if (b["alone"] and not b["prefix"] and abs(b["label_x0"] - line["bbox"][0]) <= CONTINUATION_X
                    and abs(line["bbox"][1] - b["bottom"]) <= CONTINUATION_GAP):
                b["label"] = clean(b["label"] + " " + line["text"])
                b["bottom"] = line["bbox"][3]
                continuation.append(line)
                break
    plain = [l for l in lines if l not in glyph_lines and l not in continuation]

    def paragraph_above(line: dict) -> list[dict]:
        """`line` plus the plain lines directly above it in the same paragraph, top first."""
        out = [line]
        while True:
            top = out[0]["bbox"]
            prev = [l for l in plain if top[1] - PARAGRAPH_GAP <= l["bbox"][3] <= top[1] + PARAGRAPH_GAP
                    and l["bbox"][0] < top[2] and top[0] < l["bbox"][2] and l is not out[0]]
            if not prev:
                return out
            out.insert(0, max(prev, key=lambda l: l["bbox"][3]))

    groups: dict[str, dict] = {}
    box_lines = {}
    for b in boxes:
        if b["prefix"]:
            above = paragraph_above(b["line"])[:-1]
            question = clean(" ".join([l["text"] for l in above] + [b["prefix"]]))
        else:
            gx = b["glyph"][0]
            heads = [l for l in plain if l["bbox"][3] <= b["glyph"][1] + PARAGRAPH_GAP
                     and l["bbox"][0] - HEADING_X <= gx <= l["bbox"][2]]
            if not heads:
                raise SystemExit(f"No question found above checkbox '{b['label']}'")
            question = clean(" ".join(l["text"] for l in paragraph_above(max(heads, key=lambda l: l["bbox"][3]))))
        question = QUESTION_NAMES.get(question, question)
        number = re.match(r"(\d+)\.", question)
        group = groups.setdefault(question, {
            "question": question,
            "number": number.group(1) if number else None,
            "multiple": bool(number and number.group(1) in MULTI_SELECT),
            "boxes": [],
        })
        group["boxes"].append({"label": b["label"], "rect": printed_square(reference, b["glyph"], b["label"])})
        box_lines[(question, b["label"])] = b
    return list(groups.values()), box_lines


def printed_square(reference: np.ndarray, glyph: list[float], label: str) -> list[int]:
    """The glyph box is the font's character cell; the printed square inside it is the largest
    connected ink component there.
    https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f
    """
    x0, y0, x1, y1 = outer(*glyph)
    x0, y0 = x0 - 2, y0 - 2
    roi = (reference[y0:y1 + 2, x0:x1 + 2] < INK).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(roi)
    if count < 2:
        raise SystemExit(f"No printed square found for checkbox '{label}'")
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(v) for v in stats[i, :4])
    if abs(w - h) > 3 or not 10 <= w <= 40:
        raise SystemExit(f"Checkbox '{label}': ink component {w}x{h} px is not a square")
    return [x0 + x, y0 + y, x0 + x + w, y0 + y + h]


# --- Handwriting regions ------------------------------------------------------------------------

def ink_row(reference: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> int:
    """The image row with the most ink between the bounds: a printed underline."""
    return y0 + int(np.argmax((reference[y0:y1, x0:x1] < INK).sum(axis=1)))


def underscore_fields(lines: list[dict], reference: np.ndarray) -> list[dict]:
    """Fill-in fields typed as runs of '_' after their question."""
    out = []
    for line in lines:
        chars = line["chars"]
        idx = [i for i, (c, _) in enumerate(chars) if c == "_"]
        if not idx:
            continue
        question = clean("".join(c for c, _ in chars[:idx[0]])).rstrip(" :")
        _, ly0, _, ly1 = line["bbox"]
        x0, x1 = chars[idx[0]][1][0], chars[idx[-1]][1][2]
        number = re.match(r"(\d+)\.", question)
        out.append({
            "question": question,
            "rect": outer(x0, ly0 - HANDWRITING_ABOVE, x1, ly1 + HANDWRITING_BELOW),
            "underline_y": ink_row(reference, *outer(x0, ly0, x1, ly1)),
            "optional": "(optional)" in question,
            "number": number.group(1) if number else None,
        })
    return out


def answer_areas(lines: list[dict], table_bottom: int) -> list[dict]:
    """Blank space after each numbered question below the Likert table, up to the next paragraph."""
    below = [l for l in lines if l["bbox"][1] > table_bottom]
    paragraphs = []
    for line in below:
        if paragraphs and line["bbox"][1] - paragraphs[-1][-1]["bbox"][3] <= PARAGRAPH_GAP:
            paragraphs[-1].append(line)
        else:
            paragraphs.append([line])
    right = max(l["bbox"][2] for l in below)
    out = []
    for para, nxt in zip(paragraphs, paragraphs[1:]):
        question = clean(" ".join(l["text"] for l in para))
        number = re.match(r"(\d+)\.", question)
        if not number:
            continue
        out.append({
            "question": question,
            "rect": inner(min(l["bbox"][0] for l in para), para[-1]["bbox"][3] + ANSWER_INSET,
                          right, nxt[0]["bbox"][1] - ANSWER_INSET),
            "underline_y": None,
            "optional": False,
            "number": number.group(1),
        })
    return out


def write_ins(groups: list[dict], box_lines: dict, rules: list) -> list[dict]:
    """Space to the right of an "Other" label, up to the next column of boxes. Only boxes that
    start their own line form columns; inline boxes (e.g. "14. ... Yes No") do not."""
    lefts = sorted(b["glyph"][0] for b in box_lines.values() if b["alone"] and not b["prefix"])
    out = []
    for g in groups:
        if g["number"] not in WRITE_IN_OTHER:
            continue
        b = box_lines[(g["question"], "Other")]
        x0 = b["label_x1"] + WRITE_IN_GAP
        x1 = next((x for x in lefts if x > b["label_x1"]), max(r[2] for r in rules)) - WRITE_IN_GAP
        _, ly0, _, ly1 = b["line"]["bbox"]
        rect = inner(x0, ly0 - HANDWRITING_ABOVE, x1, ly1 + HANDWRITING_BELOW)
        under = [r for r in rules if r[2] - r[0] > r[3] - r[1] and rect[1] <= r[1] < rect[3]
                 and r[0] < rect[2] and rect[0] < r[2]]
        out.append({
            "question": f"{g['question']} - Other (write-in)",
            "rect": rect,
            "underline_y": round((under[0][1] + under[0][3]) / 2) if under else None,
            "optional": True,
            "number": g["number"],
            "write_in_for": {"question": g["question"], "label": "Other"},
        })
    return out


# --- Checks -----------------------------------------------------------------------------------

def report(geometry: dict) -> None:
    print(f"\n== page {geometry['page']}")
    for t in geometry["likert_tables"]:
        print(f"Likert table {t['name']}: {len(t['rows'])} rows")
        for r in t["rows"]:
            heights = {c[3] - c[1] for c in r["cells"]}
            print(f"   {r['cells'][0][1]}-{r['cells'][0][3]} (h {heights})  {r['question']}")
    for g in geometry["checkbox_groups"]:
        sizes = {(b["rect"][2] - b["rect"][0], b["rect"][3] - b["rect"][1]) for b in g["boxes"]}
        print(f"[{'multi' if g['multiple'] else 'single'}] {g['question']}  sizes {sizes}")
        print("     " + " | ".join(b["label"] for b in g["boxes"]))
    for h in geometry["handwriting"]:
        print(f"handwriting {h['rect']} underline {h['underline_y']} ink {h['printed_ink']}"
              f"{' optional' if h['optional'] else ''}: {h['question']}")


def draw_overlays(geometry: dict, reference: np.ndarray, number: int) -> None:
    """Likert cells green, checkboxes red, handwriting regions blue, underlines magenta."""
    aligned = cv2.imread(str(ALIGNED_TEST).format(number), cv2.IMREAD_GRAYSCALE)
    for name, base in (("reference", reference), ("aligned_test", aligned)):
        if base is None:
            continue
        img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        for t in geometry["likert_tables"]:
            for r in t["rows"]:
                for c in r["cells"]:
                    cv2.rectangle(img, c[:2], (c[2] - 1, c[3] - 1), (0, 170, 0), 2)
        for g in geometry["checkbox_groups"]:
            for b in g["boxes"]:
                cv2.rectangle(img, b["rect"][:2], (b["rect"][2] - 1, b["rect"][3] - 1), (0, 0, 255), 1)
        for h in geometry["handwriting"]:
            x0, y0, x1, y1 = h["rect"]
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 0), 2)
            if h["underline_y"] is not None:
                cv2.line(img, (x0, h["underline_y"]), (x1 - 1, h["underline_y"]), (255, 0, 255), 1)
        cv2.imwrite(str(OUT_DIR / f"overlay_{name}_page{number}.png"), img)


if __name__ == "__main__":
    main()