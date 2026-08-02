"""
pdf_highlight.py — Trace clause IDs back to their location in the source PDF
and render a highlighted, trimmed-down copy for display.

MinerU's <name>_middle.json checkpoint gives, for every parsed text block,
the page index and bounding box (already in PDF point-space, matching
PyMuPDF's page coordinate system) alongside the block's text. We correlate
those blocks against the parsed Clause objects by walking both in document
order and matching on text containment — the two are derived from the same
linear reading order, so a monotonic cursor is enough.

Table content isn't matched (MinerU's table blocks carry HTML, not plain
text spans), so table-only clauses may not resolve to a precise page — the
clause's heading text usually still anchors it to the right page.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

Bbox = list[float]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _flatten_middle_blocks(middle_json_path: Path) -> list[dict[str, Any]]:
    """Ordered list of {page, bbox, text} for every text block in the document."""
    data = json.loads(middle_json_path.read_text(encoding="utf-8"))
    blocks: list[dict[str, Any]] = []
    for page in data.get("pdf_info", []):
        page_idx = page.get("page_idx", 0)
        for block in sorted(page.get("para_blocks", []), key=lambda b: b.get("index", 0)):
            parts = [
                span.get("content", "")
                for line in block.get("lines", [])
                for span in line.get("spans", [])
                if span.get("type") == "text"
            ]
            text = " ".join(p for p in parts if p)
            bbox = block.get("bbox")
            if not text.strip() or not bbox:
                continue
            blocks.append({"page": page_idx, "bbox": bbox, "text": text})
    return blocks


_MIN_MATCH_LEN = 15   # shorter targets/blocks false-match unrelated boilerplate too easily
_SEARCH_WINDOW = 50   # empirically the sweet spot for this correlation (see notes below)


def _contains(block: str, target: str) -> bool:
    """True if a normalized block and clause target correspond by containment in
    EITHER direction.

    The clause parser strips the leading number ("5.1.1.") into the clause_id,
    but MinerU keeps it in the block text — so a block reads
    "5.1.1. any vehicle fitted with an aebs…" while the clause target is
    "any vehicle fitted with an aebs…". Here the clause is a substring of the
    block (target ⊆ block); when the parser instead concatenates several blocks
    into one clause, the block is a substring of the clause (block ⊆ target).
    Both are real matches, so test both. The min-length guard on the shorter
    side keeps generic boilerplate from false-matching."""
    if not block or len(block) < _MIN_MATCH_LEN:
        return False
    if block in target:
        return True
    return len(target) >= _MIN_MATCH_LEN and target in block


def build_clause_page_map(
    clauses: list[Any], middle_json_path: Path, window: int = _SEARCH_WINDOW,
) -> dict[str, list[dict[str, Any]]]:
    """Correlate parsed clauses to {clause_id: [{page, bbox}, ...]}.

    Clauses and MinerU's text blocks both preserve document reading order, but
    many clauses (TOC entries, empty section headers) never match a block at
    all, so a strictly-monotonic "next block" cursor stalls on the first miss.
    Instead, each clause searches a forward window from the cursor for its
    anchor block, then greedily extends over any immediately-following blocks
    still contained in the clause's text (multi-paragraph clauses).

    Tuning note: a wider window sounds safer but isn't — one long/generic
    target coincidentally matching a distant, unrelated block yanks the cursor
    far forward and misaligns every clause after it for the rest of the
    document. A tight window bounds the damage from any single false match.
    Empirically (on UN R152) window=50 clearly beats 150+ on match count.

    Known limitation: numeric clause_ids repeat across annexes (each restarts
    its own "1.", "2.", ...). Like clause_map/clause_index elsewhere in this
    codebase, the last occurrence wins here too, so a citation for a reused ID
    always points at its final occurrence in the document.
    """
    if not middle_json_path.exists():
        return {}

    blocks = _flatten_middle_blocks(middle_json_path)
    norm_blocks = [_normalize(b["text"]) for b in blocks]

    clause_pages: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for clause in clauses:
        if getattr(clause, "is_pseudo_clause", False):
            continue
        target = _normalize(f"{clause.title} {clause.text}")
        if len(target) < _MIN_MATCH_LEN:
            continue

        anchor = None
        for j in range(cursor, min(len(blocks), cursor + window)):
            if _contains(norm_blocks[j], target):
                anchor = j
                break
        if anchor is None:
            continue  # no match in range; leave cursor alone and try the next clause

        matched = [anchor]
        j = anchor + 1
        while j < len(blocks) and norm_blocks[j] and norm_blocks[j] in target:
            matched.append(j)
            j += 1

        cursor = matched[-1] + 1
        clause_pages[clause.clause_id] = [
            {"page": blocks[i]["page"], "bbox": blocks[i]["bbox"]} for i in matched
        ]

    return clause_pages


def _boxes_by_page(
    clause_ids: list[str],
    clause_page_map: dict[str, list[dict[str, Any]]],
) -> dict[int, list[Bbox]]:
    """Collect {page_idx: [bbox, ...]} for every resolved region of clause_ids."""
    boxes_by_page: dict[int, list[Bbox]] = {}
    for cid in clause_ids:
        for entry in clause_page_map.get(cid, []):
            boxes_by_page.setdefault(entry["page"], []).append(entry["bbox"])
    return boxes_by_page


def render_highlighted_pdf(
    pdf_path: Path,
    clause_ids: list[str],
    clause_page_map: dict[str, list[dict[str, Any]]],
) -> tuple[bytes, list[int]] | None:
    """Build a trimmed copy of pdf_path containing only the pages referenced by
    clause_ids, with the matched regions highlighted (non-destructive overlay —
    the original text/layout is untouched, only annotations are added).

    Returns (pdf_bytes, original_page_numbers_1_indexed), or None if none of the
    clause_ids resolved to a page.
    """
    import fitz  # PyMuPDF

    boxes_by_page = _boxes_by_page(clause_ids, clause_page_map)
    if not boxes_by_page:
        return None

    src = fitz.open(str(pdf_path))
    pages_sorted = [p for p in sorted(boxes_by_page) if p < src.page_count]

    out = fitz.open()
    for new_idx, page_idx in enumerate(pages_sorted):
        out.insert_pdf(src, from_page=page_idx, to_page=page_idx)
        out_page = out[new_idx]
        for bbox in boxes_by_page[page_idx]:
            annot = out_page.add_highlight_annot(fitz.Rect(*bbox))
            annot.set_colors(stroke=(1, 0.85, 0))
            annot.update()

    pdf_bytes = out.tobytes()
    out.close()
    src.close()
    return pdf_bytes, [p + 1 for p in pages_sorted]


# ── Category-coded extraction overlay (debugging view) ──────────────────────

# Fallback highlight colours (RGB, 0–1) per category. The UI normally passes an
# explicit map derived from the graph palette so the legend matches exactly.
# Kept in sync with graph_viz.CATEGORY_COLOR (the single source of truth); the
# UI normally passes an explicit map derived from it, so this is only a fallback.
DEFAULT_CATEGORY_RGB: dict[str, tuple[float, float, float]] = {
    "obligation": (0.902, 0.098, 0.294),   # red      #e6194b
    "test_execution": (0.961, 0.510, 0.192),  # orange   #f58231
    "test_procedure": (1.000, 0.847, 0.694),  # apricot  #ffd8b1
    "performance_data": (0.941, 0.196, 0.902), # magenta  #f032e6
    "test_condition": (0.263, 0.388, 0.847),  # blue     #4363d8
    "test_setup": (0.259, 0.831, 0.957),   # cyan     #42d4f4
    "definition": (0.235, 0.706, 0.294),   # green    #3cb44b
    "scope": (0.749, 0.937, 0.271),        # lime     #bfef45
    "informative": (0.275, 0.600, 0.565),  # teal     #469990
    "unknown": (0.816, 0.804, 0.839),      # neutral grey #d0cdd6 — catch-all/meta
    "administrative": (0.353, 0.353, 0.549),  # indigo   #5a5a8c
    "formatting": (0.604, 0.388, 0.141),   # brown    #9a6324
    "table": (1.000, 0.882, 0.098),        # yellow   #ffe119 — extracted table region
    "_default": (0.663, 0.663, 0.663),     # grey     #a9a9a9
}


def hex_to_rgb01(h: str) -> tuple[float, float, float]:
    """'#e06666' -> (0.878, 0.4, 0.4)."""
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def category_boxes_by_page(
    clause_page_map: dict[str, list[dict[str, Any]]],
    category_of: dict[str, str],
    only_categories: set[str] | None = None,
) -> dict[int, list[tuple[Bbox, str]]]:
    """{page_idx: [(bbox, category), ...]} for every located clause region.

    category_of maps a clause_id (as keyed in clause_page_map) to its category.
    Pure/deterministic — unit-testable without a PDF."""
    by_page: dict[int, list[tuple[Bbox, str]]] = {}
    for cid, entries in clause_page_map.items():
        cat = category_of.get(cid, "unknown")
        if only_categories is not None and cat not in only_categories:
            continue
        for entry in entries:
            by_page.setdefault(entry["page"], []).append((entry["bbox"], cat))
    return by_page


def all_block_regions(middle_json_path: Path) -> list[dict[str, Any]]:
    """Every extracted block as {page, bbox, type} — text, title, list, index and
    table blocks (images/equations skipped). Tables carry only a top-level bbox
    (their text lives in nested sub-blocks), so they never text-match a clause;
    exposing them here lets the overlay still show the region."""
    if not middle_json_path.exists():
        return []
    data = json.loads(middle_json_path.read_text(encoding="utf-8"))
    keep = {"text", "title", "list", "index", "table"}
    out: list[dict[str, Any]] = []
    for page in data.get("pdf_info", []):
        pidx = page.get("page_idx", 0)
        for b in page.get("para_blocks", []):
            bbox = b.get("bbox")
            if bbox and b.get("type") in keep:
                out.append({"page": pidx, "bbox": bbox, "type": b.get("type")})
    return out


def build_extraction_overlay(
    clause_page_map: dict[str, list[dict[str, Any]]],
    category_of: dict[str, str],
    middle_json_path: Path,
    only_categories: set[str] | None = None,
) -> dict[int, list[tuple[Bbox, str]]]:
    """{page: [(bbox, category), ...]} covering the extracted blocks we can
    attribute. A block that a clause resolved to is coloured by that clause's
    category; a table block → "table". Blocks that MinerU extracted but that no
    classified clause resolved to are left unhighlighted (skipped) — they'd
    otherwise blanket the page in a meaningless "unmatched" colour."""
    matched: dict[tuple[int, tuple], str] = {}
    for cid, entries in clause_page_map.items():
        cat = category_of.get(cid, "unknown")
        for e in entries:
            matched[(e["page"], tuple(e["bbox"]))] = cat

    by_page: dict[int, list[tuple[Bbox, str]]] = {}
    for blk in all_block_regions(middle_json_path):
        cat = matched.get((blk["page"], tuple(blk["bbox"])))
        if cat is None:
            if blk["type"] != "table":
                continue  # extracted but not tied to a clause — leave unhighlighted
            cat = "table"
        if only_categories is not None and cat not in only_categories:
            continue
        by_page.setdefault(blk["page"], []).append((blk["bbox"], cat))
    return by_page


def pdf_page_count(pdf_path: Path) -> int:
    import fitz  # PyMuPDF
    doc = fitz.open(str(pdf_path))
    n = doc.page_count
    doc.close()
    return n


def render_boxes_page_images(
    pdf_path: Path,
    boxes_by_page: dict[int, list[tuple[Bbox, str]]],
    page_indices: list[int] | None = None,
    zoom: float = 2.0,
    category_rgb: dict[str, tuple[float, float, float]] | None = None,
) -> list[tuple[int, bytes]]:
    """Render pages (0-indexed; all if None) drawing each (bbox, category) box in
    its category colour. Source PDF untouched (highlights on an in-memory copy).
    Returns [(page_number_1_indexed, png_bytes), ...]."""
    import fitz  # PyMuPDF

    rgb = {**DEFAULT_CATEGORY_RGB, **(category_rgb or {})}
    default_color = rgb.get("_default", (1.0, 0.9, 0.4))

    src = fitz.open(str(pdf_path))
    if page_indices is None:
        page_indices = list(range(src.page_count))

    mat = fitz.Matrix(zoom, zoom)
    images: list[tuple[int, bytes]] = []
    for pidx in page_indices:
        if pidx < 0 or pidx >= src.page_count:
            continue
        page = src[pidx]
        for bbox, cat in boxes_by_page.get(pidx, []):
            annot = page.add_highlight_annot(fitz.Rect(*bbox))
            annot.set_colors(stroke=rgb.get(cat, default_color))
            annot.update()
        pix = page.get_pixmap(matrix=mat, annots=True)
        images.append((pidx + 1, pix.tobytes("png")))
    src.close()
    return images


def render_category_page_images(
    pdf_path: Path,
    clause_page_map: dict[str, list[dict[str, Any]]],
    category_of: dict[str, str],
    page_indices: list[int] | None = None,
    zoom: float = 2.0,
    category_rgb: dict[str, tuple[float, float, float]] | None = None,
    only_categories: set[str] | None = None,
) -> list[tuple[int, bytes]]:
    """Clause-only overlay: highlight just the text-matched clause regions,
    coloured by category. (Fuller coverage is build_extraction_overlay +
    render_boxes_page_images.)"""
    by_page = category_boxes_by_page(clause_page_map, category_of, only_categories)
    return render_boxes_page_images(pdf_path, by_page, page_indices, zoom, category_rgb)


def render_highlighted_page_images(
    pdf_path: Path,
    clause_ids: list[str],
    clause_page_map: dict[str, list[dict[str, Any]]],
    zoom: float = 2.0,
) -> list[tuple[int, bytes]]:
    """Render each cited page (with its regions highlighted) to a PNG image.

    This is the reliable way to *show* the highlighted source in Streamlit —
    unlike an embedded PDF, a rasterised page always renders regardless of
    browser PDF-plugin/sandbox behaviour. The PDF itself is never modified on
    disk; highlights are drawn onto a throwaway in-memory copy before
    rasterising. Returns [(original_page_number_1_indexed, png_bytes), ...],
    empty if nothing resolved.
    """
    import fitz  # PyMuPDF

    boxes_by_page = _boxes_by_page(clause_ids, clause_page_map)
    if not boxes_by_page:
        return []

    src = fitz.open(str(pdf_path))
    mat = fitz.Matrix(zoom, zoom)
    images: list[tuple[int, bytes]] = []
    for page_idx in sorted(boxes_by_page):
        if page_idx >= src.page_count:
            continue
        page = src[page_idx]
        for bbox in boxes_by_page[page_idx]:
            annot = page.add_highlight_annot(fitz.Rect(*bbox))
            annot.set_colors(stroke=(1, 0.85, 0))
            annot.update()
        pix = page.get_pixmap(matrix=mat, annots=True)
        images.append((page_idx + 1, pix.tobytes("png")))
    src.close()
    return images
