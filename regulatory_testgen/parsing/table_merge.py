"""Stitch multi-page tables back together in MinerU markdown.

When a table runs across a page boundary, MinerU emits it as two (or more)
independent ``<table>...</table>`` blocks — the continuation restarts on the next
page, usually repeating the column header, with the page's running header / page
number sitting in between. Downstream that reads as several small tables instead
of one, which fragments both the clause text and the extracted RegulationTable.

``merge_multipage_tables`` repairs the markdown *before parsing* so the clause,
the extracted table, and the graph all see a single table.

The merge is deliberately conservative — it only stitches a continuation onto its
predecessor when ALL of these hold:

  * the two ``<table>`` blocks are adjacent (only page furniture between them);
  * they have the same number of columns; and
  * the continuation's first row repeats the predecessor's header row verbatim.

The repeated-header signal is what separates a genuine page-spanning table (e.g.
a country-assessment list continued on the next page) from two distinct tables
that merely happen to sit next to each other (different column counts, or no
repeated header). It is case-sensitive on purpose: near-duplicate per-figure
tables that differ only in header capitalisation must stay separate.

Cell HTML (including LaTeX math, colspans) is preserved byte-for-byte; only the
predecessor's closing ``</table>``, the continuation's opening ``<table>`` tag,
its repeated header row, and the page furniture between them are dropped.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_OPEN_TAG_RE = re.compile(r"^\s*<table\b[^>]*>", re.IGNORECASE)
_CLOSE_TAG_RE = re.compile(r"</table>\s*$", re.IGNORECASE)
_FIRST_ROW_RE = re.compile(r"^\s*(?:<thead\b[^>]*>\s*)?<tr\b.*?</tr>", re.IGNORECASE | re.DOTALL)
_HEADING_RE = re.compile(r"^\s*#")                       # markdown heading — a real boundary
_CLAUSE_NUM_RE = re.compile(r"^\s*\d+(?:\.\d+)+\.?\s")   # "5.2.1 ..." — a new clause, not furniture

# A page-boundary gap is furniture-scale: at most this many short non-empty lines,
# none of which is a heading or a numbered-clause start.
_MAX_FURNITURE_LINES = 3
_MAX_FURNITURE_CHARS = 200


def _rows(table_html: str) -> list[list[str]]:
    """Normalised cell text per row (whitespace-collapsed, tags stripped)."""
    out: list[list[str]] = []
    for row in _ROW_RE.findall(table_html):
        cells = [_WS_RE.sub(" ", _TAG_RE.sub("", c)).strip() for c in _CELL_RE.findall(row)]
        out.append(cells)
    return out


def _ncols(rows: list[list[str]]) -> int:
    """Dominant (most common) column count across the rows."""
    if not rows:
        return 0
    counts: dict[int, int] = {}
    for r in rows:
        counts[len(r)] = counts.get(len(r), 0) + 1
    return max(counts, key=lambda k: (counts[k], k))


def _is_furniture_gap(text: str) -> bool:
    """True when the text between two tables is only page furniture (running
    header, document code, page number) — never real content that would separate
    two genuinely distinct tables."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    if len(lines) > _MAX_FURNITURE_LINES:
        return False
    if sum(len(ln.strip()) for ln in lines) > _MAX_FURNITURE_CHARS:
        return False
    for ln in lines:
        if _HEADING_RE.match(ln) or _CLAUSE_NUM_RE.match(ln):
            return False
    return True


def _strip_close(table_html: str) -> str:
    return _CLOSE_TAG_RE.sub("", table_html)


def _strip_open_and_header(table_html: str) -> str:
    """Drop the continuation's ``<table>`` opening tag and its repeated header
    row, keeping every remaining row and the closing ``</table>``."""
    t = _OPEN_TAG_RE.sub("", table_html, count=1)
    t = _FIRST_ROW_RE.sub("", t, count=1)
    return t


def merge_multipage_tables(markdown: str) -> tuple[str, int]:
    """Return ``(merged_markdown, n_merges)``.

    ``n_merges`` counts continuation blocks folded into a predecessor (a table
    spanning three pages yields two merges).
    """
    tables = list(_TABLE_RE.finditer(markdown))
    if len(tables) < 2:
        return markdown, 0

    out: list[str] = []
    pos = 0                          # markdown consumed up to here
    n_merges = 0
    open_tbl: str | None = None      # currently-held (possibly merged) table html
    open_header: list[str] = []      # its header row (unchanged by merges)
    open_ncols = 0

    for m in tables:
        pre = markdown[pos:m.start()]     # text between the last table and this one
        html = m.group(0)
        rows = _rows(html)
        header = rows[0] if rows else []
        ncols = _ncols(rows)

        if open_tbl is not None:
            continues = (
                ncols > 0
                and ncols == open_ncols
                and bool(header)
                and header == open_header
                and _is_furniture_gap(pre)   # `pre` is the page-boundary gap here
            )
            if continues:
                open_tbl = _strip_close(open_tbl) + _strip_open_and_header(html)
                n_merges += 1
                pos = m.end()
                continue
            out.append(open_tbl)             # distinct table → emit the held one
            out.append(pre)                  # and the text between them
        else:
            out.append(pre)                  # text before the first table

        open_tbl, open_header, open_ncols = html, header, ncols
        pos = m.end()

    if open_tbl is not None:
        out.append(open_tbl)
    out.append(markdown[pos:])               # tail after the last table
    return "".join(out), n_merges
