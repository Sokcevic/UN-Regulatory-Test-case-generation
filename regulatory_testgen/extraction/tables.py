from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from html.parser import HTMLParser

from regulatory_testgen.data_models.core import Clause, ReferenceLink
from regulatory_testgen.data_models.tables import RegulationTable

HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", flags=re.I | re.S)
MARKDOWN_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
MARKDOWN_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def extract_tables(clauses: list[Clause]) -> list[RegulationTable]:
    tables: list[RegulationTable] = []
    for clause in clauses:
        tables.extend(extract_tables_from_clause(clause))
    return tables


def extract_table_links(tables: list[RegulationTable]) -> list[ReferenceLink]:
    return [
        ReferenceLink(
            source_id=table.owner_clause_id,
            target_id=table.table_id,
            source_type="clause",
            target_type="table",
            relation="contains",
            text=table.title,
        )
        for table in tables
    ]


def extract_tables_from_clause(clause: Clause) -> list[RegulationTable]:
    result: list[RegulationTable] = []
    for index, match in enumerate(HTML_TABLE_RE.finditer(clause.text), start=1):
        raw_html = match.group(0)
        grid = parse_html_table(raw_html)
        table = table_from_grid(
            grid,
            owner_clause_id=clause.clause_id,
            table_index=index,
            title=table_title_before(clause.text[: match.start()]),
            raw_html=raw_html,
        )
        if table.rows:
            result.append(table)

    md_index = len(result) + 1
    for title, markdown_table in iter_markdown_tables(clause.text):
        grid = parse_markdown_table(markdown_table)
        table = table_from_grid(
            grid,
            owner_clause_id=clause.clause_id,
            table_index=md_index,
            title=title,
            raw_markdown=markdown_table,
        )
        if table.rows:
            result.append(table)
            md_index += 1
    return result


def table_from_grid(
    grid: list[list[str]],
    *,
    owner_clause_id: str,
    table_index: int,
    title: str,
    raw_html: str | None = None,
    raw_markdown: str | None = None,
) -> RegulationTable:
    if len(grid) < 2:
        headers: list[str] = []
        rows: list[dict[str, str]] = []
    else:
        headers, data_rows = split_headers(grid)
        headers = unique_headers(headers)
        rows = [row_to_dict(headers, row) for row in data_rows if any(cell.strip() for cell in row)]
    table_type = infer_table_type(title, owner_clause_id, headers)
    table_id = make_table_id(owner_clause_id, table_index, title)
    return RegulationTable(
        table_id=table_id,
        owner_clause_id=owner_clause_id,
        title=title,
        table_type=table_type,
        headers=headers,
        rows=rows,
        units=infer_units(title, headers),
        raw_html=raw_html,
        raw_markdown=raw_markdown,
        metadata={"table_index": table_index},
    )


def parse_html_table(html: str) -> list[list[str]]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return parse_html_table_fallback(html)

    soup = BeautifulSoup(html, "html.parser")
    grid: list[list[str]] = []
    rowspans: dict[int, tuple[str, int]] = {}

    for tr in soup.find_all("tr"):
        row: list[str] = []
        column = 0

        def fill_rowspans() -> None:
            nonlocal column
            while column in rowspans:
                value, remaining = rowspans[column]
                row.append(value)
                if remaining <= 1:
                    del rowspans[column]
                else:
                    rowspans[column] = (value, remaining - 1)
                column += 1

        fill_rowspans()
        for cell in tr.find_all(["td", "th"]):
            fill_rowspans()
            value = normalize_cell(cell.get_text(" "))
            rowspan = positive_int(cell.get("rowspan"), default=1)
            colspan = positive_int(cell.get("colspan"), default=1)
            for offset in range(colspan):
                row.append(value)
                if rowspan > 1:
                    rowspans[column + offset] = (value, rowspan - 1)
            column += colspan
        fill_rowspans()
        if any(row):
            grid.append(row)
    return normalize_grid_width(grid)


class SimpleTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.current_row: list[str] | None = None
        self.current_cell: list[str] | None = None
        self.in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "tr":
            self.current_row = []
        elif tag.lower() in {"td", "th"} and self.current_row is not None:
            self.current_cell = []
            self.in_cell = True

    def handle_data(self, data: str) -> None:
        if self.in_cell and self.current_cell is not None:
            self.current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self.current_row is not None and self.current_cell is not None:
            self.current_row.append(normalize_cell(" ".join(self.current_cell)))
            self.current_cell = None
            self.in_cell = False
        elif tag == "tr" and self.current_row is not None:
            if any(self.current_row):
                self.rows.append(self.current_row)
            self.current_row = None


def parse_html_table_fallback(html: str) -> list[list[str]]:
    parser = SimpleTableParser()
    parser.feed(html)
    return normalize_grid_width(parser.rows)


def iter_markdown_tables(text: str) -> Iterable[tuple[str, str]]:
    previous_lines: list[str] = []
    table_lines: list[str] = []

    def flush() -> tuple[str, str] | None:
        nonlocal table_lines
        if not table_lines:
            return None
        table_text = "\n".join(table_lines)
        table_lines = []
        return table_title_before("\n".join(previous_lines)), table_text

    for line in text.splitlines():
        if MARKDOWN_TABLE_ROW_RE.match(line):
            table_lines.append(line)
            continue
        item = flush()
        if item is not None:
            yield item
        previous_lines.append(line)
    item = flush()
    if item is not None:
        yield item


def parse_markdown_table(markdown_table: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in markdown_table.splitlines():
        if MARKDOWN_TABLE_SEPARATOR_RE.match(line):
            continue
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        row = [normalize_cell(cell) for cell in stripped.split("|")]
        if any(row):
            rows.append(row)
    return normalize_grid_width(rows)


def split_headers(grid: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    data_start = 1
    for index, row in enumerate(grid):
        if looks_like_data_row(row):
            data_start = max(index, 1)
            break
    header_rows = grid[:data_start]
    data_rows = grid[data_start:]
    headers: list[str] = []
    width = max((len(row) for row in grid), default=0)
    for column in range(width):
        parts = []
        for row in header_rows:
            if column < len(row) and row[column]:
                parts.append(row[column])
        headers.append(normalize_cell(" / ".join(dict.fromkeys(parts))))
    return headers, data_rows


def looks_like_data_row(row: list[str]) -> bool:
    non_empty = [cell for cell in row if cell]
    if not non_empty:
        return False
    numeric_like = sum(1 for cell in non_empty if re.search(r"[-+]?\d", cell))
    return numeric_like >= max(1, len(non_empty) // 2)


def row_to_dict(headers: list[str], row: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, header in enumerate(headers):
        key = header or f"column_{index + 1}"
        result[key] = row[index] if index < len(row) else ""
    return result


def unique_headers(headers: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for header in headers:
        base = header or "column"
        count = seen.get(base, 0) + 1
        seen[base] = count
        result.append(base if count == 1 else f"{base}_{count}")
    return result


def table_title_before(text_before: str) -> str:
    lines = [normalize_line(strip_markdown_heading(line)) for line in text_before.splitlines()]
    lines = [line for line in lines if line and not line.lower().startswith("all values")]
    for line in reversed(lines[-8:]):
        lower = line.lower()
        if any(token in lower for token in ["maximum", "subject vehicle test speed", "test speed", "impact speed", "scenario"]):
            return line
    return lines[-1] if lines else ""


def infer_table_type(title: str, owner_clause_id: str, headers: list[str]) -> str:
    lower = " ".join([title, owner_clause_id, *headers]).lower()
    if "subject vehicle test speed" in lower or owner_clause_id.startswith(("6.4", "6.5", "6.6", "6.7")):
        return "test_speed_matrix"
    if "maximum" in lower and "impact speed" in lower:
        return "performance_limit"
    if owner_clause_id.lower().startswith("annex 1") or "approval" in lower and "signature" in lower:
        return "form"
    return "unknown"


def infer_units(title: str, headers: list[str]) -> dict[str, str]:
    units: dict[str, str] = {}
    text = " ".join([title, *headers]).lower()
    if "km/h" in text or "k m / h" in text:
        units["speed"] = "km/h"
    if "m/s" in text or "m / s" in text:
        units["acceleration"] = "m/s^2"
    if "lux" in text:
        units["illumination"] = "lux"
    return units


def make_table_id(owner_clause_id: str, table_index: int, title: str) -> str:
    safe_owner = re.sub(r"[^A-Za-z0-9]+", "-", owner_clause_id).strip("-") or "clause"
    digest = hashlib.sha1(f"{owner_clause_id}\n{table_index}\n{title}".encode("utf-8")).hexdigest()[:8]
    return f"T-{safe_owner}-{table_index}-{digest}"


def normalize_cell(text: object) -> str:
    value = str(text or "")
    value = HTML_TAG_RE.sub(" ", value)
    value = value.replace("\xa0", " ")
    value = WHITESPACE_RE.sub(" ", value)
    return value.strip()


def normalize_line(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def strip_markdown_heading(line: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", line).strip()


def normalize_grid_width(grid: list[list[str]]) -> list[list[str]]:
    width = max((len(row) for row in grid), default=0)
    return [row + [""] * (width - len(row)) for row in grid]


def positive_int(value: object, *, default: int) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
