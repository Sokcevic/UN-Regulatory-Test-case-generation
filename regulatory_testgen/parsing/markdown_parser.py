from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

from regulatory_testgen.data_models.core import Clause
from regulatory_testgen.parsing.table_merge import merge_multipage_tables

HEADING_RE = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$")
NUMERIC_HEADING_RE = re.compile(r"^(?P<id>\d+(?:\.\d+)*)(?:\.)?(?:\s+(?P<tail>.+))?$")
BROKEN_NUMERIC_HEADING_RE = re.compile(r"^\.(?P<frag>\d+(?:\.\d+)*)(?:\.)?\s+(?P<tail>.+)$")
ANNEX_HEADING_RE = re.compile(
    r"^(Annex\s+\d+(?:\s*-\s*Appendix\s*\d+)?)(?:\s*[-:]\s*(.+))?$",
    flags=re.IGNORECASE,
)
SCENARIO_HEADING_RE = re.compile(r"^(Scenario\s+\d+)(?:\s*[-:]\s*(.+))?$", re.I)
HTML_TABLE_START_RE = re.compile(r"<table\b", flags=re.IGNORECASE)
HTML_TABLE_END_RE = re.compile(r"</table>", flags=re.IGNORECASE)
MARKDOWN_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
DOT_LEADER_RE = re.compile(r"\.{3,}\s*\d+\s*$")
PARAGRAPH_REFERENCE_RE = re.compile(
    r"\b(?:paragraphs?|paras?\.?)\s+"
    r"((?:\d+(?:\.\d+)*\.?\s*(?:(?:,|and|or|to|-)\s*)?)*)",
    flags=re.IGNORECASE,
)
ANNEX_REFERENCE_RE = re.compile(
    r"\b(Annex\s+\d+(?:\s*-\s*Appendix\s*\d+)?|Appendix\s+\d+\s+of\s+Annex\s+\d+)",
    flags=re.IGNORECASE,
)
CLAUSE_ID_RE = re.compile(r"\b\d+(?:\.\d+)*\.?\b")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

KNOWN_STRUCTURAL_HEADINGS = {
    "agreement",
    "contents",
    "annexes",
    "introduction",
    "application for approval",
    "approval",
    "specifications",
    "specific requirements",
    "warning indication",
    "test procedure",
    "test conditions",
    "vehicle conditions",
    "test targets",
    "modification of vehicle type and extension of approval",
    "conformity of production",
    "penalties for non-conformity of production",
    "names and addresses of the technical services responsible for conducting approval tests and of type approval authorities",
    "communication",
    "arrangements of approval marks",
}


@dataclass(frozen=True)
class ParsedHeading:
    clause_id: str
    title: str = ""
    body: str = ""


class ClauseBuilder:
    def __init__(
        self,
        *,
        clause_id: str,
        title: str,
        source: str,
        line_start: int,
        section_path: list[str],
        is_pseudo_clause: bool,
    ) -> None:
        self.clause_id = clause_id
        self.title = title
        self.source = source
        self.line_start = line_start
        self.line_end = line_start
        self.section_path = list(section_path)
        self.is_pseudo_clause = is_pseudo_clause
        self.lines: list[str] = []

    def add_line(self, line: str, line_no: int) -> None:
        if HTML_TABLE_START_RE.search(line) or MARKDOWN_TABLE_ROW_RE.match(line):
            self.lines.append(line.strip())
        else:
            self.lines.append(clean_text(line))
        self.line_end = max(self.line_end, line_no)

    def to_clause(self) -> Clause:
        text = "\n".join(self.lines).strip()
        return Clause(
            uid=make_clause_uid(self.clause_id, self.line_start, max(self.line_start, self.line_end)),
            clause_id=self.clause_id,
            title=self.title,
            text=text,
            source=self.source,
            line_start=self.line_start,
            line_end=max(self.line_start, self.line_end),
            section_path=self.section_path,
            references=sorted(set(find_references(text))),
            is_pseudo_clause=self.is_pseudo_clause,
            document_region=document_region(self.clause_id, self.title, self.section_path),
        )


def parse_markdown_clauses(markdown_path: str | Path) -> list[Clause]:
    """Parse a MinerU Markdown export into structural Clause objects.

    The parser keeps raw HTML/Markdown tables inside clause text and performs only
    structural cleanup. Semantic classification is deliberately left to later stages.
    """

    path = Path(markdown_path)
    source = path.as_posix()
    text = path.read_text(encoding="utf-8")

    # Stitch tables MinerU split across page boundaries back into one before we
    # slice into lines, so the clause, the extracted table, and the graph all see
    # a single table instead of several fragments.
    text, n_merged = merge_multipage_tables(text)
    if n_merged:
        logger.info("Merged %d multi-page table continuation(s) in %s", n_merged, source)
    lines = text.splitlines()

    clauses: list[Clause] = []
    heading_stack: list[str] = []
    current: ClauseBuilder | None = None
    last_numeric_clause_id: str | None = None
    pseudo_counts: dict[str, int] = {}
    seen_numeric = False
    in_html_table = False

    def finish_current() -> None:
        nonlocal current
        if current is None:
            return
        clause = current.to_clause()
        if clause.text or clause.title:
            clauses.append(clause)
        current = None

    def start_clause(
        *,
        clause_id: str,
        title: str,
        line_no: int,
        section_path: list[str],
        is_pseudo_clause: bool,
        body: str = "",
    ) -> None:
        nonlocal current, last_numeric_clause_id, seen_numeric
        finish_current()
        current = ClauseBuilder(
            clause_id=clause_id,
            title=title,
            source=source,
            line_start=line_no,
            section_path=section_path,
            is_pseudo_clause=is_pseudo_clause,
        )
        if body:
            current.add_line(body, line_no)
        if not is_pseudo_clause and is_numeric_clause_id(clause_id):
            last_numeric_clause_id = clause_id
            seen_numeric = True

    for index, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()

        if not stripped:
            if current is not None:
                current.add_line("", index)
            continue

        if HTML_TABLE_START_RE.search(stripped):
            if current is None:
                current = ClauseBuilder(
                    clause_id="front-matter",
                    title="Front matter",
                    source=source,
                    line_start=index,
                    section_path=list(heading_stack),
                    is_pseudo_clause=True,
                )
            in_html_table = True
            current.add_line(raw_line.rstrip(), index)
            if HTML_TABLE_END_RE.search(stripped):
                in_html_table = False
            continue

        if in_html_table:
            if current is not None:
                current.add_line(raw_line.rstrip(), index)
            if HTML_TABLE_END_RE.search(stripped):
                in_html_table = False
            continue

        if MARKDOWN_TABLE_ROW_RE.match(stripped):
            if current is None:
                current = ClauseBuilder(
                    clause_id="front-matter",
                    title="Front matter",
                    source=source,
                    line_start=index,
                    section_path=list(heading_stack),
                    is_pseudo_clause=True,
                )
            current.add_line(raw_line.rstrip(), index)
            continue

        if is_noise_line(stripped) or looks_like_toc_entry(stripped):
            continue

        parsed_heading = parse_clause_heading(stripped, last_numeric_clause_id)
        if parsed_heading is not None:
            clause_id = parsed_heading.clause_id
            title = parsed_heading.title
            if is_numeric_clause_id(clause_id):
                label = heading_label(clause_id, title)
                section_path = section_path_for_numeric(heading_stack, clause_id)
                heading_stack = section_path + [label]
            elif is_annex_id(clause_id):
                label = heading_label(clause_id, title)
                section_path = []
                heading_stack = [label]
            else:
                label = heading_label(clause_id, title)
                section_path = list(heading_stack)
                heading_stack = section_path + [label]
            start_clause(
                clause_id=clause_id,
                title=title,
                body=parsed_heading.body,
                line_no=index,
                section_path=section_path,
                is_pseudo_clause=False,
            )
            continue

        md_heading = HEADING_RE.match(stripped)
        if md_heading is not None:
            title = clean_text(md_heading.group("title"))
            if should_start_pseudo_heading(title, seen_numeric=seen_numeric, current=current):
                level = len(md_heading.group("marks"))
                heading_stack = heading_stack[: max(level - 1, 0)]
                heading_stack.append(title)
                base_id = pseudo_clause_id(title)
                pseudo_counts[base_id] = pseudo_counts.get(base_id, 0) + 1
                pseudo_id = base_id if pseudo_counts[base_id] == 1 else f"{base_id}-{pseudo_counts[base_id]}"
                start_clause(
                    clause_id=pseudo_id,
                    title=title,
                    line_no=index,
                    section_path=heading_stack[:-1],
                    is_pseudo_clause=True,
                )
            elif current is not None:
                # Most non-numbered headings after the main body has started are table
                # titles or MinerU artifacts. Keep them as clause text.
                current.add_line(title, index)
            continue

        if current is None:
            current = ClauseBuilder(
                clause_id="front-matter",
                title="Front matter",
                source=source,
                line_start=index,
                section_path=list(heading_stack),
                is_pseudo_clause=True,
            )
        current.add_line(stripped, index)

    finish_current()
    return deduplicate_clauses(clauses)


def save_clauses(clauses: list[Clause], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([clause.model_dump() for clause in clauses], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_clauses(index_path: str | Path) -> list[Clause]:
    data = json.loads(Path(index_path).read_text(encoding="utf-8"))
    return [Clause.model_validate(item) for item in data]


def iter_table_blocks(text: str) -> Iterable[str]:
    for match in re.finditer(r"<table\b.*?</table>", text, flags=re.I | re.S):
        yield match.group(0)

    rows: list[str] = []
    for line in text.splitlines():
        if MARKDOWN_TABLE_ROW_RE.match(line):
            rows.append(line)
        elif rows:
            yield "\n".join(rows)
            rows = []
    if rows:
        yield "\n".join(rows)


def parse_clause_heading(line: str, last_numeric_clause_id: str | None) -> ParsedHeading | None:
    text = strip_markdown_heading(line)
    if not text or DOT_LEADER_RE.search(text) or looks_like_toc_entry(text):
        return None

    annex = ANNEX_HEADING_RE.match(text)
    if annex:
        return ParsedHeading(clause_id=clean_text(annex.group(1)), title=clean_text(annex.group(2) or ""))

    scenario = SCENARIO_HEADING_RE.match(text)
    if scenario:
        return ParsedHeading(clause_id=clean_text(scenario.group(1)), title=clean_text(scenario.group(2) or ""))

    broken = BROKEN_NUMERIC_HEADING_RE.match(text)
    if broken and last_numeric_clause_id:
        repaired_id = repair_broken_clause_id(broken.group("frag"), last_numeric_clause_id)
        tail = clean_text(broken.group("tail"))
        if looks_like_title(tail):
            return ParsedHeading(clause_id=repaired_id, title=tail)
        return ParsedHeading(clause_id=repaired_id, body=tail)

    numeric = NUMERIC_HEADING_RE.match(text)
    if not numeric:
        return None

    clause_id = numeric.group("id").rstrip(".")
    tail = clean_text(numeric.group("tail") or "")
    if has_leading_zero_top_level(clause_id):
        return None
    if not looks_like_clause_number(clause_id):
        return None
    if not tail:
        return ParsedHeading(clause_id=clause_id)
    if looks_like_title(tail):
        return ParsedHeading(clause_id=clause_id, title=tail)
    return ParsedHeading(clause_id=clause_id, body=tail)


def find_references(text: str) -> list[str]:
    refs: list[str] = []
    for match in PARAGRAPH_REFERENCE_RE.finditer(text):
        fragment = match.group(1)
        refs.extend(num.rstrip(".") for num in CLAUSE_ID_RE.findall(fragment))
    for match in ANNEX_REFERENCE_RE.finditer(text):
        refs.append(clean_text(match.group(1)))
    return [ref for ref in refs if ref]


def section_path_for_numeric(heading_stack: list[str], clause_id: str) -> list[str]:
    path: list[str] = []
    for heading in heading_stack:
        heading_id = heading_numeric_id(heading)
        if heading_id is None:
            # Keep structural non-numbered headings only when they are likely parents.
            if path or heading.lower() in KNOWN_STRUCTURAL_HEADINGS:
                path.append(heading)
            continue
        if clause_id != heading_id and clause_id.startswith(f"{heading_id}."):
            path.append(heading)
    return path


def heading_numeric_id(heading: str) -> str | None:
    match = re.match(r"^(\d+(?:\.\d+)*)\b", heading.strip())
    return match.group(1) if match else None


def heading_label(clause_id: str, title: str) -> str:
    title = clean_text(title)
    return f"{clause_id} {title}".strip()


def should_start_pseudo_heading(
    title: str,
    *,
    seen_numeric: bool,
    current: ClauseBuilder | None,
) -> bool:
    lower = clean_text(title).lower()
    if not seen_numeric:
        return True
    if lower.startswith("annex "):
        return True
    if lower in KNOWN_STRUCTURAL_HEADINGS:
        return True
    # Do not make table titles their own sections.
    if "table" in lower or "maximum" in lower or "subject vehicle test speed" in lower:
        return False
    # If there is no current clause, keep a structural placeholder.
    return current is None


def repair_broken_clause_id(fragment: str, previous_clause_id: str) -> str:
    prev_parts = previous_clause_id.split(".")
    frag_parts = fragment.split(".")
    best: tuple[int, list[str]] | None = None
    for start in range(len(prev_parts)):
        max_overlap = min(len(prev_parts) - start, len(frag_parts))
        for overlap in range(max_overlap, 0, -1):
            if prev_parts[start : start + overlap] == frag_parts[:overlap]:
                candidate = prev_parts[:start] + frag_parts
                if best is None or overlap > best[0]:
                    best = (overlap, candidate)
                break
    if best is not None:
        return ".".join(best[1])
    return f"{prev_parts[0]}.{fragment}"


def document_region(clause_id: str, title: str, section_path: list[str]) -> str:
    label = " ".join([clause_id, title, *section_path]).lower()
    top = clause_id.split(".", 1)[0].lower()
    if clause_id.lower().startswith("annex"):
        return "annex"
    if top == "1":
        return "scope"
    if top == "2":
        return "definitions"
    if top in {"3", "4", "7", "8", "9", "10", "11", "12"}:
        return "administrative"
    if top == "5":
        return "specifications"
    if top == "6":
        return "test_procedure"
    if "contents" in label:
        return "contents"
    return "front_matter" if not is_numeric_clause_id(clause_id) else "unknown"


def make_clause_uid(clause_id: str, line_start: int, line_end: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "-", clause_id).strip("-") or "clause"
    digest = hashlib.sha1(f"{clause_id}:{line_start}:{line_end}".encode("utf-8")).hexdigest()[:8]
    return f"clause-{safe}-{line_start}-{digest}"


def stable_clause_hash(clause: Clause) -> str:
    payload = "\n".join([clause.source, clause.clause_id, clause.title, normalize_for_hash(clause.text)])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def deduplicate_clauses(clauses: list[Clause]) -> list[Clause]:
    seen: set[tuple[str, int, int]] = set()
    result: list[Clause] = []
    for clause in clauses:
        key = (clause.clause_id, clause.line_start, clause.line_end)
        if key in seen:
            continue
        seen.add(key)
        result.append(clause)
    return result


def strip_markdown_heading(line: str) -> str:
    match = HEADING_RE.match(line)
    return clean_text(match.group("title")) if match else clean_text(line)


def clean_text(text: str) -> str:
    text = text.replace("\\_", "_")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def pseudo_clause_id(title: str) -> str:
    slug = "-".join(token.lower() for token in TOKEN_RE.findall(title))[:80]
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:8]
    return f"pseudo-{slug or 'section'}-{digest}"


def is_numeric_clause_id(clause_id: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)*", clause_id))


def is_annex_id(clause_id: str) -> bool:
    return clause_id.lower().startswith("annex")


def looks_like_clause_number(clause_id: str) -> bool:
    parts = clause_id.split(".")
    return all(part.isdigit() for part in parts) and 1 <= int(parts[0]) <= 99


def has_leading_zero_top_level(clause_id: str) -> bool:
    top = clause_id.split(".", 1)[0]
    return len(top) > 1 and top.startswith("0")


def looks_like_title(text: str) -> bool:
    lower = text.lower()
    if len(text) > 120:
        return False
    if re.search(r"\b(shall|must|may|when|if|there shall|the vehicle|the system)\b", lower):
        return False
    if text.endswith(":"):
        return True
    words = TOKEN_RE.findall(text)
    if not words:
        return False
    return len(words) <= 12


def looks_like_toc_entry(text: str) -> bool:
    cleaned = clean_text(strip_markdown_heading(text))
    if re.match(r"^\d+(?:\.\d+)*\.?\s+.+\.{2,}\s*\d+\s*$", cleaned):
        return True
    if re.match(r"^(?:Appendix\s+\d+|Annex\s+\d+)\s+.+\.{2,}\s*\d+\s*$", cleaned, re.I):
        return True
    return False


def is_noise_line(line: str) -> bool:
    if line in {"*", "#", "##"}:
        return True
    if re.fullmatch(r"\d+", line):
        return True
    if "<!--" in line and "-->" in line:
        return True
    return False


def recover_missing_numeric_parents(
    clauses: list[Clause],
    *,
    title_hints: dict[str, str] | None = None,
) -> list[Clause]:
    """Create Clause objects for numeric IDs implied by the hierarchy but absent from parsing.

    MinerU sometimes outputs section headings without their clause-number prefix.
    The parser then never creates a Clause for that level (e.g. '5.2' or '6.1').
    This function detects such gaps and inserts minimal Clause objects so the
    CONTAINS hierarchy is complete and those sections appear in the clause index.

    Title recovery order:
      1. Pseudo-clause that appears just before the first numbered child (≤50 lines)
      2. caller-supplied title_hints dict keyed by clause_id
      3. Empty title (the node exists but has no title)
    """
    numeric_ids: set[str] = {
        c.clause_id for c in clauses if is_numeric_clause_id(c.clause_id)
    }

    # Collect all implied-but-missing parents: {missing_id: first_child_id}
    implied: dict[str, str] = {}
    for cid in numeric_ids:
        if "." not in cid:
            continue
        parent = cid.rsplit(".", 1)[0]
        if parent not in numeric_ids and is_numeric_clause_id(parent):
            # Keep only the numerically smallest child (first encountered)
            if parent not in implied or cid < implied[parent]:
                implied[parent] = cid

    if not implied:
        return clauses

    hints = title_hints or {}
    effective_source = clauses[0].source if clauses else ""

    # Build positional lookup for fast nearest-pseudo search
    pseudo_clauses = [c for c in clauses if c.is_pseudo_clause]

    recovered: list[Clause] = []
    for missing_id, first_child_id in implied.items():
        first_child = next((c for c in clauses if c.clause_id == first_child_id), None)
        if first_child is None:
            continue

        # 1) Caller-supplied hint is authoritative (avoids misattributing a nearby
        #    pseudo-clause that belongs to a different section level)
        title = hints.get(missing_id, "")

        # 2) If no hint, look for a nearby pseudo-clause before the first child
        if not title:
            best_dist = float("inf")
            for pc in pseudo_clauses:
                if pc.line_start >= first_child.line_start:
                    continue
                dist = first_child.line_start - pc.line_end
                if dist > 50:
                    continue
                if pc.title and dist < best_dist:
                    best_dist = dist
                    title = pc.title

        inferred_line = max(0, first_child.line_start - 1)
        section_path = first_child.section_path  # same ancestors as the first child

        recovered.append(
            Clause(
                uid=make_clause_uid(missing_id, inferred_line, inferred_line),
                clause_id=missing_id,
                title=title,
                text="",
                source=effective_source,
                line_start=inferred_line,
                line_end=inferred_line,
                section_path=section_path,
                references=[],
                is_pseudo_clause=False,
                document_region=document_region(missing_id, title, section_path),
            )
        )
        logger.debug(
            "Recovered missing clause %s %r (inferred before child %s at line %d)",
            missing_id, title, first_child_id, first_child.line_start,
        )

    if not recovered:
        return clauses

    return sorted(clauses + recovered, key=lambda c: (c.line_start, c.clause_id))
