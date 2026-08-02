from __future__ import annotations

import hashlib
import re

from regulatory_testgen.data_models.core import Clause
from regulatory_testgen.data_models.requirements import Requirement
from regulatory_testgen.data_models.tables import RegulationTable

OBLIGATION_RE = re.compile(
    r"\b(shall(?:\s+not)?|must(?:\s+not)?|is required to|are required to|requirements?)\b",
    flags=re.IGNORECASE,
)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'([])")
HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", flags=re.I | re.S)
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def extract_requirement_statements(text: str) -> list[str]:
    without_tables = HTML_TABLE_RE.sub(" ", text)
    statements: list[str] = []
    seen: set[str] = set()
    for paragraph in re.split(r"\n+", without_tables):
        for sentence in SENTENCE_BOUNDARY_RE.split(paragraph.strip()):
            normalized = normalize_text(sentence)
            if not normalized or not OBLIGATION_RE.search(normalized):
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            statements.append(normalized)
    return statements


def fallback_requirements_for_clause(
    clause: Clause,
    *,
    tables: list[RegulationTable],
    section_id: str | None = None,
) -> list[Requirement]:
    statements = extract_requirement_statements(clause.text)
    requirements: list[Requirement] = []
    table_ids = [table.table_id for table in tables if table.owner_clause_id == clause.clause_id]
    for index, statement in enumerate(statements, start=1):
        req_type = infer_requirement_type(statement, clause)
        requirements.append(
            Requirement(
                requirement_id=make_requirement_id(clause.clause_id, index, statement),
                source_clause_id=clause.clause_id,
                requirement_type=req_type,
                actor=infer_actor(statement),
                legal_text=statement,
                engineering_summary=to_engineering_summary(statement),
                conditions=infer_conditions(statement),
                acceptance_criteria=infer_acceptance(statement),
                referenced_clause_ids=clause.references,
                table_ids=table_ids,
                source_section_id=section_id,
                fallback_used=True,
            )
        )
    return requirements


def infer_requirement_type(text: str, clause: Clause) -> str:
    lower = f"{clause.title} {text}".lower()
    if "impact speed" in lower or "speed reduction" in lower or "m/s" in lower or "braking demand" in lower:
        return "performance"
    if "warning" in lower:
        return "warning"
    if "deactiv" in lower:
        return "deactivation"
    if "failure" in lower or "self-check" in lower:
        return "failure_detection"
    if "documentation" in lower or "recorded" in lower or "provide a list" in lower:
        return "documentation"
    if "test" in lower or "shall be tested" in lower or clause.document_region == "test_procedure":
        return "procedure"
    if "active" in lower or "activated" in lower:
        return "activation"
    return "other"


def infer_actor(text: str) -> str | None:
    lower = text.lower()
    if "aebs" in lower or "system" in lower:
        return "AEBS"
    if "vehicle" in lower:
        return "vehicle"
    if "manufacturer" in lower:
        return "vehicle manufacturer"
    return None


def infer_conditions(text: str) -> list[str]:
    conditions: list[str] = []
    for pattern in [
        r"when [^.;]+",
        r"in absence of [^.;]+",
        r"within [^.;]+",
        r"at [^.;]+",
        r"between [^.;]+",
    ]:
        for match in re.finditer(pattern, text, flags=re.I):
            conditions.append(normalize_text(match.group(0)))
    return unique(conditions)[:8]


def infer_acceptance(text: str) -> list[str]:
    lower = text.lower()
    criteria: list[str] = []
    if "at least" in lower or "not more than" in lower or "less or equal" in lower or "shall not" in lower:
        criteria.append(text)
    return criteria


def to_engineering_summary(text: str) -> str:
    summary = text
    summary = re.sub(r"\bshall\b", "must", summary, flags=re.I)
    summary = re.sub(r"\bshall not\b", "must not", summary, flags=re.I)
    return normalize_text(summary)


def make_requirement_id(clause_id: str, index: int, text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "-", clause_id).strip("-") or "clause"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"REQ-{safe}-{index:03d}-{digest}"


def normalize_text(text: str) -> str:
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
