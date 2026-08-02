from __future__ import annotations

import re

from regulatory_testgen.data_models.core import Clause, ReferenceLink

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


def extract_reference_links(clauses: list[Clause]) -> list[ReferenceLink]:
    known_clause_ids = {normalize_id(clause.clause_id) for clause in clauses}
    links: list[ReferenceLink] = []
    for clause in clauses:
        references = sorted(set([*clause.references, *find_references(clause.text)]))
        for ref in references:
            target_type = "clause" if normalize_id(ref) in known_clause_ids else infer_target_type(ref)
            links.append(
                ReferenceLink(
                    source_id=clause.clause_id,
                    target_id=ref,
                    source_type="clause",
                    target_type=target_type,
                    relation="references",
                    text=ref,
                )
            )
    return deduplicate_links(links)


def find_references(text: str) -> list[str]:
    refs: list[str] = []
    for match in PARAGRAPH_REFERENCE_RE.finditer(text):
        refs.extend(num.rstrip(".") for num in CLAUSE_ID_RE.findall(match.group(1)))
    for match in ANNEX_REFERENCE_RE.finditer(text):
        refs.append(clean_text(match.group(1)))
    return [ref for ref in refs if ref]


def infer_target_type(ref: str) -> str:
    if ref.lower().startswith("annex") or "annex" in ref.lower():
        return "annex"
    return "unknown"


def normalize_id(value: str) -> str:
    return value.rstrip(".").strip().lower()


def deduplicate_links(links: list[ReferenceLink]) -> list[ReferenceLink]:
    seen: set[tuple[str, str, str]] = set()
    result: list[ReferenceLink] = []
    for link in links:
        key = (link.source_id, normalize_id(link.target_id), link.relation)
        if key in seen:
            continue
        seen.add(key)
        result.append(link)
    return result


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
