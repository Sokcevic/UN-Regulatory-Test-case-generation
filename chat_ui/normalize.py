"""
normalize.py — Repair parsed clauses before they reach the chat tools.

Three problems in the raw parse of a real UN regulation, all fixed here at load
time (no pipeline re-run, works for any uploaded document):

1. **Annex numbering collides with the body.** Each annex restarts numbering at
   1, so Annex 3's "§5 Reporting by Technical Service" gets the same clause_id
   "5" as the body's "§5 Specifications" (and is even mis-tagged
   region='specifications'). We use document order (line_start) and the
   region='annex' heading clauses ("Annex 3", "Annex 3 - Appendix 1") to
   **namespace** annex clauses: body "5" stays "5", Annex 3's becomes
   "Annex 3 / 5", nested under "Annex 3". Parentage is fully derivable from the
   id string (see parent_of), so it also drives the hierarchy tools + tree UI.

2. **Duplicate copies per id.** The Communication form reuses body numbers too
   ("Brief description of vehicle:" → clause "5"). choose_best_copies keeps the
   substantive body copy; prune_form_fields removes the form boilerplate.

3. **Headings live in `text` with an empty `title`.** repair_titles recovers a
   title from `text` / children, e.g. 5.2.1 → "Car to car scenario", section 5
   → "Specifications".

Everything here is pure and dict-based (the shape of 01_clauses.json entries),
so it is unit-testable and document-agnostic.
"""

from __future__ import annotations

import re
from collections import Counter

_FORMFIELD_RE = re.compile(r":\s*[.…_ ]*$")
_DOTFILL_RE = re.compile(r"[.…_]{3,}")
# A Communication-form label ("Type of vehicle:", "Date of submission for
# approval:") is a short phrase. Beyond this many words a trailing colon is
# almost certainly a normal sentence introducing a list or table ("…as shown in
# the following table:"), which must NOT be pruned as boilerplate.
_MAX_FORMFIELD_LABEL_WORDS = 8
_NONBODY_PATH = {"communication", "annexes", "annex"}
_FORM_PATH = {"communication"}

# Annex namespacing + parentage now live in the core package so the graph and the
# clause index use ONE implementation (no drift). Re-exported here for callers.
from regulatory_testgen.structure_ids import (  # noqa: E402
    _NUM_RE, _APPENDIX_RE, _ANNEX_RE, _SEP, parent_of, assign_structure,
)

_TOC_LEADER_RE = re.compile(r"[.…_]{5,}")
_TOC_PAGENUM_RE = re.compile(r"\s*\.\s*\d{1,3}\s*$")
_BARE_NUMBER_RE = re.compile(r"^[\d.\s]+$")


# ── Structure: annex namespacing + parentage ────────────────────────────────

def resolve_parent(clause_id: str, present) -> str | None:
    """Nearest ancestor of clause_id that is actually present (handles gaps and
    pruned intermediate nodes by climbing the parent_of chain)."""
    p = parent_of(clause_id)
    while p is not None and p not in present:
        p = parent_of(p)
    return p


def children_map(ids) -> dict[str, list[str]]:
    """{parent_id: [child_id, ...]} over the present ids, using resolved parents."""
    present = set(ids)
    kids: dict[str, list[str]] = {}
    for cid in present:
        p = resolve_parent(cid, present)
        if p is not None:
            kids.setdefault(p, []).append(cid)
    return kids


def _children_of(clause_id: str, ids) -> list[str]:
    present = set(ids)
    return [c for c in present if resolve_parent(c, present) == clause_id]


# ── Duplicate resolution + form-field pruning ───────────────────────────────

def looks_like_formfield(title: str) -> bool:
    """True if a title reads like a Communication-form label rather than a heading."""
    t = (title or "").strip()
    if not t:
        return False
    if _FORMFIELD_RE.search(t):
        # A trailing colon marks a form label only for a short phrase or one with
        # dot-fill leaders — not a full sentence that merely ends in a colon.
        return bool(_DOTFILL_RE.search(t)) or len(t.split()) <= _MAX_FORMFIELD_LABEL_WORDS
    return ":" in t and bool(_DOTFILL_RE.search(t))


def is_form_junk(c: dict) -> bool:
    """True if a clause copy is Communication-form / TOC boilerplate, not text."""
    path = [str(p).lower() for p in (c.get("section_path") or [])]
    if any(p in _FORM_PATH for p in path):
        return True
    if looks_like_formfield((c.get("title") or "").strip()):
        return True
    line = _first_meaningful_line(c.get("text") or "")
    return looks_like_formfield(line) or bool(_TOC_LEADER_RE.search(line))


def natkey(clause_id: str):
    """Natural sort key: '5.10' after '5.2', numeric roots before annex roots."""
    key = []
    for part in re.split(r"[.\s/]+", (clause_id or "").strip()):
        if part.isdigit():
            key.append((0, int(part), ""))
        elif part:
            key.append((1, 0, part.lower()))
    return key


def _score_copy(c: dict) -> float:
    """Higher = more likely the substantive body clause for this id."""
    path = [str(p).lower() for p in (c.get("section_path") or [])]
    title = (c.get("title") or "").strip()
    text = (c.get("text") or "").strip()
    region = (c.get("document_region") or "").strip()

    score = 0.0
    if looks_like_formfield(title):
        score -= 100.0
    if any(p == "communication" for p in path):
        score -= 80.0
    if any(p in ("annexes", "annex") for p in path):
        score -= 40.0
    if region:
        score += 5.0
    if title and not looks_like_formfield(title):
        score += 20.0
    if path and path[0] not in _NONBODY_PATH:
        score += 15.0
    score += min(len(text), 1200) / 100.0
    return score


def choose_best_copies(clauses: list[dict]) -> dict[str, dict]:
    """Group raw clause dicts by clause_id and keep the single best copy each."""
    by_id: dict[str, dict] = {}
    best_score: dict[str, float] = {}
    for c in clauses:
        cid = (c.get("clause_id") or "").strip()
        if not cid or c.get("is_pseudo_clause"):
            continue
        s = _score_copy(c)
        if cid not in by_id or s > best_score[cid]:
            by_id[cid] = c
            best_score[cid] = s
    return by_id


def prune_form_fields(best: dict[str, dict]) -> dict[str, dict]:
    """Remove Communication-form boilerplate that collides with body numbering.

    Leaf form-junk is dropped; a form-junk node that has children is a real
    section whose own copy was polluted — kept but blanked so repair_titles
    rebuilds its title from the children."""
    ids = set(best.keys())
    result: dict[str, dict] = {}
    for cid, c in best.items():
        if not is_form_junk(c):
            result[cid] = c
            continue
        if _children_of(cid, ids):
            cleaned = dict(c)
            cleaned["title"] = ""
            cleaned["text"] = ""
            result[cid] = cleaned
        # else: leaf form field → drop entirely
    return result


# ── Title recovery ──────────────────────────────────────────────────────────

def _first_meaningful_line(text: str) -> str:
    for line in (text or "").splitlines():
        s = line.strip()
        if s and not _BARE_NUMBER_RE.match(s):
            return s
    return ""


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 60:
        return False
    if looks_like_formfield(line):
        return False
    return not line.endswith(".") or line.count(".") <= 1


def _shared_child_section(clause_id: str, index: dict[str, dict], ids) -> str:
    """Most common `section_path` heading among a top-level section's children."""
    headings = Counter(
        (index[k].get("section_path") or [""])[-1].strip()
        for k in _children_of(clause_id, ids)
    )
    for h, _ in headings.most_common():
        if h and h.lower() not in _NONBODY_PATH:
            return h
    return ""


def repair_titles(index: dict[str, dict]) -> dict[str, dict]:
    """Fill in a usable `title` for every entry (see module docstring)."""
    ids = set(index.keys())
    for cid, info in index.items():
        title = (info.get("title") or "").strip()
        if title and not looks_like_formfield(title):
            info["title"] = title
            continue

        if resolve_parent(cid, ids) is None:  # top-level section
            shared = _shared_child_section(cid, index, ids)
            if shared:
                info["title"] = shared
                continue

        line = _first_meaningful_line(info.get("text") or "")
        if _looks_like_heading(line):
            info["title"] = line
            continue
        if line:
            info["title"] = (line[:80].rstrip() + "…") if len(line) > 80 else line
            continue
        info["title"] = title or "(untitled)"

    # Final tidy: strip a leading self-id prefix, trailing TOC page numbers, ws.
    for cid, info in index.items():
        cleaned = (info.get("title") or "").strip()
        bare = cid.split(_SEP)[-1]
        cleaned = re.sub(r"^" + re.escape(bare) + r"[\s.:\-]+", "", cleaned)
        cleaned = _TOC_PAGENUM_RE.sub("", cleaned).strip()
        info["title"] = re.sub(r"\s+", " ", cleaned) or "(untitled)"
    return index
