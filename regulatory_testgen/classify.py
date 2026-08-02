"""Semantic clause classification stage (hierarchy-aware).

Each parsed clause is assigned a functional CATEGORY plus a normative FORCE by
an LLM. Unlike a flat per-clause pass, classification uses the document tree so
each clause is judged in context:

  • bottom-up  — a section sees the clauses it contains (a heading over several
    braking obligations is a scenario grouping, not "formatting");
  • top-down   — every subtree is classified beneath a global outline that tags
    each top-level container's role, so obligation-shaped text inside a *sample*
    annex is marked force="example" (and thus not testable).

Context stays bounded: we send one section-subtree per call (recursing/chunking
if a subtree is large) plus a compact global outline — never the whole document.

Keying: results are keyed by each clause's unique `uid`, not `clause_id`, because
annexes restart numbering (body §5 and Annex 3 §5 share clause_id "5").
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

from regulatory_testgen.config import LLMConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Taxonomy (fixed, universal — see module design notes)
# ---------------------------------------------------------------------------

CLAUSE_CATEGORIES: dict[str, str] = {
    "obligation": (
        "A binding requirement stating what a system, vehicle, or component SHALL "
        "do, achieve, or avoid — a behavioural mandate. Primary test-case source. "
        "Signal words: 'shall', 'must'."
    ),
    "performance_data": (
        "Tabular performance limits, threshold values, speed/distance parameters or "
        "scenario matrices that directly parameterise test cases (pass/fail values)."
    ),
    "test_execution": (
        "Step-by-step procedure for running a specific test scenario: approach speed, "
        "trajectory, target placement/timing, measurement and pass/fail assessment."
    ),
    "test_condition": (
        "Mandatory environmental/equipment prerequisites for a valid test: road "
        "surface (PBC, slope), temperature, visibility/lighting, instrument standards."
    ),
    "test_setup": (
        "How the test vehicle and targets are prepared: test mass/load, conditioning, "
        "tyre identification, target specifications."
    ),
    "test_procedure": (
        "Legacy generic test clause — prefer test_execution/test_condition/test_setup."
    ),
    "definition": "Defines a term, abbreviation, or concept used elsewhere.",
    "scope": "States which vehicles/systems/situations the regulation applies to.",
    "administrative": (
        "Document-management content: form fields, approval paperwork, communication "
        "templates, report numbers, transitional provisions, record-keeping."
    ),
    "formatting": (
        "A section heading/title that only groups sub-clauses, a table-of-contents "
        "entry, or front matter with no substantive requirement of its own."
    ),
    "informative": (
        "Explanatory notes, background, or non-binding guidance imposing no requirement."
    ),
}

# Normative force — orthogonal to category. A clause may look like an obligation
# yet be a worked example inside a sample annex; only 'binding' clauses are tested.
NORMATIVE_FORCE: dict[str, str] = {
    "binding": "A real, in-force requirement/procedure that applies to compliance.",
    "example": (
        "Illustrative/sample content — a worked example, sample scenario, model form, "
        "or specimen documentation. Reads like a requirement but is not itself binding."
    ),
    "none": "Carries no normative force (definitions, headings, notes, admin).",
}

CATEGORY_ROLES: dict[str, str] = {
    "normative_body": "Substantive in-force requirements/procedures (the regulation proper).",
    "informative_annex": "Examples, samples, model forms, or explanatory material.",
    "administrative": "Communication forms, approval paperwork, front matter, TOC.",
    "mixed": "A mix — judge each clause individually.",
}

# Functional categories whose clauses are candidate test sources …
TESTABLE_CATEGORIES: frozenset[str] = frozenset(
    {"obligation", "test_execution", "test_procedure", "performance_data"}
)
TEST_RELEVANT_CATEGORIES: frozenset[str] = frozenset(
    TESTABLE_CATEGORIES | {"test_condition", "test_setup"}
)


def is_testable(entry: Any) -> bool:
    """True if a classification entry is a testable requirement.

    Accepts either the rich dict ({"category","force"}) or a bare category
    string (legacy). Only binding clauses in a testable category qualify."""
    if isinstance(entry, dict):
        return entry.get("category") in TESTABLE_CATEGORIES and entry.get("force", "binding") == "binding"
    return entry in TESTABLE_CATEGORIES


def category_of(entry: Any) -> str:
    return entry.get("category", "unknown") if isinstance(entry, dict) else (entry or "unknown")


# ---------------------------------------------------------------------------
# Document structure (annex-aware; mirrors chat_ui/normalize but self-contained)
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"^\d+(\.\d+)*$")
_APPENDIX_RE = re.compile(r"^(Annex\b.*?)\s*-\s*Appendix\b", re.IGNORECASE)
_ANNEX_RE = re.compile(r"^Annex\b", re.IGNORECASE)
_SEP = " / "


def _parent_key(key: str) -> str | None:
    if _SEP in key:
        container, tail = key.rsplit(_SEP, 1)
        if "." in tail and _NUM_RE.match(tail):
            return f"{container}{_SEP}{tail.rsplit('.', 1)[0]}"
        return container
    m = _APPENDIX_RE.match(key)
    if m:
        return m.group(1).strip()
    if _ANNEX_RE.match(key):
        return None
    if _NUM_RE.match(key) and "." in key:
        return key.rsplit(".", 1)[0]
    return None


def _assign_keys(clauses: list[Any]) -> list[tuple[str, Any]]:
    """Namespace annex clauses by document order → [(key, clause), ...] in order."""
    ordered = sorted(clauses, key=lambda c: getattr(c, "line_start", 0) or 0)
    container: str | None = None
    keyed: list[tuple[str, Any]] = []
    for c in ordered:
        cid = (getattr(c, "clause_id", "") or "").strip()
        if not cid or getattr(c, "is_pseudo_clause", False):
            keyed.append((cid, c))
            continue
        if _NUM_RE.match(cid):
            key = f"{container}{_SEP}{cid}" if container else cid
        elif _APPENDIX_RE.match(cid) or _ANNEX_RE.match(cid):
            container = cid
            key = cid
        else:
            key = f"{container}{_SEP}{cid}" if container else cid
        keyed.append((key, c))
    return keyed


def _natkey(key: str):
    out = []
    for part in re.split(r"[.\s/]+", key.strip()):
        out.append((0, int(part), "") if part.isdigit() else (1, 0, part.lower()))
    return out


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_OUTLINE_SYSTEM = (
    "You are a regulatory document analyst. Given the top-level sections of a "
    "regulation, assign each a ROLE and a one-line gist. Roles:\n{roles}\n"
    "Respond with valid JSON only."
)
_OUTLINE_USER = (
    "Top-level sections (id — title — sample of contents):\n{blocks}\n\n"
    'Return JSON {{"results":[{{"id":..,"role":<one role>,"gist":<one sentence>}}]}}.'
)

_CLASSIFY_SYSTEM = (
    "You are a regulatory document analyst. Classify EVERY clause in the given "
    "section into one functional CATEGORY and one normative FORCE, using the "
    "surrounding hierarchy for context (a heading over several obligations is a "
    "grouping; a clause inside a sample/informative annex is force='example').\n\n"
    "CATEGORIES:\n{category_block}\n\nFORCE:\n{force_block}\n\n"
    "Rules: exactly one category and one force per clause; judge by meaning, not "
    "wording; a clause is force='example' only if it (or an ancestor) is an "
    "illustrative sample/model, else 'binding' for requirements or 'none' for "
    "definitions/headings/notes. Respond with valid JSON only."
)
_CLASSIFY_USER = (
    "DOCUMENT OUTLINE (top-level context):\n{outline}\n\n"
    "SECTION under review — '{section}' (role: {role}). Classify each node below; "
    "indentation shows nesting.\n\n{nodes}\n\n"
    'Return JSON {{"results":[{{"key":<the [key]>, "category":<one>, '
    '"force":<one>, "reasoning":<one sentence>}}]}}.'
)

BATCH_MAX_CHARS = 9000  # per-classification-call budget over serialized clause text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_clauses(
    clauses: list[Any],
    llm_config: LLMConfig,
    progress_cb=None,  # callable(done_units, total_units)
) -> dict[str, dict]:
    """Classify clauses hierarchically. Returns {uid: {category, force, reasoning}}.

    Falls back gracefully (informative/none) on any LLM/parse failure.
    """
    client = OpenAI(base_url=llm_config.base_url, api_key=llm_config.api_key)

    def _uid(c: Any) -> str:
        return getattr(c, "uid", None) or getattr(c, "clause_id", "") or ""

    results: dict[str, dict] = {}

    keyed = _assign_keys(clauses)
    node_by_key: dict[str, Any] = {}
    for key, c in keyed:
        if getattr(c, "is_pseudo_clause", False) or not key:
            results[_uid(c)] = {"category": "formatting", "force": "none",
                                "reasoning": "pseudo/front-matter stub"}
        else:
            node_by_key[key] = c  # last write wins if duplicate namespaced key (rare)

    present = set(node_by_key)
    children: dict[str, list[str]] = {}
    for key in present:
        p = _parent_key(key)
        while p is not None and p not in present:
            p = _parent_key(p)
        if p is not None:
            children.setdefault(p, []).append(key)
    for p in children:
        children[p].sort(key=_natkey)
    roots = sorted([k for k in present if _resolve_root_parent(k, present) is None], key=_natkey)

    # ── Pass 1: global outline (one call) ────────────────────────────────────
    outline = _classify_outline(client, llm_config, roots, children, node_by_key)
    outline_text = "\n".join(
        f"  [{r}] {_title(node_by_key, r)} — role={outline.get(r, {}).get('role', 'mixed')}: "
        f"{outline.get(r, {}).get('gist', '')}"
        for r in roots
    )

    # ── Pass 2: per-subtree classification ───────────────────────────────────
    total = len(roots)
    for i, root in enumerate(roots):
        role = outline.get(root, {}).get("role", "mixed")
        subtree_keys = _subtree_keys(root, children)
        for chunk in _chunk(subtree_keys, node_by_key):
            got = _classify_chunk(
                client, llm_config, outline_text, root,
                _title(node_by_key, root), role, chunk, children, node_by_key,
            )
            for key, entry in got.items():
                c = node_by_key.get(key)
                if c is not None:
                    results[_uid(c)] = entry
        if progress_cb is not None:
            progress_cb(i + 1, total)

    # Anything left unclassified (parse gaps) → safe default.
    for key, c in node_by_key.items():
        results.setdefault(_uid(c), {"category": "informative", "force": "none",
                                     "reasoning": "unclassified — defaulted"})
    return results


def _resolve_root_parent(key: str, present: set[str]) -> str | None:
    p = _parent_key(key)
    while p is not None and p not in present:
        p = _parent_key(p)
    return p


def _title(node_by_key: dict[str, Any], key: str) -> str:
    c = node_by_key.get(key)
    if c is None:
        return key
    t = (getattr(c, "title", "") or "").strip()
    if t:
        return t
    txt = (getattr(c, "text", "") or "").strip().splitlines()
    return (txt[0][:60] if txt else key)


def _subtree_keys(root: str, children: dict[str, list[str]]) -> list[str]:
    """Root + descendants in pre-order (document-ish order)."""
    out: list[str] = []

    def walk(k: str) -> None:
        out.append(k)
        for ch in children.get(k, []):
            walk(ch)

    walk(root)
    return out


def _depth(key: str, root: str) -> int:
    """Nesting depth of key below root, for indentation."""
    d = 0
    cur = key
    while cur is not None and cur != root:
        cur = _parent_key(cur)
        d += 1
        if d > 12:
            break
    return d


def _chunk(keys: list[str], node_by_key: dict[str, Any]) -> list[list[str]]:
    """Split a subtree's keys into groups within BATCH_MAX_CHARS (keeps order)."""
    chunks: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for k in keys:
        c = node_by_key.get(k)
        n = len((getattr(c, "text", "") or "")[:800]) + 80
        if cur and size + n > BATCH_MAX_CHARS:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(k)
        size += n
    if cur:
        chunks.append(cur)
    return chunks or [[]]


def _classify_outline(client, cfg, roots, children, node_by_key) -> dict[str, dict]:
    if not roots:
        return {}
    blocks = []
    for r in roots:
        kids = children.get(r, [])
        sample = "; ".join(_title(node_by_key, k) for k in kids[:8])
        head = _title(node_by_key, r)
        txt = (getattr(node_by_key.get(r), "text", "") or "").strip().replace("\n", " ")[:160]
        blocks.append(f"  [{r}] {head} — {txt} | children: {sample}")
    sys = _OUTLINE_SYSTEM.format(
        roles="\n".join(f"  {k}: {v}" for k, v in CATEGORY_ROLES.items())
    )
    user = _OUTLINE_USER.format(blocks="\n".join(blocks))
    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            response_format={"type": "json_object"}, temperature=0.0,
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        out = {}
        for r in parsed.get("results", []):
            rid = (r.get("id") or "").strip()
            if rid not in node_by_key:
                rid = rid.strip("[]").strip()
            if rid in node_by_key:
                role = r.get("role") if r.get("role") in CATEGORY_ROLES else "mixed"
                out[rid] = {"role": role, "gist": r.get("gist", "")}
        return out
    except Exception as exc:
        logger.warning("Outline pass failed: %s — defaulting roles to 'mixed'", exc)
        return {}


def _classify_chunk(client, cfg, outline_text, root, section_title, role,
                    chunk, children, node_by_key) -> dict[str, dict]:
    if not chunk:
        return {}
    lines = []
    for k in chunk:
        c = node_by_key.get(k)
        indent = "  " * _depth(k, root)
        title = _title(node_by_key, k)
        body = (getattr(c, "text", "") or "").strip().replace("\n", " ")[:800]
        lines.append(f"{indent}[{k}] {title}\n{indent}    {body}")
    sys = _CLASSIFY_SYSTEM.format(
        category_block="\n".join(f"  {k}: {v}" for k, v in CLAUSE_CATEGORIES.items()),
        force_block="\n".join(f"  {k}: {v}" for k, v in NORMATIVE_FORCE.items()),
    )
    user = _CLASSIFY_USER.format(
        outline=outline_text, section=f"{root} {section_title}", role=role,
        nodes="\n".join(lines),
    )
    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            response_format={"type": "json_object"}, temperature=0.0,
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
    except Exception as exc:
        logger.warning("Classify chunk under %s failed: %s", root, exc)
        return {}

    out: dict[str, dict] = {}
    for r in parsed.get("results", []):
        key = (r.get("key") or "").strip()
        if key not in node_by_key:            # models often echo the "[key]" with brackets
            key = key.strip("[]").strip()
        if key not in node_by_key:
            continue
        cat = r.get("category")
        if cat not in CLAUSE_CATEGORIES:
            cat = "informative"
        force = r.get("force")
        if force not in NORMATIVE_FORCE:
            force = "binding" if cat in TESTABLE_CATEGORIES else "none"
        out[key] = {"category": cat, "force": force, "reasoning": r.get("reasoning", "")}
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_classifications(classifications: dict, path) -> None:
    import pathlib
    pathlib.Path(path).write_text(
        json.dumps(classifications, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_classifications(path) -> dict:
    import pathlib
    return normalize_classifications(json.loads(pathlib.Path(path).read_text(encoding="utf-8")))


def normalize_classifications(raw: dict) -> dict:
    """Accept either the rich {uid:{category,force}} format or the legacy
    {clause_id:"category"} format, returning the rich form."""
    out: dict[str, dict] = {}
    for k, v in (raw or {}).items():
        if isinstance(v, dict):
            out[k] = {"category": v.get("category", "informative"),
                      "force": v.get("force", "binding"),
                      "reasoning": v.get("reasoning", "")}
        else:
            out[k] = {"category": v or "informative",
                      "force": "binding" if v in TESTABLE_CATEGORIES else "none",
                      "reasoning": ""}
    return out
