"""
retrieval.py — Map a natural-language user query to relevant testable clause IDs.

Uses the already-generated checkpoints (no pipeline re-run needed):
  ../regulatory_testgen/output/01_clauses.json
  ../regulatory_testgen/output/01b_classifications.json

The LLM receives only clause titles + categories (compact), so this call is cheap.
"""

from __future__ import annotations

import json
from pathlib import Path

CHECKPOINTS = Path(__file__).parent.parent / "regulatory_testgen" / "output"
TESTABLE_CATEGORIES = {"obligation", "test_execution", "test_procedure", "performance_data"}
TEST_RELEVANT_CATEGORIES = TESTABLE_CATEGORIES | {"test_condition", "test_setup"}


def build_clause_index(clauses_raw: list[dict], classifications: dict[str, str]) -> dict[str, dict]:
    """Build the repaired clause index from raw parsed clauses + classifications.

    Split out from load_clause_index so it can be unit-tested against synthetic
    data. Deduplicates colliding clause_ids (Communication-form / annex reuse)
    and recovers titles — see normalize.py. Each entry keeps the FULL text (not a
    400-char preview) plus references and section_path so the chat tools can show
    complete clauses, hierarchy, and cross-references.
    """
    from normalize import (
        assign_structure, choose_best_copies, prune_form_fields, repair_titles,
        resolve_parent,
    )
    from regulatory_testgen.classify import (
        category_of, is_testable, normalize_classifications,
    )

    classifications = normalize_classifications(classifications)

    structured = assign_structure(clauses_raw)  # namespace annex clauses (no id collisions)
    best = choose_best_copies(structured)        # {clause_id: best raw copy}
    best = prune_form_fields(best)               # drop Communication-form boilerplate

    # Classifications are keyed by each clause's unique uid (annexes reuse ids).
    # Fall back to the bare clause_id for legacy caches.
    def _entry_for(c: dict) -> dict:
        e = classifications.get(c.get("uid") or "")
        if e is None:
            e = classifications.get((c.get("clause_id") or "").split(" / ")[-1])
        return e or {"category": "unknown", "force": "none"}

    index: dict[str, dict] = {}
    for cid, c in best.items():
        text = (c.get("text") or "").strip()
        entry = _entry_for(c)
        index[cid] = {
            "title": (c.get("title") or "").strip(),
            "text": text,
            "text_preview": text[:400],          # kept for backward compatibility
            "category": category_of(entry),
            "force": entry.get("force", "binding"),
            "testable": is_testable(entry),
            "section_path": list(c.get("section_path") or []),
            "references": list(c.get("references") or []),
        }

    present = set(index.keys())
    for cid, info in index.items():
        info["parent"] = resolve_parent(cid, present)

    repair_titles(index)
    return index


def load_clause_index() -> dict[str, dict]:
    """Load clauses from checkpoints → repaired {clause_id: {title, text,
    text_preview, category, section_path, references}}."""
    clauses_raw = json.loads((CHECKPOINTS / "01_clauses.json").read_text(encoding="utf-8"))

    classifications: dict[str, str] = {}
    clf_file = CHECKPOINTS / "01b_classifications.json"
    if clf_file.exists():
        classifications = json.loads(clf_file.read_text(encoding="utf-8"))

    return build_clause_index(clauses_raw, classifications)


def find_relevant_clauses(
    query: str,
    llm_config,  # LLMConfig
    clause_index: dict[str, dict],
    top_k: int = 8,
    testable_only: bool = True,
) -> list[str]:
    """Ask the LLM which clause IDs are relevant to the user query.

    testable_only=True  → only obligation/test_execution/performance_data clauses
    testable_only=False → search all categories including test_condition, test_setup, definitions

    Returns a list of clause IDs (up to top_k), ordered by relevance.
    Falls back to keyword matching if the LLM call fails.
    """
    from openai import OpenAI

    category_filter = TESTABLE_CATEGORIES if testable_only else None

    candidate_lines = []
    for cid, info in clause_index.items():
        if category_filter is not None and info["category"] not in category_filter:
            continue
        title = info["title"][:90] or "(no title)"
        # Include first 100 chars of text as hint for the LLM
        hint = info["text_preview"][:100].replace("\n", " ").strip()
        candidate_lines.append(
            f"  {cid} [{info['category']}]: {title}"
            + (f" — {hint}" if hint else "")
        )

    if not candidate_lines:
        return []

    client = OpenAI(base_url=llm_config.base_url, api_key=llm_config.api_key)

    system = (
        "You are a regulatory expert for UN Regulation No. 152 on Advanced Emergency Braking Systems (AEBS). "
        "Identify which regulatory clauses are most relevant to a given query."
    )
    user_prompt = (
        f'Query: "{query}"\n\n'
        f"Clauses (ID [category]: title — text excerpt):\n"
        + "\n".join(candidate_lines)
        + f"\n\nReturn JSON {{\"clause_ids\": [...]}} with up to {top_k} most relevant clause IDs. "
        "Only include clauses genuinely relevant to the query. Prefer specific clauses over broad sections."
    )

    try:
        response = client.chat.completions.create(
            model=llm_config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=300,
        )
        result = json.loads(response.choices[0].message.content)
        ids = result.get("clause_ids", [])
        return [cid for cid in ids if cid in clause_index][:top_k]
    except Exception:
        # Fallback: substring match against title + text
        q_lower = query.lower()
        scored: list[tuple[int, str]] = []
        for cid, info in clause_index.items():
            if category_filter is not None and info["category"] not in category_filter:
                continue
            combined = (info["title"] + " " + info["text_preview"]).lower()
            score = sum(1 for word in q_lower.split() if len(word) > 3 and word in combined)
            if score > 0:
                scored.append((score, cid))
        scored.sort(reverse=True)
        return [cid for _, cid in scored[:top_k]]
