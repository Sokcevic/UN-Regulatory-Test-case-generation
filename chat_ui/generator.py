"""
generator.py — Load the regulatory graph (once) and run the ReAct agent on demand.

Wraps the existing regulatory_testgen agent machinery. The graph is built from
the pipeline's checkpoint files so the full LLM pipeline never needs to re-run.

Row-filter support: pass a natural-language constraint (e.g. "only unladen vehicles
at 60 km/h") and it is appended to the clause text as an instruction before the
agent call — no changes to the core agent code required.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Defaults to the copy bundled in the repo (data/R152r2E); override with
# R152_DATA_DIR to point at a different MinerU output directory.
_MINERU_DIR = Path(os.environ.get("R152_DATA_DIR", str(_ROOT / "data" / "R152r2E")))
MARKDOWN_PATH = _MINERU_DIR / "R152r2E.md"
_MIDDLE_JSON_PATH = _MINERU_DIR / "R152r2E_middle.json"
_ORIGIN_PDF_PATH = _MINERU_DIR / "R152r2E_origin.pdf"

ClauseType = Any  # regulatory_testgen.parsing.markdown_parser.Clause


def load_graph_and_config():
    """Parse the regulation and build the in-memory graph. Expensive — call once and cache.

    Returns (graph, llm_config, clause_map, source_info) where clause_map is
    {clause_id: Clause} and source_info is {"pdf_path": Path|None, "clause_pages": {...}}
    for tracing answers back to the original PDF (see pdf_highlight.py).
    """
    from regulatory_testgen.config import LLMConfig  # noqa: F401
    from regulatory_testgen.graph import RegulatoryGraph
    from regulatory_testgen.extraction.tables import extract_tables
    from regulatory_testgen.parsing.markdown_parser import (
        parse_markdown_clauses,
        recover_missing_numeric_parents,
    )

    _TITLE_HINTS = {
        "5.2": "Specific Requirements",
        "6.1": "Test Conditions",
        "6.3": "Test Targets",
        "6.7": "Warning and Activation Test with a Bicycle Target",
    }

    llm_config = LLMConfig(max_tokens=32768)

    from regulatory_testgen.structure_ids import namespace_clauses

    logger.info("Parsing regulation document…")
    clauses = parse_markdown_clauses(MARKDOWN_PATH)
    clauses = recover_missing_numeric_parents(clauses, title_hints=_TITLE_HINTS)
    # Namespace annex clauses so clause_map keys match the graph + clause index
    # (the graph namespaces internally; keep clause_map consistent for lookups).
    clauses = namespace_clauses(clauses)
    tables = extract_tables(clauses)
    graph = RegulatoryGraph.build(clauses, tables)
    logger.info(
        "Graph ready: %d clauses, %d tables",
        graph.stats()["clauses"],
        len(tables),
    )

    # Attach semantic categories so get_generation_context can use the
    # formatting-root block (fmt_root_biref): the nearest titled-header ancestor's
    # subtree + its bidirectional reference closure. Categories live in the
    # classification checkpoint, not on the Clause model; if unavailable the graph
    # simply omits that block and falls back to base + self_sub.
    try:
        from retrieval import load_clause_index

        clause_index = load_clause_index()
        graph.set_categories({
            cid: (info or {}).get("category") for cid, info in clause_index.items()
        })
        logger.info("Attached categories to graph for %d clauses", len(clause_index))
    except Exception as exc:  # noqa: BLE001 — categories are optional
        logger.info("No categories attached to graph (%s); "
                    "formatting-root block disabled", exc)

    # clause_map for quick lookup by ID (longest text wins duplicates)
    clause_map: dict[str, ClauseType] = {}
    for c in clauses:
        cid = c.clause_id
        if cid not in clause_map or len(c.text or "") > len(clause_map[cid].text or ""):
            clause_map[cid] = c

    clause_pages: dict[str, list[dict]] = {}
    source_pdf_path = _ORIGIN_PDF_PATH if _ORIGIN_PDF_PATH.exists() else None
    if source_pdf_path is not None and _MIDDLE_JSON_PATH.exists():
        from pdf_highlight import build_clause_page_map
        clause_pages = build_clause_page_map(clauses, _MIDDLE_JSON_PATH)
    source_info = {
        "pdf_path": source_pdf_path,
        "clause_pages": clause_pages,
        "middle_json_path": _MIDDLE_JSON_PATH if _MIDDLE_JSON_PATH.exists() else None,
    }

    return graph, llm_config, clause_map, source_info


def get_table_preview(graph, clause_id: str) -> list[dict]:
    """Return the rows of all tables attached to clause_id (for UI display)."""
    tables = graph.get_tables(clause_id)
    rows: list[dict] = []
    for t in tables:
        rows.extend(t.rows)
    return rows


def build_generation_context_text(graph, clause_id: str, base_text: str) -> str:
    """Append parent-section + referenced-clause context to a clause's text.

    The ReAct agent's get_referenced_clauses tool follows only REFERS_TO edges,
    so it never sees the *parent* section (e.g. 5.2 for 5.2.1) that carries the
    general requirements a scenario specialises. We inject that context (parent
    chain + transitive references) directly into the prompt so a test case for
    5.2.1 is always grounded in 5.2 and everything both of them reference.

    Pure/deterministic — no LLM — so it's unit-testable against a synthetic graph.
    """
    if graph is None or not hasattr(graph, "get_generation_context"):
        return base_text
    context_clauses = graph.get_generation_context(clause_id)
    extra = [c for c in context_clauses if getattr(c, "clause_id", None) != clause_id]
    if not extra:
        return base_text
    ctx = "\n\n".join(
        f"[{c.clause_id}] {c.title or ''}\n{c.text or ''}".strip() for c in extra
    )
    return (
        base_text
        + "\n\n[RELATED REGULATORY CONTEXT — the parent section(s) and referenced "
        "clauses this scenario depends on. You MUST take these into account when "
        "generating test cases:]\n"
        + ctx
    )


def generate_for_clause(
    clause_id: str,
    graph,
    llm_config,
    clause_map: dict[str, ClauseType],
    row_filter: str | None = None,
    progress_cb=None,
) -> list:
    """Run the ReAct agent on a single clause and return TestCase objects.

    row_filter: natural-language constraint appended to clause text, e.g.
        "Only generate test cases for rows where Vehicle condition is 'unladen'
         and speed is 60 km/h. Skip all other rows."
    progress_cb: optional callable(message: str) for UI progress updates.
    """
    from regulatory_testgen.agent import analyse_clause, build_agent

    clause = clause_map.get(clause_id)
    if clause is None:
        logger.warning("Clause %s not found in clause_map", clause_id)
        return []

    title = clause.title or ""
    text = clause.text or ""

    # Ground the generation in the parent section + referenced clauses.
    text = build_generation_context_text(graph, clause_id, text)

    if row_filter:
        text = (
            text
            + "\n\n[GENERATION CONSTRAINT: "
            + row_filter.strip()
            + " Do not generate test cases for any other table rows.]"
        )

    if progress_cb:
        progress_cb(f"Running ReAct agent on clause {clause_id}…")

    agent = build_agent(graph, llm_config)
    cases = analyse_clause(agent, clause_id, title, text)

    if not cases:
        if progress_cb:
            progress_cb(f"ReAct loop empty — retrying {clause_id} with direct prompt…")
        agent2 = build_agent(graph, llm_config)
        cases = analyse_clause(agent2, clause_id, title, text, direct=True)

    return cases


def generate_for_clauses(
    clause_ids: list[str],
    graph,
    llm_config,
    clause_map: dict[str, ClauseType],
    row_filter: str | None = None,
    progress_cb=None,
) -> list:
    """Generate test cases for multiple clauses sequentially. Returns merged list."""
    all_cases = []
    for i, cid in enumerate(clause_ids, 1):
        if progress_cb:
            progress_cb(f"[{i}/{len(clause_ids)}] Generating for {cid}…")
        cases = generate_for_clause(cid, graph, llm_config, clause_map, row_filter, progress_cb)
        logger.info("Clause %s → %d test case(s)", cid, len(cases))
        all_cases.extend(cases)
    return all_cases
