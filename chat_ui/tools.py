"""
tools.py — Tool definitions and handlers for the conversational assistant.

The assistant (app.py) calls these tools to answer questions about UN R152
and to trigger test case generation. Tool results are plain strings returned
to the LLM; generated test cases are also stored in session state for rendering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)

TESTABLE_CATEGORIES = {"obligation", "test_execution", "test_procedure", "performance_data"}
TEST_RELEVANT_CATEGORIES = TESTABLE_CATEGORIES | {"test_condition", "test_setup"}

# ── OpenAI-style tool schema ─────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_clauses",
            "description": (
                "Search UN R152 clauses by keyword or topic. "
                "Returns matching clause IDs, titles, categories, and text previews. "
                "Use this to find which clauses are relevant to a user's question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic or keyword to search for"},
                    "testable_only": {
                        "type": "boolean",
                        "description": "If true, only return testable clauses (obligation / test_procedure / performance_data). Default false.",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_clause",
            "description": (
                "Read a specific clause by ID. Returns its full text, category, section path, "
                "parent section, direct sub-clauses, and cross-referenced clauses. "
                "Use this to read the exact requirement your answer relies on."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "clause_id": {"type": "string", "description": "e.g. '5.2.1' or '6.1'"},
                },
                "required": ["clause_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_test_procedures",
            "description": (
                "Get the test-procedure and test-condition clauses needed to actually "
                "RUN a test for a requirement (test conditions, setup, execution steps). "
                "Ordinary clause retrieval deliberately omits these; call this tool when "
                "you are writing the test steps and need the procedures the requirement "
                "is tested under. (Performance-data tables are part of the requirement "
                "itself and come back with normal retrieval / get_performance_table.)"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "clause_id": {
                        "type": "string",
                        "description": "The requirement clause being tested, e.g. '5.2.1'.",
                    },
                },
                "required": ["clause_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_structure",
            "description": (
                "Get a hierarchical outline of the document. With no argument, returns the "
                "top-level sections and their direct children (titles + testable counts). "
                "Pass section=\"<clause_id>\" to expand ONE section's full subtree to every "
                "depth — use this to see, e.g., the specific scenarios nested under a "
                "requirements section. Start here for 'what is this about', 'how is it "
                "structured', or 'what scenarios/topics does it cover' questions, then drill in."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": "Optional clause ID to expand fully, e.g. '5' or '5.2'.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_performance_table",
            "description": (
                "Get the performance table rows for a clause that contains test parameters "
                "(speeds, thresholds, vehicle conditions, etc.). "
                "Use this to show what parameter combinations a clause covers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "clause_id": {"type": "string"},
                },
                "required": ["clause_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_existing_test_cases",
            "description": (
                "List the test cases that have already been generated and saved in the pipeline output. "
                "Returns test IDs, titles, and source clause IDs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "clause_id_filter": {
                        "type": "string",
                        "description": "Optional: only return test cases for this specific clause ID.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_test_cases",
            "description": (
                "Generate NEW structured test cases for the specified clauses using the ReAct agent pipeline. "
                "This takes 1–5 minutes depending on the number of clauses and table rows. "
                "Only call this AFTER the user has explicitly confirmed what they want generated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "clause_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of clause IDs to generate test cases for",
                    },
                    "row_filter": {
                        "type": "string",
                        "description": (
                            "Optional natural-language constraint on which table rows to generate for, "
                            "e.g. 'only unladen vehicles at 60 km/h'. Leave empty for all rows."
                        ),
                        "default": "",
                    },
                },
                "required": ["clause_ids"],
            },
        },
    },
]


# ── Tool handlers ─────────────────────────────────────────────────────────────

def handle_search_clauses(
    query: str,
    testable_only: bool = False,
    clause_index: dict = {},
    llm_config=None,
) -> str:
    """Use the LLM to find relevant clauses for a query (accurate, not keyword-based)."""
    from retrieval import find_relevant_clauses

    if llm_config is None:
        return "Search unavailable: LLM config not provided."

    ids = find_relevant_clauses(
        query, llm_config, clause_index,
        top_k=12, testable_only=testable_only,
    )

    if not ids:
        return (
            f"No clauses found matching '{query}'. "
            "Try different keywords, or use get_document_structure to browse the regulation."
        )

    # NOTE: search results are *candidates*, not citations. We deliberately do
    # NOT push them into last_retrieved_clause_ids — otherwise the Sources panel
    # would cite every clause the model merely browsed (up to 12), not the ones
    # the answer actually used. Only clauses the model then *reads* (get_clause /
    # get_performance_table) or *generates from* are treated as citations.
    st.session_state.setdefault("last_search_candidates", []).extend(ids)

    lines = [f"Found {len(ids)} relevant clause(s) for '{query}':\n"]
    for cid in ids:
        info = clause_index.get(cid, {})
        cat = info.get("category", "?")
        testable = info.get("testable", cat in TESTABLE_CATEGORIES)
        role = (
            "TESTABLE" if testable
            else "context" if cat in TEST_RELEVANT_CATEGORIES
            else "example" if info.get("force") == "example"
            else "—"
        )
        crumb = _breadcrumb(cid, clause_index)
        crumb_str = f"  _(under: {crumb})_" if crumb else ""
        lines.append(f"• **{cid}** [{cat}] ({role}): {info.get('title') or '(no title)'}{crumb_str}")
        preview = (info.get("text") or info.get("text_preview") or "")
        if preview:
            lines.append(f"  > {preview[:200].strip()}")
    return "\n".join(lines)


def _breadcrumb(clause_id: str, clause_index: dict) -> str:
    """Titles of a clause's ancestor sections, nearest-last, e.g.
    'Annex 3 / 5' -> 'Annex 3'; '5.2.1' -> 'Specifications > Specific Requirements'."""
    crumbs: list[str] = []
    cid = clause_id
    seen = set()
    while True:
        info = clause_index.get(cid)
        parent = info.get("parent") if info else None
        if not parent or parent in seen:
            break
        seen.add(parent)
        pinfo = clause_index.get(parent)
        crumbs.append((pinfo.get("title") if pinfo else None) or parent)
        cid = parent
    return " > ".join(reversed(crumbs))


def _children_map(clause_index: dict) -> dict[str, list[str]]:
    """{parent_id: [child_ids sorted naturally]} from the explicit `parent` field."""
    from normalize import natkey
    kids: dict[str, list[str]] = {}
    for cid, info in clause_index.items():
        parent = info.get("parent")
        if parent:
            kids.setdefault(parent, []).append(cid)
    for parent in kids:
        kids[parent].sort(key=natkey)
    return kids


def _direct_children(clause_id: str, children: dict[str, list[str]]) -> list[str]:
    return children.get(clause_id, [])


def handle_get_clause(clause_id: str, clause_index: dict = {}, graph=None) -> str:
    info = clause_index.get(clause_id)
    if not info:
        return f"Clause '{clause_id}' not found. Use get_document_structure to see valid IDs."

    # Full text: prefer the graph's clause (complete), fall back to the index.
    text = (info.get("text") or info.get("text_preview") or "").strip()
    if graph is not None:
        gc = graph.get_clause(clause_id)
        if gc is not None and (gc.text or "").strip():
            text = gc.text.strip()

    cat = info.get("category", "?")
    testable = info.get("testable", cat in TESTABLE_CATEGORIES)
    force = info.get("force", "binding")
    force_note = f"  |  Force: {force}" if force != "binding" else ""
    lines = [
        f"**Clause {clause_id} — {info.get('title') or '(untitled)'}**",
        f"Category: {cat}  |  Testable: {'yes' if testable else 'no'}{force_note}",
    ]
    crumb = _breadcrumb(clause_id, clause_index)
    if crumb:
        lines.append(f"Section path: {crumb}")
    parent = info.get("parent")
    if parent:
        pinfo = clause_index.get(parent)
        lines.append(f"Parent: {parent} — {(pinfo.get('title') if pinfo else None) or '(untitled)'}")

    lines.append("")
    lines.append(text or "(no body text — this is a section heading; see its sub-clauses below)")

    children = _children_map(clause_index)
    kids = _direct_children(clause_id, children)
    if kids:
        lines.append("\nDirect sub-clauses:")
        for k in kids:
            k_info = clause_index[k]
            lines.append(f"  • {k} — {k_info.get('title') or '(untitled)'} [{k_info.get('category', '?')}]")

    refs = [r for r in (info.get("references") or []) if r in clause_index and r != clause_id]
    if refs:
        lines.append("\nCross-references (this clause points to):")
        for r in refs:
            lines.append(f"  → {r} — {clause_index[r].get('title') or '(untitled)'}")

    return "\n".join(lines)


def handle_get_test_procedures(clause_id: str, clause_index: dict = {}) -> str:
    """Step-2 retrieval, exposed to the model: the test-procedure clauses needed to
    actually run a test for `clause_id`. Uses the SAME fixed traversal as the
    obligation retrieval (capped, bidirectional, transitive) but WITHOUT excluding
    testing-labelled clauses (PROCEDURE_STRATEGY), then keeps only the testing
    clauses (conditions / setup / execution / procedure). Performance-data tables
    are part of the requirement, so they arrive with normal retrieval, not here."""
    from regulatory_testgen.context_expand import (
        generation_context_from_index, PROCEDURE_STRATEGY, PROC_CATEGORIES)

    if clause_id not in clause_index:
        return (f"Clause '{clause_id}' not found. Use get_document_structure to "
                "see valid IDs.")
    ctx = generation_context_from_index(clause_index, clause_id, **PROCEDURE_STRATEGY)
    procs = [c for c in ctx
             if (clause_index.get(c) or {}).get("category") in PROC_CATEGORIES]
    if not procs:
        return (f"No test-procedure/condition clauses are reachable from {clause_id}. "
                "This requirement may rely on a document-wide test procedure — check "
                "the §6-style test sections with get_document_structure.")
    lines = [f"Test-procedure context for testing **{clause_id}** "
             f"({len(procs)} clause(s)):"]
    for c in procs:
        info = clause_index[c]
        text = (info.get("text") or info.get("text_preview") or "").strip()
        lines.append(f"\n**{c} — {info.get('title') or '(untitled)'}** "
                     f"[{info.get('category', '?')}]")
        lines.append(text[:800] or "(no body text — see its sub-clauses)")
    return "\n".join(lines)


def handle_get_document_structure(clause_index: dict = {}, section: str = "") -> str:
    from normalize import natkey

    children = _children_map(clause_index)

    def _title(cid: str) -> str:
        info = clause_index.get(cid, {})
        return info.get("title") or "(untitled)"

    def _cat(cid: str) -> str:
        return clause_index.get(cid, {}).get("category", "?")

    def _count_testable(cid: str) -> int:
        info = clause_index.get(cid, {})
        total = 1 if info.get("testable", _cat(cid) in TESTABLE_CATEGORIES) else 0
        for child in _direct_children(cid, children):
            total += _count_testable(child)
        return total

    # ── Drill-down: full subtree of one section ──────────────────────────────
    if section:
        if section not in clause_index:
            return (
                f"Section '{section}' not found. Call get_document_structure with no "
                "argument to see the valid top-level section IDs."
            )
        lines = [f"**Full subtree of {section} — {_title(section)}**\n"]

        def _walk(cid: str, depth: int) -> None:
            indent = "  " * depth
            lines.append(f"{indent}- **{cid}** {_title(cid)}  [{_cat(cid)}]")
            for child in _direct_children(cid, children):
                _walk(child, depth + 1)

        _walk(section, 0)
        return "\n".join(lines)

    # ── Overview: top-level sections + their direct children ─────────────────
    roots = sorted(
        [cid for cid, info in clause_index.items() if not info.get("parent")],
        key=natkey,
    )
    lines = [
        "**Document structure** — top-level sections and their direct children.",
        "Call get_document_structure with section=\"<id>\" to expand a section fully "
        "(e.g. to see nested scenarios/sub-topics). Annex sections are namespaced, "
        "e.g. \"Annex 3 / 5\" is section 5 *within* Annex 3 (distinct from body §5).\n",
    ]
    for root in roots:
        tc = _count_testable(root)
        tc_str = f" — {tc} testable" if tc > 0 else ""
        lines.append(f"**{root}** {_title(root)}{tc_str}")
        for child in _direct_children(root, children):
            c_tc = _count_testable(child)
            c_tc_str = f" ({c_tc} testable)" if c_tc > 0 else ""
            lines.append(f"  • {child}: {_title(child)}{c_tc_str}")
            grand = _direct_children(child, children)
            if grand:
                lines.append(f"      ↳ {len(grand)} sub-clause(s) — expand with section=\"{child}\"")
    return "\n".join(lines)


def handle_get_performance_table(clause_id: str, graph=None) -> str:
    if graph is None:
        return "Graph not available."
    tables = graph.get_tables(clause_id)
    if not tables:
        return f"Clause {clause_id} has no performance table."

    parts = []
    for t in tables:
        if not t.rows:
            continue
        parts.append(f"**Table for {clause_id}** — {t.title or 'Performance table'}")
        parts.append(f"{len(t.rows)} rows × {len(t.headers)} columns: {', '.join(t.headers)}")
        parts.append("")
        for i, row in enumerate(t.rows, 1):
            parts.append(f"Row {i}: " + " | ".join(f"{k}={v}" for k, v in row.items()))

    return "\n".join(parts) if parts else f"Tables for {clause_id} are empty."


def handle_list_existing_test_cases(clause_id_filter: str = "", checkpoints_dir: Optional[Path] = None) -> str:
    if checkpoints_dir is None:
        checkpoints_dir = Path(__file__).parent.parent / "regulatory_testgen" / "output"

    tc_file = checkpoints_dir / "03_test_cases.json"
    if not tc_file.exists():
        return "No test cases have been generated yet."

    import json as _json
    cases = _json.loads(tc_file.read_text(encoding="utf-8"))

    if clause_id_filter:
        cases = [c for c in cases if clause_id_filter in c.get("source_clause_ids", [])]

    if not cases:
        return f"No existing test cases found{' for clause ' + clause_id_filter if clause_id_filter else ''}."

    lines = [f"**{len(cases)} existing test case(s)**{' for clause ' + clause_id_filter if clause_id_filter else ''}:\n"]
    for tc in cases[:30]:  # cap at 30 for readability
        src = ", ".join(tc.get("source_clause_ids", []))
        lines.append(f"• **{tc.get('test_id', '?')}** — {tc.get('title', '')}  _(source: {src})_")
    if len(cases) > 30:
        lines.append(f"… and {len(cases) - 30} more.")
    return "\n".join(lines)


def handle_generate_test_cases(
    clause_ids: list[str],
    row_filter: str = "",
    graph=None,
    llm_config=None,
    clause_map: dict = {},
) -> str:
    """Run the generation pipeline and store results in st.session_state."""
    from generator import generate_for_clause

    if not clause_ids:
        return "No clause IDs provided."

    results = []
    status_placeholder = st.empty()

    for i, cid in enumerate(clause_ids, 1):
        status_placeholder.info(f"⚙️ Generating test cases for **{cid}** ({i}/{len(clause_ids)})…")

        def _cb(msg: str):
            status_placeholder.info(f"⚙️ {msg}")

        cases = generate_for_clause(
            cid, graph, llm_config, clause_map,
            row_filter=row_filter or None,
            progress_cb=_cb,
        )
        results.extend(cases)
        logger.info("Clause %s → %d test case(s)", cid, len(cases))

    status_placeholder.empty()

    if not results:
        return "Generation completed but no test cases were produced. The clauses may not contain testable requirements."

    # Store for rendering by app.py
    dumped = [tc.model_dump() for tc in results]
    st.session_state.pending_results = dumped

    # Cite the clauses the ReAct agent actually drew each test case from
    # (source_clause_ids), falling back to the requested clause_ids. These are
    # real citations, so they feed the turn's Sources panel.
    cited: list[str] = []
    for tc in dumped:
        cited.extend(tc.get("source_clause_ids") or [])
    if not cited:
        cited = list(clause_ids)
    st.session_state.setdefault("last_retrieved_clause_ids", []).extend(cited)

    row_note = f" (filter: *{row_filter}*)" if row_filter else ""
    return (
        f"Generated **{len(results)} test case(s)** for {len(clause_ids)} clause(s){row_note}. "
        f"Results are shown below."
    )


# ── Dispatcher ───────────────────────────────────────────────────────────────

def dispatch_tool(
    name: str, arguments: dict, *, graph, llm_config, clause_map, clause_index, checkpoints_dir=None,
) -> str:
    """Route a tool call to the correct handler. Returns a string result."""
    try:
        if name == "search_clauses":
            return handle_search_clauses(
                arguments.get("query", ""),
                testable_only=arguments.get("testable_only", False),
                clause_index=clause_index,
                llm_config=llm_config,
            )
        elif name == "get_clause":
            cid = arguments.get("clause_id", "")
            if cid:
                st.session_state.setdefault("last_retrieved_clause_ids", []).append(cid)
            return handle_get_clause(cid, clause_index=clause_index, graph=graph)
        elif name == "get_test_procedures":
            cid = arguments.get("clause_id", "")
            if cid:
                st.session_state.setdefault("last_retrieved_clause_ids", []).append(cid)
            return handle_get_test_procedures(cid, clause_index=clause_index)
        elif name == "get_document_structure":
            return handle_get_document_structure(
                clause_index=clause_index,
                section=arguments.get("section", "") or "",
            )
        elif name == "get_performance_table":
            cid = arguments.get("clause_id", "")
            if cid:
                st.session_state.setdefault("last_retrieved_clause_ids", []).append(cid)
            return handle_get_performance_table(cid, graph=graph)
        elif name == "list_existing_test_cases":
            return handle_list_existing_test_cases(
                arguments.get("clause_id_filter", ""),
                checkpoints_dir=checkpoints_dir,
            )
        elif name == "generate_test_cases":
            clause_ids = arguments.get("clause_ids", [])
            # Citations are recorded inside the handler from each test case's
            # real source_clause_ids — not the raw request — so no extend here.
            return handle_generate_test_cases(
                clause_ids=clause_ids,
                row_filter=arguments.get("row_filter", ""),
                graph=graph,
                llm_config=llm_config,
                clause_map=clause_map,
            )
        else:
            return f"Unknown tool: {name}"
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return f"Tool {name} failed: {exc}"
