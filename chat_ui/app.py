"""
app.py — Conversational chat UI for UN R152 regulatory test case generation.

The LLM drives the entire interaction using function calling:
  - search_clauses         → find relevant clauses by topic
  - get_clause             → read a specific clause
  - get_document_structure → overview of the regulation's structure
  - get_performance_table  → inspect table rows for a clause
  - list_existing_test_cases → what's already been generated
  - generate_test_cases    → run the ReAct agent pipeline

No state machine. The model decides when to use tools and when to ask for
clarification, just like a chat with a domain expert.

Run from thesis-code root:
  source regulatory_testgen/venv/bin/activate
  streamlit run chat_ui/app.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

# ── sys.path (must precede local imports) ────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

from retrieval import load_clause_index, TESTABLE_CATEGORIES  # noqa: E402
from generator import load_graph_and_config  # noqa: E402
from tools import TOOL_DEFINITIONS, dispatch_tool  # noqa: E402
from ingest import ingest_document, SUPPORTED_SUFFIXES, UPLOAD_DIR, MINERU_API_URL  # noqa: E402

logging.basicConfig(level=logging.WARNING)

DEFAULT_CHECKPOINTS_DIR = _ROOT / "regulatory_testgen" / "output"
DEFAULT_DOC_NAME = "UN R152 (default)"

# Selectable LLM backends. Hosted providers (e.g. OpenAI) set requires_key so the
# sidebar shows a key field. Any OpenAI-compatible endpoint works via "Custom…"
# (e.g. a self-hosted vLLM server — leave the key field blank for those).
#   requires_key : show an API-key field for this backend
#   api_key_env  : env var to prefill that field from (never written to disk)
MODEL_PRESETS: dict[str, dict] = {
    "OpenAI · gpt-4.1-mini": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4.1-mini",
        "requires_key": True,
        "api_key_env": "OPENAI_API_KEY",
    },
}
CUSTOM_CHOICE = "Custom…"
CHAT_MAX_TOKENS = 4096  # headroom so reasoning models don't truncate to an empty answer

BASE_SYSTEM_PROMPT = """You are a regulatory expert assistant helping engineers navigate a regulatory document and generate executable test cases from it.

You help engineers understand the regulation, navigate its structure, and generate executable test cases.

CLAUSE CATEGORIES — this is important for understanding the regulation:
  WHAT must be tested (→ generate test cases for these):
    • obligation        — "the system SHALL do X" — behavioural mandates (Section 5, Section 8)
    • performance_data  — tables of pass/fail thresholds and scenario parameters (Annex 3)
    • test_execution    — also testable; concrete test run procedure (Section 6.4–6.9)

  HOW compliance is verified (→ use as context, not standalone test cases):
    • test_condition    — environmental prerequisites: road surface (PBC), temperature, lighting, slope (Section 6.1)
    • test_setup        — vehicle/target preparation: test mass, pre-conditioning, target specs (Section 6.2–6.3)

  Non-testable:
    • definition, scope, administrative, formatting, informative

KEY INSIGHT: Section 5 defines WHAT the AEBS must achieve; Section 6 defines HOW to verify it.
A complete test case references a Section 5 obligation AND incorporates the Section 6 test conditions,
setup, and execution procedure. The agent does this automatically via get_referenced_clauses.

TOOLS AT YOUR DISPOSAL:
- get_document_structure: outline of the document. No argument → top-level sections + their
  direct children. section="<id>" → expand ONE section's full subtree to every depth.
- search_clauses: find clauses by keyword/topic (LLM-based, returns candidates with breadcrumbs)
- get_clause: read a specific clause's FULL text, plus its parent, sub-clauses, and cross-references
- get_test_procedures: the test-condition/setup/execution clauses needed to RUN a test for a
  requirement (normal retrieval omits these; call it when writing the test steps)
- get_performance_table: inspect the parameter rows in a clause's table
- list_existing_test_cases: see what test cases have already been generated
- generate_test_cases: run the ReAct agent to create NEW test cases (1–5 min — confirm first)

HOW TO EXPLORE (do this before answering — never rely on memory):
- "What is this document about?" / "How is it structured?" → call get_document_structure (no arg)
  to see the sections, then read the 2–3 most relevant top-level sections with get_clause to
  ground your summary.
- "What scenarios / topics / cases does it cover?" → get_document_structure (no arg) to find the
  section that holds them, then get_document_structure(section="<that id>") to expand its full
  subtree, then get_clause on the specific sub-clauses to confirm details before answering.
  Do NOT stop at the first level — the specific items are usually nested one or two levels deeper.
- Prefer MULTIPLE tool calls: outline → drill into the section → read the specific clauses. A good
  answer is usually preceded by 2–5 tool calls, not one.

HOW TO BEHAVE:
- Be conversational and concise, but fully answer the question — don't give a one-liner when the
  user asked what something covers; enumerate the actual items you found.
- Use search_clauses or get_document_structure before answering questions about clause content.
- For generate_test_cases: confirm the exact clauses and any row filter BEFORE calling it.
  Tell the user how many clauses, and for table-driven clauses, how many rows are involved.
- After generation, briefly summarize. The UI renders full test cases automatically.
- If the user wants specific table rows only, use the row_filter parameter.
- The UI cites your answer automatically, using ONLY the clauses you actually open with
  get_clause / get_performance_table (or generate test cases from). Clauses that merely
  appear in search_clauses results are NOT cited, and neither are clause IDs you just mention
  in prose. So: read the specific clauses your answer relies on with get_clause before
  stating what they require — that both grounds your answer and produces correct citations.
  For a broad "what is this document about" overview, get_document_structure is enough and no
  Sources panel is expected. Never answer clause content from memory."""

# Extra context appended only when the built-in R152 document is active — a
# custom upload has no such fixed structure, so the LLM should rely on
# get_document_structure / search_clauses instead.
R152_SYSTEM_PROMPT_ADDENDUM = """

CURRENTLY LOADED DOCUMENT: UN Regulation No. 152 — Advanced Emergency Braking Systems (AEBS)
for M1/N1 (passenger cars and light commercial vehicles).

REGULATION STRUCTURE:
- Section 1: Scope
- Section 2: Definitions (2.1–2.15: AEBS, collision warning, targets, etc.)
- Section 3: Application for type approval
- Section 4: Approval marks
- Section 5: Performance requirements — the WHAT
  • 5.1: General AEBS requirements (warning, braking, speed range, error avoidance)
  • 5.2: Moving target requirements (5.2.1 vehicle, 5.2.2 pedestrian, 5.2.3 bicycle)
  • 5.3: Stationary target requirements
  • 5.4–5.6: Additional conditions (cut-in, intersection, night)
- Section 6: Test conditions and procedures — the HOW
  • 6.1: Test environment (road, temperature, visibility, lighting)
  • 6.2–6.3: Vehicle/target setup
  • 6.4–6.9: Test execution procedures per target type
  • 6.10: Robustness
- Annex 3: Performance tables (speed × load condition × target type matrices)
"""


def build_system_prompt(doc_name: str) -> str:
    if doc_name == DEFAULT_DOC_NAME:
        return BASE_SYSTEM_PROMPT + R152_SYSTEM_PROMPT_ADDENDUM
    return (
        BASE_SYSTEM_PROMPT
        + f"\n\nCURRENTLY LOADED DOCUMENT: {doc_name} (user-uploaded). "
        "Use get_document_structure and search_clauses to learn its structure — "
        "do not assume UN R152's section layout."
    )


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="R152 Test Case Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load default resources once ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading regulatory graph…")
def _load_default():
    graph, base_llm_config, clause_map, source_info = load_graph_and_config()
    clause_index = load_clause_index()
    return graph, base_llm_config, clause_map, clause_index, source_info

_default_graph, _base_llm_config, _default_clause_map, _default_clause_index, _default_source_info = _load_default()


def _resolve_llm_config(base_url: str, model: str, api_key: str | None = None):
    """Clone the loaded LLMConfig but point it at the chosen server/model/key.

    All LLM callers (chat, retrieval, classification, the generation agent) read
    base_url/model/api_key from this single object, so overriding it here switches
    every call at once. Empty key → "EMPTY" (keyless vLLM ignores it, and the
    OpenAI client rejects a blank key)."""
    return _base_llm_config.model_copy(
        update={"base_url": base_url, "model": model, "api_key": api_key or "EMPTY"}
    )


# llm_config / client are (re)built from the sidebar selection just below the
# sidebar block, before any turn runs. Seed with the loaded defaults so module
# import order is safe.
llm_config = _base_llm_config
client = OpenAI(base_url=llm_config.base_url, api_key=llm_config.api_key)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # OpenAI-format message dicts
if "pending_results" not in st.session_state:
    st.session_state.pending_results = []   # test cases waiting to be rendered
if "active_doc_name" not in st.session_state:
    st.session_state.active_doc_name = DEFAULT_DOC_NAME
    st.session_state.graph = _default_graph
    st.session_state.clause_map = _default_clause_map
    st.session_state.clause_index = _default_clause_index
    st.session_state.checkpoints_dir = DEFAULT_CHECKPOINTS_DIR
    st.session_state.source_info = _default_source_info
if "processed_upload_id" not in st.session_state:
    st.session_state.processed_upload_id = None
if "upload_error" not in st.session_state:
    st.session_state.upload_error = None
if "last_retrieved_clause_ids" not in st.session_state:
    st.session_state.last_retrieved_clause_ids = []   # clause IDs touched by tools this turn
if "turn_sources" not in st.session_state:
    st.session_state.turn_sources = {}   # {message_index: [clause_ids]} — persists across reruns
if "turn_test_cases" not in st.session_state:
    st.session_state.turn_test_cases = {}   # {message_index: [test_case_dicts]} — persists across reruns
if "turn_tool_calls" not in st.session_state:
    st.session_state.turn_tool_calls = {}   # {message_index: [{name, args, result}]} — for inspection
if "turn_reasoning" not in st.session_state:
    st.session_state.turn_reasoning = {}    # {message_index: reasoning_text} — model's thinking

graph = st.session_state.graph
clause_map = st.session_state.clause_map
clause_index = st.session_state.clause_index


def _switch_document(upload_id: str, uploaded_file) -> None:
    """Save the uploaded file, run the ingest pipeline with a live progress bar,
    and swap the active graph/clause_map/clause_index in session state."""
    suffix = Path(uploaded_file.name).suffix.lower()
    doc_dir = UPLOAD_DIR / upload_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    input_path = doc_dir / f"source{suffix}"
    input_path.write_bytes(uploaded_file.getvalue())

    progress_bar = st.progress(0.0, text="Starting ingestion…")

    def _progress_cb(frac: float, message: str) -> None:
        progress_bar.progress(min(max(frac, 0.0), 1.0), text=message)

    try:
        new_graph, new_clause_map, new_clause_index, new_source_info = ingest_document(
            input_path, llm_config, doc_dir / "output", progress_cb=_progress_cb,
        )
    except Exception as exc:
        logging.exception("Document ingestion failed")
        st.session_state.upload_error = str(exc)
        progress_bar.empty()
        return

    st.session_state.graph = new_graph
    st.session_state.clause_map = new_clause_map
    st.session_state.clause_index = new_clause_index
    st.session_state.checkpoints_dir = doc_dir / "output"
    st.session_state.active_doc_name = uploaded_file.name
    st.session_state.processed_upload_id = upload_id
    st.session_state.upload_error = None
    st.session_state.source_info = new_source_info
    st.session_state.messages = []
    st.session_state.pending_results = []
    st.session_state.last_retrieved_clause_ids = []
    st.session_state.turn_sources = {}
    st.session_state.turn_test_cases = {}
    st.session_state.turn_tool_calls = {}
    st.session_state.turn_reasoning = {}
    progress_bar.empty()
    st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚗 Regulatory Test Assistant")
    st.caption(f"Active document: **{st.session_state.active_doc_name}**")
    st.divider()

    st.markdown("**Model / server**")
    _choice = st.selectbox(
        "LLM backend",
        list(MODEL_PRESETS) + [CUSTOM_CHOICE],
        key="model_choice",
        label_visibility="collapsed",
    )
    if _choice == CUSTOM_CHOICE:
        _sel_base_url = st.text_input(
            "Base URL", value="", key="custom_base_url",
            help="Any OpenAI-compatible endpoint, e.g. https://api.openai.com/v1 "
                 "or a self-hosted vLLM server's /v1 URL.",
        ).strip()
        _sel_model = st.text_input(
            "Model ID", value="", key="custom_model"
        ).strip()
        _needs_key = True   # unknown provider — offer a key; leave blank for keyless vLLM
        _key_env = "OPENAI_API_KEY"
    else:
        _preset = MODEL_PRESETS[_choice]
        _sel_base_url = _preset["base_url"]
        _sel_model = _preset["model"]
        _needs_key = _preset.get("requires_key", False)
        _key_env = _preset.get("api_key_env", "OPENAI_API_KEY")

    if _needs_key:
        _sel_api_key = st.text_input(
            "API key",
            value=os.environ.get(_key_env, ""),
            type="password",
            key="provider_api_key",
            help=f"Sent as the Bearer token. Prefilled from ${_key_env} if set. "
                 "Kept in session only — never written to disk. Leave blank for a keyless vLLM server.",
        ).strip()
    else:
        _sel_api_key = _base_llm_config.api_key   # "EMPTY" — keyless vLLM

    st.caption(f"→ `{_sel_model}` @ `{_sel_base_url}`")
    if _needs_key and _choice != CUSTOM_CHOICE and not _sel_api_key:
        st.warning("This backend needs an API key.", icon="🔑")
    st.divider()

    st.markdown("**Upload a regulatory document**")
    if MINERU_API_URL:
        st.caption(f"🖥️ MinerU conversion via remote GPU: `{MINERU_API_URL}`")
    else:
        st.caption("💻 MinerU conversion runs locally (CPU) — set `MINERU_API_URL` to offload to a GPU server.")
    uploaded_file = st.file_uploader(
        "Replaces the active document — PDF, DOCX, PPTX, XLSX, image, or Markdown.",
        type=[s.lstrip(".") for s in sorted(SUPPORTED_SUFFIXES)],
    )
    if uploaded_file is not None:
        upload_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:16]
        if upload_id != st.session_state.processed_upload_id:
            _switch_document(upload_id, uploaded_file)

    if st.session_state.upload_error:
        st.error(f"Ingestion failed: {st.session_state.upload_error}")

    _source_info = st.session_state.get("source_info") or {}
    if _source_info.get("pdf_path") is None:
        st.caption("ℹ️ No source PDF for highlighting (Markdown upload or non-PDF input).")
    elif not _source_info.get("clause_pages"):
        st.caption("ℹ️ Source PDF available, but clause-to-page tracing found no matches.")
    else:
        st.caption("✅ Source PDF highlighting available.")

    st.divider()

    stats = graph.stats()
    testable_n = sum(
        1 for v in clause_index.values()
        if v.get("testable", v.get("category") in TESTABLE_CATEGORIES)
    )
    st.metric("Clauses in graph", stats["clauses"])
    st.metric("Testable clauses", testable_n)
    st.metric("Cross-references", stats.get("refers_to_edges", stats.get("refers_to", "—")))

    st.divider()
    st.markdown(
        "**Example questions:**\n"
        "- *How is the document structured?*\n"
        "- *What test cases exist for braking distance?*\n"
        "- *Show me clause 5.2.1*\n"
        "- *Generate test cases for stationary target at 60 km/h*\n"
        "- *What performance tables does Annex 3 have?*"
    )
    st.divider()

    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_results = []
        st.session_state.turn_sources = {}
        st.session_state.turn_test_cases = {}
        st.session_state.turn_tool_calls = {}
        st.session_state.turn_reasoning = {}
        st.rerun()

    st.caption(f"Model: `{_sel_model.split('/')[-1]}`")


# Rebuild the active LLM config + client from the sidebar selection. Done here
# (after the sidebar renders) so every downstream call this run — chat, retrieval,
# classification, generation — uses the chosen server/model.
llm_config = _resolve_llm_config(_sel_base_url, _sel_model, _sel_api_key)
client = OpenAI(base_url=llm_config.base_url, api_key=llm_config.api_key)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.dialog("📄 Highlighted source", width="large")
def _highlight_dialog(pdf_path_str: str, clause_ids: list[str], clause_pages: dict, cache_key: str) -> None:
    """Modal showing the cited pages as highlighted images. Rendering is cached
    per cache_key so interacting inside the modal (e.g. the download button)
    doesn't re-rasterise — the modal body re-runs on every such interaction."""
    from pdf_highlight import render_highlighted_page_images, render_highlighted_pdf

    img_key = f"{cache_key}_imgs"
    dl_key = f"{cache_key}_dl"
    if img_key not in st.session_state:
        with st.spinner("Rendering highlighted pages…"):
            images = render_highlighted_page_images(
                Path(pdf_path_str), clause_ids, clause_pages, zoom=1.5,
            )
            pdf = render_highlighted_pdf(Path(pdf_path_str), clause_ids, clause_pages)
            st.session_state[img_key] = images
            st.session_state[dl_key] = pdf[0] if pdf else None

    images = st.session_state[img_key]
    if not images:
        st.warning("Couldn't locate these clauses in the source PDF.")
        return

    page_nums = [p for p, _ in images]
    st.caption(f"Original page(s): {', '.join(map(str, page_nums))} — highlights mark the cited passages.")
    if st.session_state.get(dl_key):
        st.download_button(
            "⬇ Download highlighted PDF",
            data=st.session_state[dl_key],
            file_name="highlighted_source.pdf",
            mime="application/pdf",
            key=f"{cache_key}_dlbtn",
        )
    for page_no, png in images:
        st.image(png, caption=f"Page {page_no}", use_container_width=True)


def _render_sources(clause_ids: list[str], key_prefix: str) -> None:
    """List which clauses/pages an answer drew on, with an optional highlighted
    source view. No-op if nothing was retrieved or no source PDF is available.

    The highlighted source is shown as page *images* (rasterised with the
    highlight overlay baked in), not an embedded PDF — a rendered image always
    displays in the browser, whereas an embedded/base64 PDF is silently blocked
    by many browsers inside Streamlit's sandboxed iframe. The original PDF is
    never modified; a download button offers the annotated copy for offline use.
    """
    clause_ids = list(dict.fromkeys(cid for cid in clause_ids if cid))  # de-dup, keep order
    if not clause_ids:
        return

    source_info = st.session_state.get("source_info") or {}
    pdf_path = source_info.get("pdf_path")
    clause_pages = source_info.get("clause_pages") or {}

    # Only clauses we could actually locate in the PDF are highlightable.
    locatable = [cid for cid in clause_ids if clause_pages.get(cid)]

    with st.expander(f"📄 Sources ({len(clause_ids)} clause(s))", expanded=False):
        for cid in clause_ids:
            info = clause_index.get(cid, {})
            pages = sorted({e["page"] + 1 for e in clause_pages.get(cid, [])})
            page_str = f"p. {', '.join(map(str, pages))}" if pages else "page not located"
            st.markdown(f"- **{cid}** — {info.get('title') or '(no title)'} _({page_str})_")

        if pdf_path is None:
            st.caption(
                "No source PDF available for highlighting — this document was "
                "uploaded as Markdown directly, or has no traceable PDF."
            )
            return
        if not locatable:
            st.caption(
                "None of these clauses could be located in the source PDF for "
                "highlighting (their text didn't align to a page region)."
            )
            return

        if st.button("🔍 View highlighted pages", key=f"{key_prefix}_view_btn"):
            # Drop any stale cached render so the modal rasterises fresh, then
            # open it. Calling the @st.dialog function opens the modal; Streamlit
            # keeps re-running its body on interactions until the user closes it.
            st.session_state.pop(f"{key_prefix}_imgs", None)
            st.session_state.pop(f"{key_prefix}_dl", None)
            _highlight_dialog(str(pdf_path), locatable, clause_pages, key_prefix)


def _render_test_cases(cases: list[dict], key_prefix: str = "tc"):
    """Render generated test cases as expandable cards with a download button."""
    import pandas as pd

    st.markdown(f"#### Generated test cases ({len(cases)})")
    for i, tc in enumerate(cases):
        with st.expander(f"**{tc.get('test_id', '?')}** — {tc.get('title', '')}", expanded=False):
            st.markdown(f"**Scenario:**\n\n{tc.get('scenario', '')}")

            preconds = tc.get("preconditions") or []
            if preconds:
                st.markdown("**Preconditions:**")
                for p in preconds:
                    st.markdown(f"- {p}")

            steps = tc.get("test_steps") or []
            if steps:
                st.markdown("**Test steps:**")
                for s in steps:
                    st.markdown(f"- {s}")

            expected = tc.get("expected_behavior") or []
            if expected:
                st.markdown("**Expected behavior:**")
                for e in expected:
                    st.markdown(f"- {e}")

            params = tc.get("parameters") or {}
            if params:
                st.markdown("**Parameters:**")
                st.dataframe(pd.DataFrame([params]), use_container_width=True, hide_index=True)

            st.caption(f"Source: {', '.join(tc.get('source_clause_ids', []))}")
            _render_sources(
                tc.get("source_clause_ids", []),
                key_prefix=f"{key_prefix}_tc{i}_{tc.get('test_id', '')}",
            )

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇ Download JSON",
            data=json.dumps(cases, indent=2, ensure_ascii=False),
            file_name="test_cases.json",
            mime="application/json",
            use_container_width=True,
            key=f"{key_prefix}_dl_json",
        )
    with col2:
        rows = [["test_id", "title", "scenario"]]
        for tc in cases:
            rows.append([
                tc.get("test_id", ""),
                tc.get("title", ""),
                tc.get("scenario", "").replace("\n", " "),
            ])
        import csv, io
        buf = io.StringIO()
        csv.writer(buf).writerows(rows)
        st.download_button(
            "⬇ Download CSV",
            data=buf.getvalue(),
            file_name="test_cases.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{key_prefix}_dl_csv",
        )


def _turn_citations(read_ids: list[str]) -> list[str]:
    """A turn's citations = only the clauses the model actually READ
    (get_clause / get_performance_table) or GENERATED test cases from.

    We deliberately do NOT mine the answer text for clause IDs: prose mentions
    ranges and examples ("the procedures 6.4-6.9", "see Section 5") that are
    descriptive, not sources — extracting those produced wrong citations. And
    we do NOT include raw search candidates (everything the model merely
    browsed). What's left is grounded: clauses the model deliberately opened."""
    return list(dict.fromkeys(cid for cid in read_ids if cid))


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _extract_reasoning(msg) -> tuple[str, str]:
    """Return (reasoning_text, visible_content) for an assistant message.

    vLLM exposes chain-of-thought as `reasoning_content` when a reasoning parser
    is configured; some models instead wrap it in <think>…</think> inside the
    content. Handle both so the thinking is captured and stripped from the answer.
    """
    reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
    content = getattr(msg, "content", None) or ""
    m = _THINK_RE.search(content)
    if m:
        reasoning = (reasoning + "\n\n" + m.group(1).strip()).strip()
        content = _THINK_RE.sub("", content).strip()
    return reasoning, content


def _render_trace(idx: int) -> None:
    """Persistent, inspectable view of a turn's reasoning and tool calls.

    Unlike the live st.status box (which is ephemeral), this reads from session
    state so the full trace stays available in history for debugging where the
    model went wrong."""
    reasoning = st.session_state.turn_reasoning.get(idx)
    calls = st.session_state.turn_tool_calls.get(idx)
    if reasoning:
        with st.expander("🧠 Model reasoning", expanded=False):
            st.markdown(reasoning)
    if calls:
        with st.expander(f"🔧 Tool calls ({len(calls)})", expanded=False):
            for i, c in enumerate(calls, 1):
                st.markdown(f"**{i}. `{c['name']}`**")
                if c.get("args"):
                    st.code(json.dumps(c["args"], indent=2, ensure_ascii=False), language="json")
                result = c.get("result", "")
                st.text(result[:2000] + ("… (truncated)" if len(result) > 2000 else ""))


def _run_conversation_turn(user_text: str):
    """
    Add user message, call the LLM in a tool-calling loop, update session state.
    Renders the assistant response (and any tool call activity) live in the UI.
    """
    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.last_retrieved_clause_ids = []

    # Build the full message list for the API call
    system_prompt = build_system_prompt(st.session_state.active_doc_name)
    api_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

    with st.chat_message("assistant"):
        # Show tool activity in a collapsible status box while the loop runs
        status = st.status("Thinking…", expanded=False)
        response_placeholder = st.empty()
        final_text = ""
        turn_tool_calls: list[dict] = []   # {name, args, result} — persisted for inspection
        reasoning_parts: list[str] = []     # model's thinking across rounds

        # Tool-calling loop (max 8 rounds to prevent runaway)
        for _ in range(8):
            response = client.chat.completions.create(
                model=llm_config.model,
                messages=api_messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=CHAT_MAX_TOKENS,
            )

            choice = response.choices[0]
            msg = choice.message
            round_reasoning, visible_content = _extract_reasoning(msg)
            if round_reasoning:
                reasoning_parts.append(round_reasoning)

            if choice.finish_reason == "tool_calls":
                # Add assistant tool-call message to history
                api_messages.append(msg.model_dump(exclude_unset=True))

                # Execute each tool call
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        fn_args = {"_raw": tc.function.arguments}

                    status.update(label=f"🔧 {fn_name}({', '.join(f'{k}={repr(v)[:40]}' for k, v in fn_args.items())})", expanded=True)

                    result = dispatch_tool(
                        fn_name, fn_args,
                        graph=graph,
                        llm_config=llm_config,
                        clause_map=clause_map,
                        clause_index=clause_index,
                        checkpoints_dir=st.session_state.checkpoints_dir,
                    )

                    status.write(f"**{fn_name}** → {result[:300]}{'…' if len(result) > 300 else ''}")
                    turn_tool_calls.append({"name": fn_name, "args": fn_args, "result": result})

                    api_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                # Final text response
                final_text = visible_content
                break

        # Empty answers happen: reasoning models sometimes return no content, or
        # the tool loop exhausts its rounds without prose. Force one more no-tools
        # completion explicitly asking for the answer before giving up.
        if not final_text.strip():
            try:
                forced = client.chat.completions.create(
                    model=llm_config.model,
                    messages=api_messages + [{
                        "role": "user",
                        "content": (
                            "Provide your final answer to the user now, in plain text, "
                            "based on everything gathered above. Do not call any tools."
                        ),
                    }],
                    temperature=0.2,
                    max_tokens=CHAT_MAX_TOKENS,
                )
                final_text = (forced.choices[0].message.content or "").strip()
            except Exception:
                logging.exception("Forced final-answer completion failed")

        # Never leave it empty — an empty assistant message is skipped by the
        # history renderer, which would drop this turn's Sources/test cases.
        if not final_text.strip():
            final_text = (
                "_(Done — see the results below.)_"
                if st.session_state.pending_results
                else "_(I couldn't generate a text answer — please rephrase your question or try again.)_"
            )

        status.update(label="Done", state="complete", expanded=False)
        response_placeholder.markdown(final_text)

        # Persist conversation (strip system prompt — it's prepended fresh each turn)
        # Append everything added to api_messages beyond the system prompt
        new_messages = api_messages[1 + len(st.session_state.messages):]  # what was added this turn (tool rounds)
        st.session_state.messages.extend(new_messages)
        final_idx = len(st.session_state.messages)  # index the final assistant message will occupy
        st.session_state.messages.append({"role": "assistant", "content": final_text})

        # Citations = clauses actually read/generated-from this turn, plus any the
        # assistant explicitly named in its answer (NOT raw search candidates).
        # Keyed by the message index so the history loop redraws them on every
        # rerun (otherwise they'd vanish when any later widget triggers a rerun).
        citations = _turn_citations(st.session_state.last_retrieved_clause_ids)
        if citations:
            st.session_state.turn_sources[final_idx] = citations

        # Persist the trace (reasoning + tool calls) so it stays inspectable in
        # history across reruns, not just in the ephemeral live status box.
        if turn_tool_calls:
            st.session_state.turn_tool_calls[final_idx] = turn_tool_calls
        reasoning_text = "\n\n---\n\n".join(reasoning_parts).strip()
        if reasoning_text:
            st.session_state.turn_reasoning[final_idx] = reasoning_text

        # Same persistence fix for generated test cases: move them out of the
        # transient pending_results into a per-message store.
        if st.session_state.pending_results:
            st.session_state.turn_test_cases[final_idx] = st.session_state.pending_results
            st.session_state.pending_results = []

        _render_trace(final_idx)
        if citations:
            _render_sources(citations, key_prefix=f"turn_{final_idx}")
        if final_idx in st.session_state.turn_test_cases:
            _render_test_cases(st.session_state.turn_test_cases[final_idx], key_prefix=f"turn_{final_idx}")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("Regulatory Test Case Assistant")
st.caption(
    f"Active document: **{st.session_state.active_doc_name}** — "
    "ask about its structure, explore clauses, or request test case generation."
)

import graph_viz as _gv  # noqa: E402

# ── Document structure (collapsible folder tree) ─────────────────────────────
with st.expander("🗂 Document structure — browse sections", expanded=False):
    st.caption(
        "Click a section to expand it; open as many as you like. Annex sections are "
        "namespaced (e.g. **Annex 3 / 5** is section 5 *within* Annex 3, distinct from body §5)."
    )
    if clause_index:
        _n_roots = sum(1 for _i in clause_index.values() if not _i.get("parent"))
        _tree_height = min(max(_n_roots * 32 + 30, 220), 560)
        components.html(_gv.build_tree_html(clause_index), height=_tree_height, scrolling=True)
    else:
        st.caption("No clauses loaded.")

# ── Graph explorer ────────────────────────────────────────────────────────
# Interactive view of the clause graph (CONTAINS hierarchy + REFERS_TO cross-
# references). The built HTML is cached in session state keyed by a signature of
# the controls + active document, so ordinary chat reruns don't rebuild it.
with st.expander("🕸 Graph explorer — clause relationships", expanded=False):
    from normalize import natkey

    _all_ids = sorted(clause_index.keys(), key=natkey)
    if not _all_ids:
        st.caption("No clauses loaded.")
    else:
        _c1, _c2, _c3 = st.columns([2, 1, 2])
        with _c1:
            _mode = st.radio(
                "View", ["Focused neighborhood", "Full graph"],
                key="graph_mode", horizontal=True,
            )
        with _c2:
            _hops = st.slider(
                "Hops", 1, 3, 1, key="graph_hops",
                disabled=(_mode == "Full graph"),
            )
        with _c3:
            _edge_types = st.multiselect(
                "Edge types", ["CONTAINS", "REFERS_TO"],
                default=["CONTAINS", "REFERS_TO"], key="graph_edges",
                help="CONTAINS = section hierarchy · REFERS_TO = cross-references",
            )
        _focus = None
        if _mode == "Focused neighborhood":
            _default = "5.2.1" if "5.2.1" in clause_index else _all_ids[0]
            _focus = st.selectbox(
                "Focus clause", _all_ids,
                index=_all_ids.index(_default),
                format_func=lambda c: f"{c} — {clause_index[c].get('title') or '(untitled)'}",
                key="graph_focus",
            )
        st.caption(
            "🔴 obligation · 🟠 test procedure · 🔵 test condition/setup · "
            "🟣 performance data · 🟢 definition/scope · ⚪ informative.  "
            "Solid grey = CONTAINS, dashed blue = REFERS_TO. Drag nodes; hover for titles."
        )

        _etypes = tuple(_edge_types) if _edge_types else ("CONTAINS", "REFERS_TO")
        _sig = (st.session_state.active_doc_name, _mode, _focus, _hops, tuple(sorted(_etypes)))
        if st.session_state.get("_graph_sig") != _sig:
            try:
                _edges = _gv.edges_from_index(clause_index)
                if _mode == "Full graph":
                    _nodes, _sub = _gv.filter_edges(_edges, _etypes)
                else:
                    _nodes, _sub = _gv.neighborhood(_edges, _focus, hops=_hops, edge_types=_etypes)
                st.session_state["_graph_html"] = _gv.build_network_html(
                    _nodes, _sub, clause_index, focus=_focus,
                )
                st.session_state["_graph_meta"] = (len(_nodes), len(_sub))
                st.session_state["_graph_sig"] = _sig
                st.session_state["_graph_err"] = None
            except Exception as exc:
                logging.exception("Graph render failed")
                st.session_state["_graph_err"] = str(exc)

        if st.session_state.get("_graph_err"):
            st.error(f"Couldn't render graph: {st.session_state['_graph_err']}")
        elif st.session_state.get("_graph_html"):
            _n, _e = st.session_state.get("_graph_meta", (0, 0))
            st.caption(f"Showing {_n} clause(s), {_e} relationship(s).")
            components.html(st.session_state["_graph_html"], height=640, scrolling=True)

# ── Extraction & categorization overlay (PDF debug) ──────────────────────────
# Highlights every located clause on the original PDF, colour-coded by category,
# so you can see WHAT was extracted and HOW it was classified. Rendered per page
# and cached by (document, page, category filter).
_dbg_src = st.session_state.get("source_info") or {}
_dbg_pdf = _dbg_src.get("pdf_path")
_dbg_pages = _dbg_src.get("clause_pages") or {}
if _dbg_pdf and _dbg_pages:
    with st.expander("🎨 Extraction & categorization — PDF overlay", expanded=False):
        import pdf_highlight as _ph

        _dbg_mid = _dbg_src.get("middle_json_path")
        _total_pages = _ph.pdf_page_count(_dbg_pdf)
        st.session_state.setdefault("pdf_dbg_page", 1)

        st.caption(
            "What the parser extracted, highlighted on the original PDF and colour-coded by "
            "category. Grey = 'unknown' category; yellow = table region. Text the parser "
            "extracted but couldn't tie to a classified clause is left unhighlighted."
        )
        # clause_page_map is keyed by ORIGINAL (bare) clause ids, so map categories that way.
        _cat_by_orig: dict[str, str] = {}
        for _cid, _info in clause_index.items():
            _cat_by_orig[_cid.split(" / ")[-1]] = _info.get("category", "unknown")

        # Full overlay (every extracted block) when the middle.json is available;
        # otherwise fall back to clause-matched regions only.
        _rgb = {c: _ph.hex_to_rgb01(h) for c, h in _gv.CATEGORY_COLOR.items()}
        if _dbg_mid and Path(_dbg_mid).exists():
            _boxes = _ph.build_extraction_overlay(_dbg_pages, _cat_by_orig, Path(_dbg_mid))
        else:
            _boxes = _ph.category_boxes_by_page(_dbg_pages, _cat_by_orig)
        _present = sorted({cat for boxes in _boxes.values() for _b, cat in boxes}) or ["unknown"]

        _LEGEND_HEX = {**_gv.CATEGORY_COLOR, "table": "#ffe119"}
        _swatches = "".join(
            f'<span style="display:inline-flex;align-items:center;margin:0 12px 4px 0;white-space:nowrap;">'
            f'<span style="width:12px;height:12px;border-radius:3px;margin-right:5px;border:1px solid rgba(120,120,140,.4);'
            f'background:{_LEGEND_HEX.get(c, "#cccccc")};"></span>{c}</span>'
            for c in _present
        )
        st.markdown(f'<div style="font-size:13px;line-height:1.9;">{_swatches}</div>',
                    unsafe_allow_html=True)

        _ctop1, _ctop2 = st.columns(2)
        with _ctop1:
            _sel_cats = st.multiselect("Show categories", _present, default=_present, key="pdf_dbg_cats")
        with _ctop2:
            _disp_w = st.slider("Display size (px)", 320, 1200, 640, 40, key="pdf_dbg_width",
                                help="Shrink or enlarge the rendered page.")

        _rgb_full = {**_rgb, "table": _ph.hex_to_rgb01("#ffe119")}
        _page = min(int(st.session_state["pdf_dbg_page"]), _total_pages)
        _sig = (st.session_state.active_doc_name, _page, tuple(sorted(_sel_cats)))
        if st.session_state.get("_pdf_dbg_sig") != _sig:
            try:
                _filtered = {
                    p: [(b, c) for (b, c) in boxes if c in set(_sel_cats)]
                    for p, boxes in _boxes.items()
                }
                _imgs = _ph.render_boxes_page_images(
                    Path(_dbg_pdf), _filtered, page_indices=[_page - 1], zoom=2.0,
                    category_rgb=_rgb_full,
                )
                st.session_state["_pdf_dbg_img"] = _imgs[0][1] if _imgs else None
                st.session_state["_pdf_dbg_sig"] = _sig
                st.session_state["_pdf_dbg_err"] = None
            except Exception as _exc:
                logging.exception("PDF debug render failed")
                st.session_state["_pdf_dbg_err"] = str(_exc)

        if st.session_state.get("_pdf_dbg_err"):
            st.error(f"Couldn't render page: {st.session_state['_pdf_dbg_err']}")
        elif st.session_state.get("_pdf_dbg_img"):
            _il, _ic, _ir = st.columns([1, 6, 1])
            with _ic:
                st.image(st.session_state["_pdf_dbg_img"], width=_disp_w)
        else:
            st.caption("Nothing to show for this page.")

        # ── Bottom navigation ────────────────────────────────────────────────
        def _dbg_prev():
            st.session_state.pdf_dbg_page = max(1, int(st.session_state.get("pdf_dbg_page", 1)) - 1)

        def _dbg_next(_n=_total_pages):
            st.session_state.pdf_dbg_page = min(_n, int(st.session_state.get("pdf_dbg_page", 1)) + 1)

        _nb1, _nb2, _nb3 = st.columns([1, 2, 1])
        with _nb1:
            st.button("◀ Previous", key="pdf_dbg_prev", on_click=_dbg_prev,
                      disabled=_page <= 1, use_container_width=True)
        with _nb2:
            st.markdown(
                f"<div style='text-align:center;padding-top:6px;color:#808495;'>"
                f"Page {_page} of {_total_pages} · {len(_dbg_pages)} clauses located</div>",
                unsafe_allow_html=True,
            )
        with _nb3:
            st.button("Next ▶", key="pdf_dbg_next", on_click=_dbg_next,
                      disabled=_page >= _total_pages, use_container_width=True)

# Render existing chat history
for idx, msg in enumerate(st.session_state.messages):
    role = msg.get("role")
    if role not in ("user", "assistant"):
        continue  # skip tool call / tool result messages
    content = msg.get("content") or ""
    has_panels = role == "assistant" and (
        idx in st.session_state.turn_sources
        or idx in st.session_state.turn_test_cases
        or idx in st.session_state.turn_tool_calls
        or idx in st.session_state.turn_reasoning
    )
    # Skip only tool-call assistant stubs (no text AND no attached panels).
    if not content and not has_panels:
        continue
    with st.chat_message(role):
        if content:
            st.markdown(content)
        if role == "assistant":
            _render_trace(idx)
        if role == "assistant" and idx in st.session_state.turn_sources:
            _render_sources(st.session_state.turn_sources[idx], key_prefix=f"turn_{idx}")
        if role == "assistant" and idx in st.session_state.turn_test_cases:
            _render_test_cases(st.session_state.turn_test_cases[idx], key_prefix=f"turn_{idx}")

# Chat input
if prompt := st.chat_input("Ask anything about UN R152 or request test cases…"):
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    _run_conversation_turn(prompt)
