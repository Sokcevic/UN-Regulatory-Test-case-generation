# Regulatory Test Case Generator — Chat UI

Interactive Streamlit app for generating UN R152 test cases via natural-language queries,
with support for uploading your own regulatory document.

## How it works

The assistant drives the conversation via LLM function calling — no fixed state machine.
It searches clauses, reads specific clauses, inspects performance tables, and runs the
ReAct test-case generator, deciding when to use each tool based on what you ask.

You can also **upload your own document** (PDF, DOCX, PPTX, XLSX, image, or Markdown) from
the sidebar — it fully replaces the active document for the rest of the session. Non-Markdown
uploads are converted via MinerU, then parsed/classified/graphed exactly like the built-in
R152 document. Uploads are content-hash-keyed under `chat_ui/uploads/<sha256>/`, and the
two slow stages — MinerU conversion and LLM clause classification — are **cached** there, so
re-uploading the same file (even after a restart) skips straight to the graph.

**Model / server selector.** The sidebar has a dropdown to pick the LLM backend.
**`OpenAI · gpt-4.1-mini`** ships as the default preset, plus a **"Custom…"** option for any
other OpenAI-compatible endpoint (e.g. a self-hosted vLLM server). The choice drives every LLM
call (chat, retrieval, classification, generation).

**API keys for hosted providers.** Backends that need a key (the OpenAI preset, and any custom
provider) show a password **API key** field in the sidebar. It's prefilled from `$OPENAI_API_KEY`
if that env var is set, kept in Streamlit session state only (never written to disk), and sent as
the Bearer token. Leave it blank when pointing "Custom…" at a keyless vLLM server. To add another
hosted model, either pick "Custom…" (enter base URL + model ID + key) or add an entry to
`MODEL_PRESETS` in `app.py`.

> **Note on OpenAI reasoning models.** The GPT-5 / o-series reasoning models reject the
> `temperature` and `max_tokens` parameters this app sends on every call (chat, retrieval,
> classify, and the ReAct agent), so they will 400. `gpt-4.1-mini` is the default OpenAI preset
> precisely because it accepts the standard parameters and is inexpensive. Selecting a GPT-5-class
> model would require making `temperature`/`max_tokens` conditional across the pipeline first.

**Hierarchy-aware classification.** Clauses are classified with document context rather than in
isolation (see `regulatory_testgen/classify.py`): a compact global outline tags each top-level
container's role, then each section-subtree is classified in one call so a heading "sees" the
obligations it groups (bottom-up) and clauses inside a *sample* annex are demoted (top-down).
Each clause gets a fixed functional **category** plus a **normative force** (`binding` / `example`
/ `none`); only `binding` testable clauses are generated from, so worked examples and model forms
in annexes no longer produce spurious test cases. Results are keyed by clause `uid` (annexes reuse
ids). Document-specific structure (scenarios, topics) is carried by the hierarchy, not by
per-document categories.

**Grounded generation.** When you ask for test cases on a specific scenario (e.g. 5.2.1), the
generator automatically feeds the agent the clause's **parent section** (5.2, 5 — the general
requirements) and its **transitively referenced** clauses, not just the sub-clause in
isolation. See `RegulatoryGraph.get_generation_context`.

**Document repair (Graph-RAG quality).** Real UN regulations reuse the body's clause numbers
in their *Communication form* and *annexes*, and each annex restarts numbering at 1 — so Annex 3's
"§5 Reporting by Technical Service" collides with the body's "§5 Specifications" under the same
clause id, and many section/scenario headings live in the `text` field with an empty `title`.
`normalize.py` fixes all of this at load time (no pipeline re-run, works for uploads too):

- **Annex namespacing** — using document order (`line_start`) and the `region='annex'` heading
  clauses, annex clauses are re-keyed under their container: the body keeps `5` = "Specifications",
  while Annex 3's becomes `Annex 3 / 5` = "Reporting by Technical Service", nested under `Annex 3`.
  Parentage is fully derivable from the id string (`normalize.parent_of`), which drives the
  hierarchy tools and the folder-tree UI.
- **Duplicate resolution + form-field pruning** — keeps the substantive body copy of a colliding
  id and drops Communication-form boilerplate ("Brief description of vehicle:").
- **Title recovery** — rebuilds titles from `text` / children, so Section 5 → "Specifications",
  5.2.1 → "Car to car scenario", 5.2.2 → "Car to pedestrian scenario", 5.2.3 → "Car to bicycle".

**Document structure browser.** A "🗂 Document structure" panel renders a collapsible folder tree
(native `<details>`/`<summary>`, no JS) so you can expand/collapse sections — annex sections appear
nested under their annex (e.g. **Annex 3 ▸ Annex 3 / 5**).

**Exploration tools.** `get_document_structure` returns the outline (top sections + children);
call it with `section="<id>"` to expand one section's full subtree to every depth, so the model
can see nested scenarios. `get_clause` returns a clause's **full** text plus its parent, direct
sub-clauses, and cross-references (not a 400-char stub). `search_clauses` shows each hit's
section breadcrumb. The system prompt instructs the model to explore *outline → drill into the
relevant section → read specific clauses → answer and cite* before answering.

**Graph explorer.** A "🕸 Graph explorer" panel (top of the main page) renders the clause graph
interactively (pyvis/vis.js), for debugging and seeing which clauses correlate. Nodes are clauses
(coloured by category), edges are **CONTAINS** (section hierarchy, solid grey) and **REFERS_TO**
(cross-references, dashed blue). Default is a *focused neighborhood*: pick a clause and see its
parents/children and cross-references out to N hops; a *full graph* toggle and edge-type filter
are also available. The graph is the in-memory `RegulatoryGraph` (see `RegulatoryGraph.all_edges`)
— no external database. The rendered HTML is cached per (document, view, focus, hops, edge-types)
so ordinary chat interactions don't rebuild it. `pyvis` is bundled via `chat_ui/requirements.txt`.

**Extraction & categorization overlay.** For PDF-sourced documents, a "🎨 Extraction &
categorization" panel renders the original pages with **every extracted block highlighted and
colour-coded**: clauses in their category colour (obligation, test procedure, test condition,
performance data, …), table regions in tan, and text MinerU extracted but didn't tie to a
classified clause in grey. It answers "what did the parser extract, and how was each part
classified?" — filter by category (legend + multiselect), resize the page (display-size slider),
and page through with the bottom **◀ Previous / Next ▶** controls. Rendering is non-destructive
(the source PDF is untouched, highlights drawn on an in-memory copy) and cached per page.

**Trace inspection.** Every assistant turn keeps a persistent, collapsible **🔧 Tool calls**
panel (each call's name, arguments, and full result) and a **🧠 Model reasoning** panel (the
model's thinking, captured from `reasoning_content` or `<think>` blocks) — so you can see exactly
which sections were retrieved and where the model's logic went, even in scrolled-back history.

Every answer shows a **Sources** panel listing which clauses it used, with page numbers and
(for PDF-sourced documents) a highlighted view of the original pages. Citations are precise:
they come from the clauses the assistant actually **reads** (`get_clause` /
`get_performance_table`), the real `source_clause_ids` of any **generated** test cases, and the
clause IDs the assistant **names in its answer** — *not* every candidate a search happened to
surface. The highlighted pages are rendered as images (highlights baked in) so they always
display in the browser; the original PDF is never modified, and an annotated copy is
downloadable.

## Setup

```bash
# From the thesis-code root — activate the pipeline venv
source regulatory_testgen/venv/bin/activate
# streamlit and pandas are already installed there; nothing extra needed
```

The pipeline venv in `regulatory_testgen/venv` already has all required packages
(dynamiq, openai, streamlit, pandas). Activate it with:
```bash
source regulatory_testgen/venv/bin/activate
```

## Run

```bash
# From thesis-code root (important — relative imports depend on this)
streamlit run chat_ui/app.py
```

Then open http://localhost:8501

## Example queries

- `emergency braking test cases for unladen vehicles`
- `AEBS collision warning activation distance`
- `stationary car target approach at 60 km/h`
- `test procedure for forward vehicles`
- `performance limits annex 3`

## Row filter examples

When a clause has a performance table, you can constrain which rows to generate for:

- `Only rows where vehicle condition is 'unladen' and speed is 60 km/h`
- `Only the first three rows`
- `Rows for stationary target at any speed`

Leave blank to generate test cases for every row.

## Document upload & MinerU

Non-Markdown uploads are converted to Markdown by MinerU, which runs as a subprocess from
its own venv (`chat_ui/venv_mineru`, Python 3.12 — MinerU needs 3.10–3.13, which may not
match the main pipeline venv). Set it up once:

```bash
python3.12 -m venv chat_ui/venv_mineru
chat_ui/venv_mineru/bin/pip install "mineru[core]"
chat_ui/venv_mineru/bin/pip install "pdftext==0.6.3"   # newer pdftext breaks mineru 3.4.0's txt-extraction path
```

By default MinerU runs locally on CPU (`pipeline` backend), which is fine for a handful of
pages but slow on larger documents.

### Offloading conversion to a remote GPU

MinerU has a built-in client/server split: the CLI can act as a thin client against a
`mineru-api` FastAPI service running elsewhere, so all model inference (layout detection,
OCR, or the VLM backend) happens on that remote machine instead of the local CPU.

**On the GPU server** (needs Python 3.10–3.13, a CUDA-capable GPU, and network access from
this machine):

```bash
python3.12 -m venv mineru_gpu_venv
mineru_gpu_venv/bin/pip install "mineru[core]"
mineru_gpu_venv/bin/pip install "pdftext==0.6.3"

# Bind to 0.0.0.0 so this machine can reach it; --allow-public-http-client is required
# for a non-localhost client to use *-http-client backends / a remote --api-url.
mineru_gpu_venv/bin/mineru-api --host 0.0.0.0 --port 8008 --allow-public-http-client
```

**On this machine**, point the app at it and pick a GPU-accelerated backend:

```bash
export MINERU_API_URL="http://<gpu-server-host>:8008"
export MINERU_BACKEND="hybrid-engine"   # or "vlm-engine" — needs ~6GB+ VRAM on the server;
                                         # leave unset ("pipeline") if the server has no GPU either
streamlit run chat_ui/app.py
```

With `MINERU_API_URL` set, the local `mineru` binary only uploads the file and downloads the
result — no local model weights or GPU are needed on this machine at all. Firewall/VPN the
port appropriately; `mineru-api` has no built-in auth.

## Architecture

```
chat_ui/
├── app.py             # Streamlit UI — conversational, tool-calling loop
├── ingest.py           # Document upload pipeline: MinerU → parse → classify → graph
├── pdf_highlight.py     # Correlates clauses to PDF page/bbox for the Sources panel
├── retrieval.py        # LLM-based query → clause ID mapping
├── generator.py        # Default (R152) graph loader + ReAct agent wrapper
├── tools.py             # Tool definitions/handlers the LLM calls
└── requirements.txt
```

No existing pipeline files are modified. The chat UI is a thin layer on top of the existing
`regulatory_testgen` pipeline code.
