# Chat UI — manual UI / design review

This is the qualitative counterpart to `test_chat_ui.py`. Automated tests can't
judge layout, wording, or flow, so this file records a structured walkthrough of
the interface (`chat_ui/app.py`) — what works, what's rough, and what was fixed.
Re-run this walkthrough by hand after UI changes.

## How to review

```bash
source regulatory_testgen/venv/bin/activate
streamlit run chat_ui/app.py
# open http://localhost:8501
```

## Layout inventory

| Region | Contents | Verdict |
|--------|----------|---------|
| Sidebar | Title, active-doc caption, uploader + MinerU mode note, source-PDF status, graph metrics, example questions, clear button, model name | Clear hierarchy; the divided sections read well. |
| Header | Title + active-doc caption | Fine, slightly redundant with the sidebar caption. |
| Chat history | User/assistant bubbles; assistant bubbles may carry a Sources expander and/or test-case cards | Good; the persistence fix keeps panels from vanishing on rerun. |
| Live turn | Status box (tool activity) → answer → Sources → test cases | Good; the status box collapses to "Done". |
| Sources expander | Clause list with pages + "View highlighted pages" button → modal | Fixed this round (see below). |
| Test-case cards | Expandable per case; each has its own Sources; JSON/CSV download | Good. |

## Findings this round

### 1. Highlighted pages were oversized and appeared only after a blank gap — FIXED
- **Was:** clicking "View highlighted in source PDF" rendered full-width images
  (`use_container_width=True` at zoom 2.0) inline in the expander, on the main
  wide canvas, with no feedback while rasterising — so at first "nothing
  appears", then a wall of huge images.
- **Now:** a proper `st.dialog` modal (`width="large"`) with a `st.spinner`
  ("Rendering highlighted pages…") while it rasterises. Images render at zoom
  1.5 inside the bounded modal, so they're readable but not overwhelming. The
  render is cached per panel, so clicking the in-modal download button doesn't
  re-rasterise. Closing the modal returns to the conversation unchanged.

### 2. Citations included clauses the answer didn't actually use — FIXED
- **Was:** two independent over-citation sources. (a) `search_clauses` pushed all
  ~12 candidates into the citation list. (b) The answer text was mined for clause
  IDs, which grabbed *ranges/examples* — e.g. "the procedures 6.4-6.9" cited both
  6.4 and 6.9 even though the answer was a structural overview using none of them.
- **Now:** citations come only from clauses the model deliberately **reads**
  (`get_clause` / `get_performance_table`) or **generates** test cases from. A
  `get_document_structure` overview correctly shows **no** Sources panel. Covered
  by `TestCitationRecording`.

### 3. Empty final answer dropped a turn's results — FIXED
- If the tool loop ended without prose (round cap / tool-only finish), the empty
  assistant message was skipped by the history renderer, taking its Sources and
  test cases with it. Now empty text gets a placeholder and the renderer draws
  the panels regardless.

## Still-open / lower-priority observations (not blocking)

- **Header vs sidebar caption** both state the active document. Minor redundancy;
  harmless.
- **"Testable clauses" metric** counts the whole index each rerun — negligible,
  but if the index grows large it's O(n) per rerun.
- **Uploaded documents don't persist generated test cases to disk**, so
  `list_existing_test_cases` finds nothing for them (it reads
  `03_test_cases.json`, which only the default doc has). Acceptable for a session
  tool; note it if persistence is ever wanted.
- **Clause page mapping covers ~138/202 clauses** on the default R152 doc; an
  un-located citation shows "page not located" and is excluded from the modal.
  This is the known limitation of the text-correlation heuristic, surfaced
  honestly rather than silently.
- **Long tool-result strings** are truncated to 300 chars in the status box —
  good for noise, but the full result still goes to the model.

## Accessibility / polish quick-checks

- Emoji are decorative and paired with text labels — screen-reader safe.
- Color: highlight is amber `(1, 0.85, 0)`; the Sources list also gives page
  numbers textually, so the highlight color isn't the only signal.
- Download buttons have unique `key=`s per turn/case — no Streamlit key
  collisions across persisted turns (the earlier cause of duplicate-widget
  errors).
