# Agentic LLM Test Case Generation for Regulatory Documents

Bachelor's thesis project: generating UN-regulation-compliant test cases (starting with
UN R152, AEBS) from regulatory PDFs using an agentic LLM pipeline (Dynamiq + LlamaIndex
Graph-RAG), exposed through a conversational chat UI.

## Quickstart (Docker)

```bash
docker compose up --build
```

Then open http://localhost:8501. The container ships with the UN R152 regulation
already parsed (`data/R152r2E/`), so you can start asking questions immediately — no
setup required.

**Picking an LLM backend.** The sidebar's model dropdown offers:
- `OpenAI · gpt-4.1-mini` (default) — needs an API key. Either export it before starting
  the container (`OPENAI_API_KEY=sk-... docker compose up --build`) so it prefills the
  sidebar field, or paste it directly into the sidebar once the app is running.
- `Custom…` — point at any other OpenAI-compatible endpoint, including a self-hosted vLLM server.

## What's in here

| Path | What it is |
|---|---|
| `chat_ui/` | Streamlit chat interface — the thing `docker compose up` runs. See `chat_ui/README.md`. |
| `regulatory_testgen/` | Pipeline: parse → classify → build graph → generate test cases (Dynamiq + LlamaIndex). See `regulatory_testgen/README.md`. |
| `data/R152r2E/` | Pre-parsed UN R152 regulation (MinerU output) bundled so the container works out of the box. |

## Running without Docker

Each sub-project's README has native setup instructions (venvs, Python version
constraints, etc.) if you'd rather not use Docker — see `chat_ui/README.md` and
`regulatory_testgen/README.md`.
