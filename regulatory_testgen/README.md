# regulatory_testgen

Agentic LLM system for generating UN-compliant test cases from regulatory PDFs, using
Dynamiq (agent orchestration) and LlamaIndex (Graph-RAG).

## Architecture

```
MinerU PDF → Markdown
      ↓
Stage 1: Parse (parsing/markdown_parser.py)
      ↓  247 Clause objects
Stage 2: Tables (extraction/tables.py)
      ↓  14 RegulationTable objects
Stage 3: Build Knowledge Graph (LlamaIndex PropertyGraphStore)
      ↓  196 CLAUSE nodes, 83 REFERS_TO edges, 155 CONTAINS edges
Stage 4: Generate test cases (Dynamiq ReAct agent × 84 testable clauses)
      ↓
03_test_cases.json  +  03_test_cases.csv
```

### Key design decisions

| Decision | Rationale |
|---|---|
| LlamaIndex PropertyGraphStore | Typed REFERS_TO / CONTAINS edge labels; built-in `get_triplets` for BFS |
| BFS transitive closure | Guarantees complete cross-reference context (BifrostRAG +12.3 F1) |
| Dynamiq ReAct Agent | Interleaved reasoning + tool calls; agent can follow unexpected refs |
| Template + instantiation | Table-driven clauses: one LLM template × N rows = N identical-structure tests |

## Setup

```bash
# Python 3.12 required (Dynamiq caps at <3.14)
python3.12 -m venv venv
venv/bin/pip install -r requirements.txt
```

## Usage

```bash
# Full pipeline
venv/bin/python -m regulatory_testgen run \
    /path/to/R152r2E.md \
    --output-dir output

# Point at a different OpenAI-compatible endpoint (hosted or self-hosted vLLM)
venv/bin/python -m regulatory_testgen run \
    /path/to/R152r2E.md \
    --base-url http://your-vllm-host:8000/v1 \
    --model your-model-id

# Inspect graph stats after running
venv/bin/python -m regulatory_testgen graph-stats --output-dir output
```

## Output

| File | Contents |
|---|---|
| `01_clauses.json` | Parsed clause objects (247 for R152) |
| `01b_classifications.json` | LLM clause category classifications |
| `02_graph_summary.json` | Graph statistics + clause index |
| `03_test_cases.json` | Full TestCase objects with traceability |
| `03_test_cases.csv` | 3-column CSV: `Unique_ID, Test_title, test_scenario` |

`output/` ships pre-populated with the R152 checkpoints so `chat_ui/` works without
re-running the pipeline.

## Package structure

```
regulatory_testgen/
├── __init__.py           version info
├── config.py             LLMConfig, PipelineConfig
├── models.py             TestCase, ClauseRole
├── graph.py              RegulatoryGraph — LlamaIndex PropertyGraphStore + BFS
├── prompts.py            LLM prompt templates + few-shot examples
├── agent.py              Dynamiq tools (GetClause, GetReferencedClauses, GetTables) + agent factory
├── classify.py           LLM-based clause classification
├── pipeline.py           4-stage orchestration with JSON checkpoints
├── export.py             JSON + CSV writers
├── cli.py                argparse CLI (`python -m regulatory_testgen`)
├── parsing/              Markdown → Clause objects (markdown_parser.py, tree_builder.py)
├── extraction/           Clause text → tables/references/requirements
├── data_models/          Clause, DocumentTree, RegulationTable, and related dataclasses
└── output/               Pre-computed checkpoints for the bundled R152 regulation
```
