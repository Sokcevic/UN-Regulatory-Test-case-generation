"""
regulatory_testgen — Agentic test case generation from UN regulatory PDFs.

Architecture:
  1. Parsing    — MinerU Markdown → structured Clause objects (parsing/, extraction/, data_models/)
  2. Graph      — LlamaIndex PropertyGraphStore; REFERS_TO edges + BFS transitive closure
  3. Agent      — Dynamiq ReAct agent with regulatory tools
  4. Pipeline   — 4-stage orchestration with JSON checkpoints
  5. Export     — JSON (full) + CSV (3-column, matches ground truth format)

Python ≥ 3.12 required (Dynamiq constraint).
"""

__version__ = "1.0.0"
