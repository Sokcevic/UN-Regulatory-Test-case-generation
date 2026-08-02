"""Dynamiq-based ReAct agent for regulatory clause analysis.

The agent is given three tools that operate on the RegulatoryGraph:

  get_clause             — fetch one clause by ID
  get_referenced_clauses — BFS transitive closure from a clause (Graph-RAG)
  get_tables             — fetch performance tables owned by a clause

Tools are created as closures via the @function_tool decorator so that they
capture the RegulatoryGraph without needing PrivateAttr or manual input_schema
wiring. The decorator also auto-generates the input_schema Pydantic model from
the function's type annotations, which Dynamiq's schema_generator needs.

Design based on:
  ReAct (Yao et al., 2022)     — interleaved reasoning + acting
  AgenticIE (Colakoglu, 2024)  — planner-executor pattern for regulatory docs
"""

from __future__ import annotations

import json
import logging
from typing import Any

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents.agent import Agent
from dynamiq.nodes.agents.base import AgentInputSchema
from dynamiq.nodes.llms.openai import OpenAI as DynamiqOpenAI
from dynamiq.nodes.tools.function_tool import FunctionTool, function_tool
from dynamiq.runnables import RunnableConfig

from regulatory_testgen.config import LLMConfig
from regulatory_testgen.graph import RegulatoryGraph
from regulatory_testgen.models import TestCase
from regulatory_testgen.prompts import AGENT_SYSTEM_PROMPT, build_clause_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


def _make_tools(graph: RegulatoryGraph) -> list[FunctionTool]:
    """Create the three regulatory tools, each closing over the given graph.

    The @function_tool decorator auto-generates the input_schema Pydantic model
    from the function's type annotations (required by Dynamiq's schema_generator).
    The returned class must be instantiated before being passed to the Agent.
    """

    @function_tool
    def get_clause(clause_id: str, **kwargs: Any) -> str:
        """Fetch the full text and title of a regulatory clause by its ID (e.g. '5.2.1').
        Use this when you need the content of a specific clause."""
        clause = graph.get_clause(clause_id)
        if clause is None:
            return f"Clause '{clause_id}' not found in this regulation."
        return f"[{clause.clause_id}] {clause.title}\n\n{clause.text}"

    @function_tool
    def get_referenced_clauses(clause_id: str, **kwargs: Any) -> str:
        """Retrieve ALL clauses transitively referenced by the given clause via BFS over
        REFERS_TO edges. Always call this before generating test cases to guarantee
        complete regulatory context."""
        clauses = graph.get_transitive_context(clause_id)
        if not clauses:
            return f"No clauses found for '{clause_id}'."
        parts = [f"### [{c.clause_id}] {c.title}\n{c.text}" for c in clauses]
        header = f"Transitive context for '{clause_id}' ({len(clauses)} clause(s)):\n"
        return header + "\n\n".join(parts)

    @function_tool
    def get_tables(clause_id: str, **kwargs: Any) -> str:
        """Retrieve all performance tables owned by the given clause.
        If tables are found, generate exactly one test case per table row — all
        test cases must be structurally identical, differing only in parameter values."""
        tables = graph.get_tables(clause_id)
        if not tables:
            return f"No performance tables found for clause '{clause_id}'."
        parts = []
        for table in tables:
            lines = [f"Table: {table.title or table.table_id}  (type: {table.table_type})"]
            if table.headers:
                lines.append("Headers: " + " | ".join(table.headers))
            for i, row in enumerate(table.rows, 1):
                row_str = " | ".join(f"{k}={v}" for k, v in row.items() if v)
                lines.append(f"Row {i}: {row_str}")
            parts.append("\n".join(lines))
        return f"{len(tables)} table(s) for '{clause_id}':\n\n" + "\n\n".join(parts)

    # @function_tool returns a class — instantiate each tool before returning
    return [get_clause(), get_referenced_clauses(), get_tables()]


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def build_agent(graph: RegulatoryGraph, llm_config: LLMConfig) -> Agent:
    """Create a Dynamiq ReAct agent wired to the regulatory knowledge graph."""
    connection = OpenAIConnection(
        url=llm_config.base_url,
        api_key=llm_config.api_key,
    )
    # litellm requires the "openai/" prefix for custom OpenAI-compatible endpoints
    model_id = llm_config.model
    if not model_id.startswith("openai/"):
        model_id = f"openai/{model_id}"

    llm = DynamiqOpenAI(
        connection=connection,
        model=model_id,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
    )
    return Agent(
        llm=llm,
        tools=_make_tools(graph),
        role=AGENT_SYSTEM_PROMPT,
        max_loops=llm_config.max_loops,
    )


# ---------------------------------------------------------------------------
# Running the agent
# ---------------------------------------------------------------------------


def analyse_clause(
    agent: Agent,
    clause_id: str,
    clause_title: str,
    clause_text: str,
    direct: bool = False,
) -> list[TestCase]:
    """Run the agent on one clause and parse the returned test cases.

    Args:
        direct: If True, use a simplified prompt that instructs the agent to
                generate directly from the provided text without tool calls.
                Used as a fallback when the full ReAct loop times out.

    Returns an empty list if the clause is not testable or if the agent fails.
    """
    prompt = build_clause_prompt(clause_id, clause_title, clause_text, direct=direct)
    try:
        # Agent.execute() requires AgentInputSchema, not a raw dict.
        # The validator reads context.context.get("role"), so we supply
        # the context dict via model_validate rather than the bare constructor.
        agent_input = AgentInputSchema.model_validate(
            {"input": prompt}, context={"role": ""}
        )
        result = agent.execute(input_data=agent_input, config=RunnableConfig())
    except Exception as exc:
        logger.warning("Agent execution failed for clause %s: %s", clause_id, exc)
        return []

    # execute() returns {"content": <final_answer_str>} on success
    if not isinstance(result, dict) or "content" not in result:
        logger.warning(
            "Agent returned unexpected output for clause %s: %r", clause_id, result
        )
        return []

    raw_output: str = result["content"]
    return _parse_agent_output(raw_output, clause_id)


def _parse_agent_output(raw: str, clause_id: str) -> list[TestCase]:
    """Extract test cases from the agent's JSON output.

    The agent is instructed to return a single JSON object. We strip common
    markdown code fences and parse, then convert to TestCase objects.
    """
    text = raw.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:] if lines[0].startswith("```") else lines)
        if text.endswith("```"):
            text = text[: text.rfind("```")]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Last resort: extract the first JSON object from free-form text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                logger.warning("Could not parse JSON for clause %s", clause_id)
                return []
        else:
            logger.warning("No JSON in agent output for clause %s", clause_id)
            return []

    reasoning = data.get("reasoning", "")
    raw_cases = data.get("test_cases", [])
    if not isinstance(raw_cases, list):
        return []

    test_cases: list[TestCase] = []
    for tc in raw_cases:
        if not isinstance(tc, dict):
            continue
        test_cases.append(
            TestCase(
                test_id=tc.get("test_id", f"TC-{clause_id}-{len(test_cases) + 1}"),
                title=tc.get("title", ""),
                scenario=tc.get("scenario", ""),
                preconditions=tc.get("preconditions", []),
                test_steps=tc.get("test_steps", []),
                expected_behavior=tc.get("expected_behavior", []),
                source_clause_ids=tc.get("source_clause_ids", [clause_id]),
                parameters=tc.get("parameters", {}),
                reasoning_trace=reasoning,
            )
        )

    return test_cases


# ---------------------------------------------------------------------------
# Unused but re-exported for symmetry with earlier imports
# ---------------------------------------------------------------------------

__all__ = ["build_agent", "analyse_clause", "AgentInputSchema"]
