"""Export helpers — JSON (full detail), CSV (3-column ground-truth format), Markdown.

The CSV format mirrors extracted_scenarios_number.csv:
  Unique_ID, Test_title, test_scenario

The Markdown format is a human-readable report for engineers, one section per test case.
It is also embedded in the JSON output under the `markdown` field of each TestCase.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from regulatory_testgen.models import TestCase


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(tc: TestCase) -> str:
    """Render one TestCase as a Markdown section."""
    lines: list[str] = []

    lines.append(f"## {tc.test_id}")
    lines.append("")
    lines.append(f"**Title:** {tc.title}  ")
    lines.append(f"**Source clauses:** {', '.join(tc.source_clause_ids)}  ")
    if tc.parameters:
        params_str = ", ".join(f"{k}: {v}" for k, v in tc.parameters.items())
        lines.append(f"**Parameters:** {params_str}  ")
    lines.append("")

    lines.append("### Overview")
    lines.append("")
    lines.append(tc.scenario)
    lines.append("")

    if tc.preconditions:
        lines.append("### Preconditions")
        lines.append("")
        for p in tc.preconditions:
            lines.append(f"- {p}")
        lines.append("")

    if tc.test_steps:
        lines.append("### Test Steps")
        lines.append("")
        for step in tc.test_steps:
            # Steps may already be prefixed with "1." etc — keep them as-is
            lines.append(f"{step}")
        lines.append("")

    if tc.expected_behavior:
        lines.append("### Expected Result")
        lines.append("")
        for e in tc.expected_behavior:
            lines.append(f"- {e}")
        lines.append("")

    if tc.parameters:
        lines.append("### Parameters")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for k, v in tc.parameters.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    return "\n".join(lines)


def attach_markdown(test_cases: list[TestCase]) -> list[TestCase]:
    """Render and attach markdown to each TestCase in-place. Returns the list."""
    for tc in test_cases:
        tc.markdown = render_markdown(tc)
    return test_cases


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def save_json(test_cases: list[TestCase], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    attach_markdown(test_cases)
    data = [tc.model_dump() for tc in test_cases]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(test_cases: list[TestCase], path: str | Path) -> None:
    """Write the 3-column CSV matching the manually-created ground truth format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
        writer.writerow(["Unique_ID", "Test_title", "test_scenario"])
        for i, tc in enumerate(test_cases, start=1):
            writer.writerow([i, tc.test_id, tc.scenario])


def save_markdown(test_cases: list[TestCase], path: str | Path) -> None:
    """Write a single Markdown document with all test cases — one section each."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "# Generated Test Cases\n\n"
        f"Total: {len(test_cases)} test cases generated from UN Regulation No. 152.\n\n"
        "---\n\n"
    )
    body = "\n---\n\n".join(
        tc.markdown if tc.markdown else render_markdown(tc)
        for tc in test_cases
    )
    path.write_text(header + body, encoding="utf-8")
