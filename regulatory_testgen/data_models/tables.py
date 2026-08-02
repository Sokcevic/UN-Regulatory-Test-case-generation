from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RegulationTable(BaseModel):
    """First-class table extracted from a regulatory clause."""

    table_id: str
    owner_clause_id: str
    title: str = ""
    table_type: Literal[
        "performance_limit",
        "test_speed_matrix",
        "form",
        "definition_table",
        "unknown",
    ] = "unknown"
    headers: list[str] = Field(default_factory=list)
    rows: list[dict[str, str]] = Field(default_factory=list)
    units: dict[str, str] = Field(default_factory=dict)
    raw_html: str | None = None
    raw_markdown: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
