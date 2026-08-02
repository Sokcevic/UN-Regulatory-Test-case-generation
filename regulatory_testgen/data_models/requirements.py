from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Requirement(BaseModel):
    """Normalized requirement extracted from one regulatory clause."""

    requirement_id: str
    source_clause_id: str
    requirement_type: Literal[
        "performance",
        "warning",
        "activation",
        "deactivation",
        "failure_detection",
        "documentation",
        "test_condition",
        "procedure",
        "other",
    ] = "other"
    actor: str | None = None
    legal_text: str
    engineering_summary: str = ""
    conditions: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    referenced_clause_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    source_section_id: str | None = None
    fallback_used: bool = False
