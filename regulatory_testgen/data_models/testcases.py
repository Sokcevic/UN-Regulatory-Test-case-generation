from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from regulatory_testgen.data_models.core import Evidence


class GeneratedTestCase(BaseModel):
    """Final engineer-readable and machine-readable test case."""

    id: str
    title: str
    scenario_type: Literal["physical", "simulation", "review"] = "physical"
    objective: str
    scenario: str
    preconditions: list[str] = Field(default_factory=list)
    test_steps: list[str] = Field(default_factory=list)
    expected_behavior: list[str] = Field(default_factory=list)
    pass_fail_criteria: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    source_clause_ids: list[str] = Field(default_factory=list)
    regulatory_references: list[Evidence] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
