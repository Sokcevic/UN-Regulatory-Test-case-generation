from __future__ import annotations

from pydantic import BaseModel, Field


class TestCandidate(BaseModel):
    """Deterministic intermediate object that defines one possible test case."""

    candidate_id: str
    requirement_ids: list[str] = Field(default_factory=list)
    source_clause_ids: list[str] = Field(default_factory=list)
    procedure_clause_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    scenario_family: str = ""
    target_type: str = ""
    vehicle_category: str = ""
    load_condition: str = ""
    parameters: dict[str, str] = Field(default_factory=dict)
    expected_limits: dict[str, str] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
