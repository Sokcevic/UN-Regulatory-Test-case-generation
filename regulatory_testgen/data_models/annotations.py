from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SectionAnnotation(BaseModel):
    """LLM or fallback annotation for one structural section."""

    section_id: str
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)
    regulatory_function: Literal[
        "performance_requirement",
        "test_procedure",
        "definition",
        "parameter_table",
        "administrative",
        "documentation_requirement",
        "informative",
        "unknown",
    ] = "unknown"
    test_generation_role: Literal[
        "generate_test",
        "procedure_context",
        "acceptance_criteria",
        "context_only",
        "ignore",
        "unknown",
    ] = "unknown"
    relevance: Literal["high", "medium", "low", "none", "unknown"] = "unknown"
    requires_context: bool = False
    missing_context_refs: list[str] = Field(default_factory=list)
    fallback_used: bool = False
