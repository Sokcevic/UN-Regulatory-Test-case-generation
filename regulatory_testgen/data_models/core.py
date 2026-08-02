from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """Short traceability reference back to the regulatory source."""

    clause_id: str
    source: str
    line_start: int
    line_end: int
    quote: str = ""


class Clause(BaseModel):
    """A structurally parsed segment of regulatory text."""

    # Internal unique id. clause_id is the legal paragraph number and may repeat
    # inside annexes or forms.
    uid: str = ""
    clause_id: str
    title: str = ""
    text: str
    source: str
    line_start: int
    line_end: int
    section_path: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    is_pseudo_clause: bool = False

    # Structural region only. Do not use this as semantic classification.
    document_region: str = "unknown"

    @property
    def citation(self) -> str:
        title = f" {self.title}" if self.title else ""
        return f"{self.source}:{self.line_start}-{self.line_end} [{self.clause_id}{title}]"


class SectionNode(BaseModel):
    """Pure structural node. Semantic annotations live elsewhere."""

    id: str
    title: str
    number: str | None = None
    level: int = 0
    parent_id: str | None = None
    child_ids: list[str] = Field(default_factory=list)
    clause_ids: list[str] = Field(default_factory=list)


class DocumentTree(BaseModel):
    root_id: str
    sections: dict[str, SectionNode] = Field(default_factory=dict)


class ReferenceLink(BaseModel):
    """Typed edge between knowledge objects."""

    source_id: str
    target_id: str
    source_type: Literal["section", "clause", "table", "requirement"] = "clause"
    target_type: Literal[
        "section",
        "clause",
        "table",
        "requirement",
        "annex",
        "unknown",
    ] = "unknown"
    relation: Literal[
        "references",
        "parent_of",
        "contains",
        "defines",
        "tested_by",
        "constrained_by",
        "related_to",
    ] = "references"
    text: str = ""
    weight: float = 1.0
