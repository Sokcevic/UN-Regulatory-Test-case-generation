"""Regulatory Reference Graph.

Builds a LlamaIndex SimplePropertyGraphStore from parsed clauses and tables.
Each clause is an EntityNode (label="CLAUSE"). Cross-references extracted by the
parser become directed REFERS_TO edges. Parent-child section hierarchy becomes
CONTAINS edges.

The key operation — get_transitive_context — performs a BFS over REFERS_TO edges
to collect the complete transitive closure of all clauses referenced from a given
starting clause. This guarantees that no required context is silently dropped when
the LLM generates a test case.

Literature basis:
  BifrostRAG (Zhang et al., 2025)       — iterative REFERS_TO traversal on safety regs
  GraphCompliance (Chung et al., 2025)  — bidirectional reference closure
  ComplianceNLP (Guo et al., 2026)      — +16.8 F1 from graph-based re-ranking
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, Relation

if TYPE_CHECKING:
    from regulatory_testgen.data_models.core import Clause
    from regulatory_testgen.data_models.tables import RegulationTable


class RegulatoryGraph:
    """In-memory property graph of regulatory clauses with transitive retrieval."""

    def __init__(
        self,
        store: SimplePropertyGraphStore,
        clause_map: dict[str, "Clause"],
        table_map: dict[str, list["RegulationTable"]],
    ) -> None:
        self._store = store
        self._clause_map = clause_map
        self._table_map = table_map
        # clause_id -> semantic category ('formatting', 'obligation', …). Not
        # carried by the Clause model (it is structural only), so it is plumbed
        # in from the classification step via set_categories(). Empty by default;
        # when empty, get_generation_context simply omits the formatting-root
        # block (fmt_root_biref) and behaves as base + self_sub.
        self._categories: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        clauses: list["Clause"],
        tables: list["RegulationTable"],
    ) -> "RegulatoryGraph":
        """Build the graph from parsed clauses and extracted tables.

        Only non-pseudo clauses are inserted as nodes. REFERS_TO edges come from
        cross-references already extracted by the Markdown parser (regex-based).
        CONTAINS edges reflect the numeric parent-child hierarchy (e.g., 5.2 → 5.2.1).
        """
        from regulatory_testgen.structure_ids import namespace_clauses, parent_of

        # Namespace annex clauses so their numbering no longer collides with the
        # body (every annex restarts at 1). Keeps the graph's id space identical
        # to the chat clause index. Idempotent.
        clauses = namespace_clauses(clauses)

        store = SimplePropertyGraphStore()
        real_clauses = [c for c in clauses if not c.is_pseudo_clause]
        known_ids = {c.clause_id for c in real_clauses}

        # --- nodes -------------------------------------------------------
        nodes = [
            EntityNode(
                name=c.clause_id,
                label="CLAUSE",
                properties={
                    "clause_id": c.clause_id,
                    "title": c.title,
                    "text": c.text,
                    "document_region": c.document_region,
                    "section_path": " > ".join(c.section_path),
                },
            )
            for c in real_clauses
        ]
        store.upsert_nodes(nodes)

        # --- edges -------------------------------------------------------
        relations: list[Relation] = []

        # REFERS_TO: cross-references extracted from clause text by the parser
        for c in real_clauses:
            for ref_id in c.references:
                if ref_id in known_ids and ref_id != c.clause_id:
                    relations.append(
                        Relation(
                            label="REFERS_TO",
                            source_id=c.clause_id,
                            target_id=ref_id,
                        )
                    )

        # CONTAINS: structural parent-child via the namespaced-aware parent_of
        # ("5.2"→"5.2.1" and "Annex 3 / 5"→"Annex 3 / 5.1"→"Annex 3"), climbing to
        # the nearest present ancestor so gaps/pruned intermediates don't break it.
        for c in real_clauses:
            p = parent_of(c.clause_id)
            while p is not None and p not in known_ids:
                p = parent_of(p)
            if p is not None:
                relations.append(
                    Relation(label="CONTAINS", source_id=p, target_id=c.clause_id)
                )

        store.upsert_relations(relations)

        clause_map = {c.clause_id: c for c in real_clauses}
        table_map: dict[str, list["RegulationTable"]] = {}
        for t in tables:
            table_map.setdefault(t.owner_clause_id, []).append(t)

        return cls(store, clause_map, table_map)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_clause(self, clause_id: str) -> "Clause | None":
        return self._clause_map.get(clause_id)

    def get_tables(self, clause_id: str) -> list["RegulationTable"]:
        return self._table_map.get(clause_id, [])

    def all_clause_ids(self) -> list[str]:
        return list(self._clause_map.keys())

    def get_transitive_context(self, clause_id: str) -> list["Clause"]:
        """BFS transitive closure over REFERS_TO edges.

        Starting from clause_id, follows every outgoing REFERS_TO edge recursively
        until no new clauses are reachable. Returns the collected clauses in BFS
        order (starting clause first).

        This implements the core insight from BifrostRAG: iterating until no new
        sections are added guarantees that all transitively required context is
        included, regardless of cross-reference chain depth.
        """
        visited: set[str] = set()
        queue: deque[str] = deque([clause_id])
        result: list["Clause"] = []

        while queue:
            cid = queue.popleft()
            if cid in visited:
                continue
            visited.add(cid)

            clause = self._clause_map.get(cid)
            if clause is not None:
                result.append(clause)

            # Get outgoing REFERS_TO triplets from this node.
            # Triplet layout: (source EntityNode, Relation, target EntityNode)
            triplets = self._store.get_triplets(
                entity_names=[cid],
                relation_names=["REFERS_TO"],
            )
            for src, _rel, tgt in triplets:
                if src.id == cid and tgt.id not in visited:
                    queue.append(tgt.id)

        return result

    def get_ancestors(self, clause_id: str) -> list[str]:
        """Return the parent-section chain for a clause, nearest first.

        e.g. get_ancestors("5.2.1") -> ["5.2", "5"] (only IDs present in the graph).
        These are the CONTAINS parents — the sections that hold the general
        requirements a sub-clause specialises, which REFERS_TO traversal alone
        never reaches.
        """
        from regulatory_testgen.structure_ids import parent_of

        ancestors: list[str] = []
        p = parent_of(clause_id)
        while p is not None:
            if p in self._clause_map:
                ancestors.append(p)
            p = parent_of(p)
        return ancestors

    def set_categories(self, categories: "dict[str, str | None]") -> None:
        """Attach per-clause semantic categories (from the classification step).
        Enables the formatting-root block in get_generation_context. Only ids
        present in the graph with a non-empty category are kept."""
        self._categories = {
            cid: cat for cid, cat in categories.items()
            if cid in self._clause_map and cat
        }

    def get_children(self, clause_id: str) -> list[str]:
        """Direct CONTAINS children of a clause — the sub-clauses one level down
        (e.g. get_children("5.2.1") -> ["5.2.1.1", "5.2.1.2", ...]). Only ids
        present in the graph are returned."""
        triplets = self._store.get_triplets(
            entity_names=[clause_id], relation_names=["CONTAINS"]
        )
        return [tgt.id for src, _rel, tgt in triplets if src.id == clause_id]

    def _refs_out(self, clause_id: str) -> list[str]:
        """Present clauses this clause REFERS_TO (outgoing cross-references)."""
        triplets = self._store.get_triplets(
            entity_names=[clause_id], relation_names=["REFERS_TO"])
        return [tgt.id for src, _rel, tgt in triplets
                if src.id == clause_id and tgt.id in self._clause_map]

    def _refs_in(self, clause_id: str) -> list[str]:
        """Present clauses that REFER_TO this clause (incoming cross-references)."""
        triplets = self._store.get_triplets(
            entity_names=[clause_id], relation_names=["REFERS_TO"])
        return [src.id for src, _rel, tgt in triplets
                if tgt.id == clause_id and src.id in self._clause_map]

    def get_generation_context(self, clause_id: str) -> list["Clause"]:
        """Complete context for generating test cases from clause_id.

        Delegates the id-set computation to the shared, store-agnostic algorithm
        in regulatory_testgen.context_expand (so the deployed retrieval and the
        weak-label validator's preview cannot drift). The context is, in order:
        the clause; its parent-section chain; the transitive OUTGOING REFERS_TO
        closure of those; the seed's sibling sub-block (self_sub); and — when
        categories have been attached via set_categories — the nearest
        `formatting`-labelled (titled-header) ancestor's whole subtree plus that
        block's BIDIRECTIONAL reference closure (fmt_root_biref). Without
        categories, the formatting-root block is simply omitted.

        The ancestor chain is CAPPED at the formatting root — we climb only to the
        nearest titled header, not the very top section (see context_expand's
        cap_ancestors_at_formatting_root). When categories are absent there is no
        root to cap at, so the full chain is kept. References are followed
        BIDIRECTIONALLY (cites + cited-by) and TRANSITIVELY (to a fixpoint), while
        EXCLUDING testing-labelled clauses from both retrieval and traversal — so
        related obligation sections are reached in either direction but test
        procedures are not pulled in. This is step 1 (obligations); test procedures
        are retrieved separately (step 2). See
        evaluate/augment/RETRIEVAL_STRATEGY.md.

        See evaluate/augment/FULLSUPPORT.md (§4b) for the rationale/measurements.
        Returned with the target clause first, de-duplicated.
        """
        from regulatory_testgen.context_expand import (
            generation_context_ids, OBLIGATION_STRATEGY)
        from regulatory_testgen.structure_ids import parent_of as _parent_of

        def parent_of(cid: str) -> str | None:
            p = _parent_of(cid)
            while p is not None and p not in self._clause_map:
                p = _parent_of(p)
            return p

        ids = generation_context_ids(
            clause_id,
            parent_of=parent_of,
            children_of=self.get_children,
            refs_out=self._refs_out,
            refs_in=self._refs_in,
            category_of=lambda cid: self._categories.get(cid),
            present=lambda cid: cid in self._clause_map,
            **OBLIGATION_STRATEGY,
        )
        return [self._clause_map[c] for c in ids if c in self._clause_map]

    def all_edges(self) -> list[tuple[str, str, str]]:
        """Every edge as (source_clause_id, relation_label, target_clause_id).

        Used by the graph explorer UI to render the clause network. Covers both
        REFERS_TO (cross-references) and CONTAINS (section hierarchy)."""
        triplets = self._store.get_triplets(ids=list(self._clause_map.keys()))
        return [(src.id, rel.id, tgt.id) for src, rel, tgt in triplets]

    def stats(self) -> dict:
        """Return summary statistics for logging."""
        all_triplets = self._store.get_triplets(ids=list(self._clause_map.keys()))
        refers_to = sum(1 for _, r, _ in all_triplets if r.id == "REFERS_TO")
        contains = sum(1 for _, r, _ in all_triplets if r.id == "CONTAINS")
        return {
            "clauses": len(self._clause_map),
            "tables": sum(len(v) for v in self._table_map.values()),
            "refers_to_edges": refers_to,
            "contains_edges": contains,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist graph stats and clause index to JSON for inspection."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "stats": self.stats(),
            "clauses": [
                {
                    "clause_id": c.clause_id,
                    "title": c.title,
                    "document_region": c.document_region,
                    "references": c.references,
                }
                for c in self._clause_map.values()
            ],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
