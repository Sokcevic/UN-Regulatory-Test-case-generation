"""Shared, pure computation of the generation-context retrieval set.

Both the deployed graph (`RegulatoryGraph.get_generation_context`) and the
weak-label validator need to answer the same question — *which clauses does a
test-case generator get fed for a seed clause?* — but they back onto different
stores (a LlamaIndex property graph vs. the chat `clause_index` JSON). To keep
the two provably identical, the algorithm lives here once, expressed over small
accessor callables; each caller supplies adapters for its store. A consistency
test builds both adapters from one fixture and asserts they agree.

This is STEP 1 of a two-step retrieval (see evaluate/augment/RETRIEVAL_STRATEGY.md):
step 1 retrieves the OBLIGATION context for a requirement seed; step 2
(procedure_context_from_index) separately retrieves the test-procedure context.
Keeping them separate lets each stay focused — the obligation gold labels exclude
test-procedure sections, and the procedures are added back only when a test is
finally generated.

The obligation context, in order, is:

  1. the seed clause;
  2. its parent-section chain (ancestors), capped at the formatting root;
  3. the transitive **outgoing** REFERS_TO closure of (1)+(2) — clauses they cite;
  4. **self_sub** — the seed's own sub-clauses plus its immediate parent's other
     children (the sibling requirement sub-block);
  5. **fmt-root closure** — climb to the nearest `formatting`-labelled ancestor (a
     titled section/subsection header), take its whole subtree, and add that
     block's reference closure.

References are followed **bidirectionally** (cites + cited-by) and **transitively**
(to a fixpoint) in the deployed path, while **excluding testing-labelled clauses**
from both retrieval and traversal. This reaches related *obligation* sections in
either direction — the way an engineer follows a requirement and the requirements
that point back at it — without dragging in the test procedures (those come from
step 2). Uni-directional, bounded-hop, and include-testing all remain available as
options for measurement.

Rationale and measurements: evaluate/augment/FULLSUPPORT.md (§4b),
evaluate/augment/RETRIEVAL_STRATEGY.md.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

# Accessor signatures (all take a clause id):
#   parent_of(cid)   -> nearest PRESENT parent id, or None
#   children_of(cid) -> list of direct child ids (present)
#   refs_out(cid)    -> list of present clause ids this clause REFERS_TO
#   refs_in(cid)     -> list of present clause ids that REFER_TO this clause
#   category_of(cid) -> the clause's category string, or None
#   present(cid)     -> True if cid is a real clause in the store
Accessor = Callable[[str], object]

_ANCESTOR_LIMIT = 20
_SUBTREE_LIMIT = 200
_BLOCK_REF_HOPS = 2

# Categories that mark a clause as testing-related (step 2 seeds / the "test"
# filter used by exclude_testing). NOTE: `performance_data` is deliberately NOT
# here — performance tables are the values a requirement is tested against (part
# of the obligation), not the how-to-test procedure, so they stay in obligation
# retrieval rather than being excluded.
PROC_CATEGORIES = frozenset({
    "test_condition", "test_execution", "test_setup", "test_procedure",
})

# ─────────────────────────────────────────────────────────────────────────────
# THE retrieval strategy. This is FIXED in code and deterministic — retrieval for
# test-case generation is not a tunable knob. There is exactly one traversal
# (cap at the formatting root, bidirectional + transitive references); the only
# variation is whether testing-labelled clauses are excluded:
#   * OBLIGATION_STRATEGY (step 1, the default for any requirement retrieval)
#     excludes them — reach related obligations both ways without dragging in §6.
#   * PROCEDURE_STRATEGY (step 2, exposed to the model as the get_test_procedures
#     tool) keeps them — used only when test procedures are what's wanted.
# Both are spread into generation_context_ids at their single call sites so the
# strategy has one source of truth. The individual flags remain as parameters for
# offline MEASUREMENT only (evaluate/augment/*.py); production never varies them.
# ─────────────────────────────────────────────────────────────────────────────
OBLIGATION_STRATEGY = dict(
    cap_ancestors_at_formatting_root=True, transitive_refs=True,
    bidirectional_refs=True, exclude_testing=True,
)
PROCEDURE_STRATEGY = dict(
    cap_ancestors_at_formatting_root=True, transitive_refs=True,
    bidirectional_refs=True, exclude_testing=False,
)


def _ancestors(seed: str, parent_of: Accessor) -> list[str]:
    out: list[str] = []
    cur = parent_of(seed)
    seen = 0
    while cur and seen < _ANCESTOR_LIMIT:
        out.append(cur)  # type: ignore[arg-type]
        cur = parent_of(cur)  # type: ignore[arg-type]
        seen += 1
    return out


def _descendants(root: str, children_of: Accessor, limit: int = _SUBTREE_LIMIT) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    queue: deque[str] = deque([root])
    while queue and len(out) < limit:
        for c in children_of(queue.popleft()):  # type: ignore[union-attr]
            if c not in seen:
                seen.add(c)
                out.append(c)
                queue.append(c)
    return out


def _formatting_root(seed: str, parent_of: Accessor, category_of: Accessor) -> str | None:
    """Nearest ancestor whose category is 'formatting' — the tightest enclosing
    titled section/subsection header. None if there is none above the seed."""
    cur = parent_of(seed)
    seen = 0
    while cur and seen < _ANCESTOR_LIMIT:
        if category_of(cur) == "formatting":  # type: ignore[arg-type]
            return cur  # type: ignore[return-value]
        cur = parent_of(cur)  # type: ignore[arg-type]
        seen += 1
    return None


def _ref_closure(roots: set[str], refs_out: Accessor, refs_in: Accessor,
                 present: Accessor, hops: "int | None" = _BLOCK_REF_HOPS,
                 bidirectional: bool = True,
                 blocked: "Callable[[str], bool] | None" = None) -> set[str]:
    """REFERS_TO closure seeded from a block. Always follows OUTGOING references
    (what each clause cites); if bidirectional, ALSO incoming (what cites it).

    hops=None runs to a fixpoint (transitive closure) — keep following references
    of newly-reached clauses until nothing new appears. A bounded integer stops
    after that many hops.

    blocked(cid): if given, such clauses are neither added NOR traversed — the
    closure stops at them, so they cannot bridge to further clauses."""
    block = blocked or (lambda _c: False)
    out: set[str] = set()
    frontier = {r for r in roots if not block(r)}
    steps = 0
    while frontier and (hops is None or steps < hops):
        nxt: set[str] = set()
        for cid in frontier:
            nxt.update(refs_out(cid))       # type: ignore[arg-type]
            if bidirectional:
                nxt.update(refs_in(cid))    # type: ignore[arg-type]
        nxt = {r for r in nxt if r not in out and not block(r)}
        out |= nxt
        frontier = {r for r in nxt if present(r)}
        steps += 1
    return out


def generation_context_ids(
    seed: str,
    *,
    parent_of: Accessor,
    children_of: Accessor,
    refs_out: Accessor,
    refs_in: Accessor,
    category_of: Accessor,
    present: Accessor,
    cap_ancestors_at_formatting_root: bool = False,
    transitive_refs: bool = False,
    bidirectional_refs: bool = True,
    exclude_testing: bool = False,
) -> list[str]:
    """Return the ordered, de-duplicated clause ids of the generation context for
    `seed` (see module docstring). Seed first, then ancestors and the outgoing
    reference closure, then the sibling sub-block, then the formatting-root block
    with its bidirectional reference closure.

    cap_ancestors_at_formatting_root: if True and the seed has a formatting-root
    ancestor, the ancestor chain (step 2) is truncated *at* that root — sections
    above the nearest titled header are dropped instead of climbing to the very
    top. The formatting-root block itself is unaffected. Default False here, but
    **the deployed path passes True** (graph.get_generation_context and the
    validator's recompute): retrieval is rooted at the nearest titled section.
    Note: on the current WEAK labels raw recall drops (68.7->61.5) but this is
    entirely parent SECTION-HEADER nodes — all 26 gold clauses capping removes are
    gold-type 'parent' (20 formatting, 6 unknown; zero real-content clauses). Under
    recall_parents_credited (which credits design-guaranteed parents) capping is
    exactly neutral: 71.9% either way. The small precision dip (35.1->33.3) is
    those same headers being counted as gold true-positives; if section headers are
    treated as not-gold, capping instead raises precision. See
    evaluate/augment/cap_ancestors.py.

    transitive_refs: if True, the formatting block's reference closure runs to a
    fixpoint (keep following references of newly-reached clauses until nothing new
    appears) instead of the default 2 hops. Default False here, but **the deployed
    path passes True** — surface the full chain of related sections without
    returning the whole document.

    bidirectional_refs: if True (default), the block's closure follows references
    in BOTH directions (cites + cited-by). **The deployed path passes True together
    with exclude_testing=True** — bidirectional to reach related obligation sections
    either way, with procedures blocked so the incoming direction doesn't drag in
    §6. Set False for the uni-directional (outgoing-only) variant.

    exclude_testing: if True, clauses whose category is a testing label
    (PROC_CATEGORIES) are dropped from the result AND from traversal — the closure
    stops at them, so a procedure cannot bridge to further clauses. Lets the
    bidirectional closure reach related *obligation* sections (via incoming
    references) without pulling in the procedures those references belong to."""
    order: list[str] = []
    visited: set[str] = set()

    def blocked(cid: str) -> bool:
        # exclude_testing: testing-labelled clauses are neither retrieved nor
        # traversed (they can't bridge the closure to further clauses).
        return exclude_testing and category_of(cid) in PROC_CATEGORIES

    def add(cid: str) -> None:
        if cid in visited:
            return
        visited.add(cid)          # mark seen even if blocked, so it isn't retried
        if not blocked(cid):
            order.append(cid)

    ancestors = _ancestors(seed, parent_of)
    root = _formatting_root(seed, parent_of, category_of)

    # optionally truncate the ancestor chain at (and including) the formatting root
    if cap_ancestors_at_formatting_root and root is not None and root in ancestors:
        ancestors = ancestors[: ancestors.index(root) + 1]

    # (1)-(3) base: BFS over OUTGOING references from seed + ancestors
    queue: deque[str] = deque([seed, *ancestors])
    while queue:
        cid = queue.popleft()
        if cid in visited:
            continue
        add(cid)
        if blocked(cid):          # do not follow references of a blocked clause
            continue
        for tgt in refs_out(cid):  # type: ignore[union-attr]
            if tgt not in visited:
                queue.append(tgt)

    # (4) self_sub: seed's descendants + immediate parent's other children
    sib_block = _descendants(seed, children_of)
    if ancestors:
        sib_block += list(children_of(ancestors[0]))  # type: ignore[arg-type]
    for cid in sib_block:
        add(cid)

    # (5) fmt_root_biref: nearest titled-header block + bidirectional ref closure
    if root is not None:
        block = [root, *_descendants(root, children_of)]
        for cid in block:
            add(cid)
        hops = None if transitive_refs else _BLOCK_REF_HOPS
        for cid in _ref_closure(set(block), refs_out, refs_in, present, hops=hops,
                                bidirectional=bidirectional_refs, blocked=blocked):
            add(cid)

    return order


def _index_accessors(clause_index: dict):
    """Build the (parent_of, children_of, refs_out, refs_in, category_of, present)
    accessors for a chat clause_index dict."""
    children: dict[str, list[str]] = {}
    rev: dict[str, list[str]] = {}
    for cid, info in clause_index.items():
        parent = (info or {}).get("parent")
        if parent:
            children.setdefault(parent, []).append(cid)
        for r in ((info or {}).get("references") or []):
            rev.setdefault(str(r), []).append(cid)

    def refs_out(cid: str) -> list[str]:
        refs = ((clause_index.get(cid) or {}).get("references") or [])
        return [str(r) for r in refs if str(r) in clause_index]

    return dict(
        parent_of=lambda c: (clause_index.get(c) or {}).get("parent"),
        children_of=lambda c: children.get(c, []),
        refs_out=refs_out,
        refs_in=lambda c: rev.get(c, []),
        category_of=lambda c: (clause_index.get(c) or {}).get("category"),
        present=lambda c: c in clause_index,
    )


def generation_context_from_index(clause_index: dict, seed: str,
                                  cap_ancestors_at_formatting_root: bool = False,
                                  transitive_refs: bool = False,
                                  bidirectional_refs: bool = True,
                                  exclude_testing: bool = False) -> list[str]:
    """Convenience adapter: compute the STEP-1 obligation context from a chat
    `clause_index` dict — {clause_id: {"category", "parent", "references", …}}.

    This is the exact set the deployed graph returns (same algorithm), computed
    without a graph — used by the weak-label validator to preview the deployed
    retrieval, and by the graph/index consistency test."""
    return generation_context_ids(
        seed,
        **_index_accessors(clause_index),
        cap_ancestors_at_formatting_root=cap_ancestors_at_formatting_root,
        transitive_refs=transitive_refs,
        bidirectional_refs=bidirectional_refs,
        exclude_testing=exclude_testing,
    )


def procedure_context_from_index(clause_index: dict,
                                 transitive_refs: bool = True,
                                 bidirectional_refs: bool = False) -> list[str]:
    """STEP 2: the test-procedure context for a document.

    Every clause whose category is a testing label (PROC_CATEGORIES) is treated
    like an obligation seed — climb to its formatting root, take the subtree, and
    take the uni-directional transitive reference closure — and the results are
    unioned. Document-level and seed-independent: this is the procedure knowledge a
    generated test needs regardless of which requirement it targets. Combined with
    step 1 only when a test is actually generated, never mixed into the obligation
    gold used for retrieval evaluation."""
    acc = _index_accessors(clause_index)
    proc_seeds = [cid for cid, info in clause_index.items()
                  if (info or {}).get("category") in PROC_CATEGORIES]
    out: list[str] = []
    seen: set[str] = set()
    for s in proc_seeds:
        for cid in generation_context_ids(
                s, **acc, cap_ancestors_at_formatting_root=True,
                transitive_refs=transitive_refs, bidirectional_refs=bidirectional_refs):
            if cid not in seen:
                seen.add(cid)
                out.append(cid)
    return out
