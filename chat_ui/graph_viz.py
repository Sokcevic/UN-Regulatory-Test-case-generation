"""
graph_viz.py — Interactive clause-relationship explorer for the chat UI.

Renders the in-memory RegulatoryGraph (LlamaIndex property graph) as an
interactive network with pyvis/vis.js, embedded in Streamlit via an HTML
component. Two views:
  • focused neighborhood — pick a clause, show its CONTAINS parents/children and
    REFERS_TO cross-references out to N hops (the default; legible), and
  • full graph — every clause at once.

The neighborhood computation is pure (operates on an edge list) so it is
unit-testable without a running graph or browser.
"""

from __future__ import annotations

from collections import deque

# Edge styling. CONTAINS = section hierarchy; REFERS_TO = cross-reference.
EDGE_STYLE = {
    "CONTAINS": {"color": "#9aa0a6", "dashes": False},
    "REFERS_TO": {"color": "#4c8bf5", "dashes": True},
}

# Node fill by clause category. Hues are chosen for separability — in particular
# informative / unknown / administrative used to all read as grey.
# Maximally-distinct categorical palette (adapted from Trubetskoy's 20-colour
# set). Every category is a different hue so no two are confusable — in
# particular performance_data (magenta) and unknown (purple) are now clearly
# apart. Related families (test_*) still cluster loosely by hue but each is
# individually distinguishable.
CATEGORY_COLOR = {
    "obligation": "#e6194b",       # red
    "test_execution": "#f58231",   # orange
    "test_procedure": "#ffd8b1",   # apricot (pale orange)
    "performance_data": "#f032e6", # magenta
    "test_condition": "#4363d8",   # blue
    "test_setup": "#42d4f4",       # cyan
    "definition": "#3cb44b",       # green
    "scope": "#bfef45",            # lime
    "informative": "#469990",      # teal
    "unknown": "#d0cdd6",          # neutral grey — catch-all/meta, deliberately
                                   # not a hue so it never competes with a real
                                   # category (in particular performance_data's
                                   # magenta). Unclassified reads as "greyed out".
    "administrative": "#5a5a8c",   # muted indigo
    "formatting": "#9a6324",       # brown
}
_DEFAULT_NODE_COLOR = "#cccccc"


def edges_from_index(clause_index: dict) -> list[tuple[str, str, str]]:
    """Build the display graph directly from the repaired index so it respects
    annex namespacing: CONTAINS from each clause's `parent`, REFERS_TO from its
    `references`. (Decoupled from RegulatoryGraph, whose ids aren't namespaced.)"""
    ids = set(clause_index)
    edges: list[tuple[str, str, str]] = []
    for cid, info in clause_index.items():
        parent = info.get("parent")
        if parent and parent in ids:
            edges.append((parent, "CONTAINS", cid))
        for ref in (info.get("references") or []):
            if ref in ids and ref != cid:
                edges.append((cid, "REFERS_TO", ref))
    return edges


def neighborhood(
    edges: list[tuple[str, str, str]],
    focus: str,
    hops: int = 1,
    edge_types: tuple[str, ...] = ("CONTAINS", "REFERS_TO"),
) -> tuple[set[str], list[tuple[str, str, str]]]:
    """BFS out from `focus` over the selected edge types, treating edges as
    undirected for reachability. Returns (node_ids, edges_within_subgraph).

    Only edges of a selected type are traversed AND returned, and a returned
    edge is kept only when both endpoints are within the discovered node set."""
    allowed = set(edge_types)
    adj: dict[str, list[str]] = {}
    for src, rel, tgt in edges:
        if rel not in allowed:
            continue
        adj.setdefault(src, []).append(tgt)
        adj.setdefault(tgt, []).append(src)

    visited: set[str] = {focus}
    frontier: deque[tuple[str, int]] = deque([(focus, 0)])
    while frontier:
        node, depth = frontier.popleft()
        if depth >= hops:
            continue
        for nbr in adj.get(node, []):
            if nbr not in visited:
                visited.add(nbr)
                frontier.append((nbr, depth + 1))

    sub_edges = [
        (s, r, t) for (s, r, t) in edges
        if r in allowed and s in visited and t in visited
    ]
    return visited, sub_edges


def filter_edges(
    edges: list[tuple[str, str, str]],
    edge_types: tuple[str, ...],
) -> tuple[set[str], list[tuple[str, str, str]]]:
    """Full-graph view: keep only edges of the selected types, plus their nodes."""
    allowed = set(edge_types)
    kept = [(s, r, t) for (s, r, t) in edges if r in allowed]
    nodes = {n for (s, _r, t) in kept for n in (s, t)}
    return nodes, kept


def _node_label(cid: str, clause_index: dict) -> str:
    return cid


def _node_tooltip(cid: str, clause_index: dict) -> str:
    info = clause_index.get(cid, {})
    title = info.get("title") or "(untitled)"
    cat = info.get("category", "?")
    return f"{cid} — {title} [{cat}]"


def _esc(s: str) -> str:
    return (
        str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_tree_html(clause_index: dict, open_depth: int = 0) -> str:
    """A collapsible folder-tree of the document (native <details>/<summary>,
    no JS). Sections nest by the explicit `parent` field, so annex sections
    appear under their annex (e.g. 'Annex 3' ▸ 'Annex 3 / 5').

    open_depth=0 → everything collapsed (see sections 1..n, expand as needed).
    Theme-aware (light/dark) with a transparent background so it blends into the
    Streamlit page rather than reading as a pasted-in white box."""
    from normalize import natkey

    children: dict[str, list[str]] = {}
    for cid, info in clause_index.items():
        parent = info.get("parent")
        if parent:
            children.setdefault(parent, []).append(cid)
    for parent in children:
        children[parent].sort(key=natkey)
    roots = sorted([c for c, i in clause_index.items() if not i.get("parent")], key=natkey)

    def _count_desc(cid: str) -> int:
        kids = children.get(cid, [])
        return len(kids) + sum(_count_desc(k) for k in kids)

    def node(cid: str, depth: int) -> str:
        info = clause_index.get(cid, {})
        cat = info.get("category", "")
        color = CATEGORY_COLOR.get(cat, _DEFAULT_NODE_COLOR)
        short = _esc(cid.split(" / ")[-1])          # inner id segment; full id in tooltip
        kids = children.get(cid, [])
        count = f'<span class="count">{_count_desc(cid)}</span>' if kids else ""
        label = (
            f'<span class="dot" style="background:{color}"></span>'
            f'<span class="cid" title="{_esc(cid)}">{short}</span>'
            f'<span class="tt">{_esc(info.get("title") or "(untitled)")}</span>'
            f'{count}'
        )
        if kids:
            is_open = " open" if depth < open_depth else ""
            inner = "".join(node(k, depth + 1) for k in kids)
            return (f'<details{is_open}><summary><span class="row">{label}</span></summary>'
                    f'<div class="ch">{inner}</div></details>')
        return f'<div class="leaf"><span class="row">{label}</span></div>'

    body = "".join(node(r, 0) for r in roots)
    css = """
    <style>
      :root{
        --fg:#31333f; --muted:#808495; --hover:rgba(120,120,140,.12);
        --line:rgba(120,120,140,.28); --chip:rgba(120,120,140,.16);
      }
      @media (prefers-color-scheme: dark){
        :root{ --fg:#e6e6ea; --muted:#9aa0ac; --hover:rgba(255,255,255,.07);
               --line:rgba(255,255,255,.16); --chip:rgba(255,255,255,.10); }
      }
      html,body{margin:0;background:transparent;}
      .tree{font-family:"Source Sans Pro","Segoe UI",-apple-system,Roboto,sans-serif;
            font-size:13.5px;line-height:1.5;color:var(--fg);background:transparent;padding:2px 2px 8px;}
      .tree details{margin:0;}
      .tree summary{list-style:none;outline:none;}
      .tree summary::-webkit-details-marker{display:none;}
      .tree .row{display:flex;align-items:center;gap:7px;cursor:default;
            padding:3px 8px;border-radius:7px;transition:background .08s;}
      .tree summary .row{cursor:pointer;}
      .tree summary .row:hover, .tree .leaf .row:hover{background:var(--hover);}
      .tree summary .row::before{content:"▸";color:var(--muted);font-size:11px;
            width:.8em;display:inline-block;transition:transform .12s;flex:none;}
      .tree details[open]>summary .row::before{transform:rotate(90deg);}
      .tree .ch{margin-left:.75em;border-left:1px solid var(--line);padding-left:.5em;}
      .tree .leaf .row{padding-left:calc(8px + .8em + 7px);}   /* align with expandable rows */
      .tree .dot{width:9px;height:9px;border-radius:50%;flex:none;}
      .tree .cid{font-weight:600;font-variant-numeric:tabular-nums;flex:none;}
      .tree .tt{color:var(--fg);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
      .tree .count{margin-left:auto;flex:none;color:var(--muted);font-size:11px;
            background:var(--chip);border-radius:10px;padding:0 7px;}
    </style>
    """
    return css + f'<div class="tree">{body}</div>'


def build_network_html(
    node_ids: set[str],
    sub_edges: list[tuple[str, str, str]],
    clause_index: dict,
    focus: str | None = None,
    height: str = "600px",
) -> str:
    """Build a self-contained (offline) pyvis HTML document for the subgraph."""
    from pyvis.network import Network

    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#202124",
        cdn_resources="in_line",   # inline vis.js so it renders offline / under CSP
    )
    net.barnes_hut(spring_length=140, gravity=-8000)

    for cid in sorted(node_ids):
        info = clause_index.get(cid, {})
        color = CATEGORY_COLOR.get(info.get("category", ""), _DEFAULT_NODE_COLOR)
        is_focus = cid == focus
        net.add_node(
            cid,
            label=_node_label(cid, clause_index),
            title=_node_tooltip(cid, clause_index),
            color={"background": color, "border": "#202124" if is_focus else color},
            borderWidth=4 if is_focus else 1,
            size=28 if is_focus else 18,
            shape="dot",
        )

    for src, rel, tgt in sub_edges:
        if src not in node_ids or tgt not in node_ids:
            continue
        style = EDGE_STYLE.get(rel, {"color": "#cccccc", "dashes": False})
        net.add_edge(
            src, tgt,
            title=rel,
            color=style["color"],
            dashes=style["dashes"],
            arrows="to",
        )

    # pyvis 0.3.x: generate_html() returns the full document string.
    return net.generate_html(notebook=False)
