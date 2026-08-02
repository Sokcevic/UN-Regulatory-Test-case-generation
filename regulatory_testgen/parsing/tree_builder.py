from __future__ import annotations

import hashlib
import re

from regulatory_testgen.data_models.core import Clause, DocumentTree, ReferenceLink, SectionNode

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
NUMBER_RE = re.compile(r"^(?P<number>\d+(?:\.\d+)*|Annex\s+\d+(?:\s*-\s*Appendix\s*\d+)?|Scenario\s+\d+)\b", re.I)


def build_document_tree(clauses: list[Clause]) -> DocumentTree:
    root = SectionNode(id="root", title="Document", level=0)
    sections: dict[str, SectionNode] = {root.id: root}
    path_to_id: dict[tuple[str, ...], str] = {(): root.id}

    for clause in clauses:
        path = full_section_path(clause)
        parent_path: tuple[str, ...] = ()
        parent_id = root.id
        for depth, label in enumerate(path, start=1):
            current_path = (*parent_path, label)
            section_id = path_to_id.get(current_path)
            if section_id is None:
                section_id = section_id_for_path(current_path)
                number = section_number(label)
                node = SectionNode(
                    id=section_id,
                    title=label,
                    number=number,
                    level=depth,
                    parent_id=parent_id,
                )
                sections[section_id] = node
                path_to_id[current_path] = section_id
                sections[parent_id].child_ids.append(section_id)
            parent_path = current_path
            parent_id = section_id
        if clause.clause_id not in sections[parent_id].clause_ids:
            sections[parent_id].clause_ids.append(clause.clause_id)

    return DocumentTree(root_id=root.id, sections=sections)


def structural_links(tree: DocumentTree) -> list[ReferenceLink]:
    links: list[ReferenceLink] = []
    for section in tree.sections.values():
        for child_id in section.child_ids:
            links.append(
                ReferenceLink(
                    source_id=section.id,
                    target_id=child_id,
                    source_type="section",
                    target_type="section",
                    relation="parent_of",
                )
            )
        for clause_id in section.clause_ids:
            links.append(
                ReferenceLink(
                    source_id=section.id,
                    target_id=clause_id,
                    source_type="section",
                    target_type="clause",
                    relation="contains",
                )
            )
    return links


def full_section_path(clause: Clause) -> list[str]:
    path = [clean_text(item) for item in clause.section_path if clean_text(item)]
    label = clause_label(clause)
    if label and (not path or path[-1] != label):
        path.append(label)
    if not path:
        path = ["Front matter"]
    return path


def clause_label(clause: Clause) -> str:
    if clause.title:
        return f"{clause.clause_id} {clause.title}".strip()
    return clause.clause_id or clause.title or "Untitled section"


def section_id_for_path(path: tuple[str, ...]) -> str:
    raw = "__".join(path)
    slug = "-".join(token.lower() for token in TOKEN_RE.findall(raw))[:96]
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"section-{slug or 'untitled'}-{digest}"


def section_number(label: str) -> str | None:
    match = NUMBER_RE.match(label.strip())
    return match.group("number") if match else None


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sections_by_depth(tree: DocumentTree) -> dict[int, list[SectionNode]]:
    result: dict[int, list[SectionNode]] = {}
    for section in tree.sections.values():
        result.setdefault(section.level, []).append(section)
    return result


def parent_chain(tree: DocumentTree, section_id: str) -> list[SectionNode]:
    chain: list[SectionNode] = []
    current = tree.sections.get(section_id)
    while current is not None and current.parent_id is not None:
        parent = tree.sections.get(current.parent_id)
        if parent is None:
            break
        chain.append(parent)
        current = parent
    chain.reverse()
    return chain
