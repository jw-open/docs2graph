"""
Section-based graph extraction — pure Python, no NLP dependencies.

Strategies:
- extract_section_graph: Markdown headings → nodes, cross-references → edges
- extract_paragraph_graph: paragraph blocks → nodes, sequential edges
"""

import re
from typing import Any, Dict, List, Optional

from ..types import GraphDict, make_edge, make_node


# ---------------------------------------------------------------------------
# Markdown heading extractor
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")


def extract_section_graph(text: str, source: str = "") -> GraphDict:
    """
    Build a graph from a Markdown document.

    Each heading (H1–H6) becomes a node. The content between two consecutive
    headings is attached to the first heading's node. Cross-references
    (mentions of other headings within the body) become edges.

    Parameters
    ----------
    text : str
        Raw Markdown text.
    source : str
        Source file path or URL — stored as a node attribute.

    Returns
    -------
    GraphDict
        ``{"nodes": [...], "edges": [...]}``.
    """
    headings = list(_HEADING_RE.finditer(text))
    if not headings:
        # No headings — treat entire text as one node
        node_id = _make_id(source or "document", 0)
        return {
            "nodes": [make_node(node_id, source or "Document", content=text.strip())],
            "edges": [],
        }

    nodes: List[Dict[str, Any]] = []
    label_to_id: Dict[str, str] = {}

    for idx, match in enumerate(headings):
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        node_id = _make_id(heading_text, idx)

        # Content: text between this heading and the next
        body_start = match.end()
        body_end = headings[idx + 1].start() if idx + 1 < len(headings) else len(text)
        content = text[body_start:body_end].strip()

        attrs: Dict[str, Any] = {"level": level}
        if source:
            attrs["source"] = source

        nodes.append(make_node(node_id, heading_text, content=content, attributes=attrs))
        label_to_id[heading_text.lower()] = node_id

    # Detect cross-references: heading B mentioned in heading A's content
    edges: List[Dict[str, Any]] = []
    seen_edges = set()

    for node in nodes:
        content = node.get("content") or ""
        from_id = node["id"]
        for target_label, to_id in label_to_id.items():
            if to_id == from_id:
                continue
            if target_label in content.lower():
                edge_key = (from_id, to_id)
                if edge_key not in seen_edges:
                    edges.append(make_edge(from_id, to_id, "references"))
                    seen_edges.add(edge_key)

    # Also add parent → child containment edges based on heading level
    edges.extend(_build_containment_edges(nodes, headings))

    return {"nodes": nodes, "edges": edges}


def _build_containment_edges(
    nodes: List[Dict[str, Any]],
    headings: List[re.Match],
) -> List[Dict[str, Any]]:
    """Add 'contains' edges from parent heading to child headings."""
    edges: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = []  # (level, node)

    for node, match in zip(nodes, headings):
        level = node.get("attributes", {}).get("level", 1)
        # Pop stack until we find a parent with smaller level
        while stack and stack[-1].get("attributes", {}).get("level", 0) >= level:
            stack.pop()
        if stack:
            edges.append(make_edge(stack[-1]["id"], node["id"], "contains"))
        stack.append(node)

    return edges


# ---------------------------------------------------------------------------
# Paragraph extractor (for plain text / non-Markdown docs)
# ---------------------------------------------------------------------------

def extract_paragraph_graph(text: str, source: str = "") -> GraphDict:
    """
    Build a graph from plain text by splitting into paragraphs.

    Each non-empty paragraph becomes a node. Sequential edges connect
    consecutive paragraphs.

    Parameters
    ----------
    text : str
        Plain text content.
    source : str
        Document title or file path.

    Returns
    -------
    GraphDict
        ``{"nodes": [...], "edges": [...]}``.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return {"nodes": [], "edges": []}

    nodes: List[Dict[str, Any]] = []
    for idx, para in enumerate(paragraphs):
        # Use first sentence (up to 80 chars) as label
        first_sentence = re.split(r"[.!?]", para)[0].strip()[:80]
        label = first_sentence or f"Paragraph {idx + 1}"
        node_id = _make_id(source, idx)
        attrs: Dict[str, Any] = {"index": idx}
        if source:
            attrs["source"] = source
        nodes.append(make_node(node_id, label, content=para, attributes=attrs))

    edges: List[Dict[str, Any]] = [
        make_edge(nodes[i]["id"], nodes[i + 1]["id"], "followed_by")
        for i in range(len(nodes) - 1)
    ]

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(label: str, idx: int) -> str:
    """Generate a stable node ID from a label and index."""
    slug = re.sub(r"\W+", "_", label.lower()).strip("_")[:40]
    return f"{slug}_{idx}"
