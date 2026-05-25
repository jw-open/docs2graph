"""Decision graph extraction for architecture and design documents."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from ..types import GraphDict, make_edge, make_node

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+)$", re.MULTILINE)

_SECTION_TYPES = {
    "problem": ("problem", "challenge", "motivation", "issue"),
    "context": ("context", "background", "goals", "requirements", "constraints"),
    "option": ("option", "alternative", "approach", "solution"),
    "pros": ("pros", "benefits", "advantages"),
    "cons": ("cons", "drawbacks", "disadvantages"),
    "tradeoff": ("tradeoff", "trade-off", "consideration"),
    "decision": ("decision", "selected", "chosen", "resolution"),
    "consequence": ("consequence", "impact", "result", "risks"),
}


def extract_decision_graph(text: str, source: str = "") -> GraphDict:
    """
    Extract documented architecture/design decisions as a graph.

    Static extraction finds explicit decision evidence in ADR/RFC/design docs.
    Nodes are labelled by role: problem, context, option, pros, cons,
    tradeoff, decision, and consequence. LLM-inferred reasoning should be
    added later as separate nodes with ``extraction_method=llm_inferred``.
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seen_nodes = set()
    seen_edges = set()

    root_id = _node_id("decision_document", source or "document")
    _add_node(nodes, seen_nodes, root_id, source or "Decision Document", text[:4000], {
        "type": "decision_document",
        "source": source,
        "extraction_method": "static",
    })

    sections = _split_sections(text)
    last_problem_id = None
    current_option_id = None

    for index, section in enumerate(sections):
        kind = _classify(section["title"], section["content"])
        section_id = _node_id(kind, f"{index}-{section['title']}")
        _add_node(nodes, seen_nodes, section_id, section["title"], section["content"], {
            "type": kind,
            "level": section["level"],
            "source": source,
            "extraction_method": "static",
        })
        _add_edge(edges, seen_edges, root_id, section_id, "contains")

        if kind == "problem":
            last_problem_id = section_id
        elif kind == "option":
            current_option_id = section_id
            if last_problem_id:
                _add_edge(edges, seen_edges, last_problem_id, section_id, "has_option")
        elif kind in {"pros", "cons", "tradeoff"} and current_option_id:
            _add_edge(edges, seen_edges, current_option_id, section_id, kind)
        elif kind == "decision":
            if last_problem_id:
                _add_edge(edges, seen_edges, last_problem_id, section_id, "resolved_by")
            if current_option_id:
                _add_edge(edges, seen_edges, section_id, current_option_id, "selects")
        elif kind == "consequence" and current_option_id:
            _add_edge(edges, seen_edges, current_option_id, section_id, "has_consequence")

        for bullet_index, bullet in enumerate(_BULLET_RE.findall(section["content"])):
            bullet_kind = _classify_bullet(bullet, default=kind)
            bullet_id = _node_id(bullet_kind, f"{index}-{bullet_index}-{bullet}")
            _add_node(nodes, seen_nodes, bullet_id, _label(bullet), bullet, {
                "type": bullet_kind,
                "source": source,
                "section": section["title"],
                "extraction_method": "static",
            })
            _add_edge(edges, seen_edges, section_id, bullet_id, "contains")
            if bullet_kind in {"pros", "cons", "tradeoff"} and current_option_id:
                _add_edge(edges, seen_edges, current_option_id, bullet_id, bullet_kind)

    return {"nodes": nodes, "edges": edges, "current_node_id": root_id}


def _split_sections(text: str) -> List[Dict[str, Any]]:
    headings = list(_HEADING_RE.finditer(text))
    if not headings:
        return [{"title": "Decision", "level": 1, "content": text.strip()}]

    sections: List[Dict[str, Any]] = []
    for index, match in enumerate(headings):
        start = match.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        sections.append({
            "title": match.group(2).strip(),
            "level": len(match.group(1)),
            "content": text[start:end].strip(),
        })
    return sections


def _classify(title: str, content: str) -> str:
    title_lower = title.lower()
    for kind, cues in _SECTION_TYPES.items():
        if any(cue in title_lower for cue in cues):
            return kind

    text = f"{title}\n{content[:300]}".lower()
    for kind, cues in _SECTION_TYPES.items():
        if any(cue in text for cue in cues):
            return kind
    return "decision_context"


def _classify_bullet(text: str, default: str) -> str:
    lower = text.lower()
    if lower.startswith(("pro:", "benefit:", "advantage:")):
        return "pros"
    if lower.startswith(("con:", "drawback:", "risk:", "cost:")):
        return "cons"
    if "tradeoff" in lower or "trade-off" in lower or "but " in lower:
        return "tradeoff"
    if lower.startswith(("decide", "decision:", "choose", "chosen")):
        return "decision"
    return default


def _label(text: str, limit: int = 96) -> str:
    return text.strip().replace("\n", " ")[:limit].rstrip()


def _node_id(prefix: str, value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")[:80]
    return f"{prefix}:{slug or 'item'}"


def _add_node(
    nodes: List[Dict[str, Any]],
    seen: set,
    node_id: str,
    label: str,
    content: str | None = None,
    attributes: Dict[str, Any] | None = None,
) -> None:
    if node_id in seen:
        return
    nodes.append(make_node(node_id, label, content=content, attributes=attributes))
    seen.add(node_id)


def _add_edge(edges: List[Dict[str, Any]], seen: set, from_id: str, to_id: str, label: str) -> None:
    key = (from_id, to_id, label)
    if key in seen:
        return
    edges.append(make_edge(from_id, to_id, label))
    seen.add(key)
