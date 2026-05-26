"""
Node and edge type definitions for doc2graph.

Node shape
----------
{
    "id":         str,          # unique identifier
    "label":      str,          # title, heading, or name — used for query matching
    "content":    str | None,   # full text body of the section/document
    "attributes": dict | None,  # type, source, page, url, ...
}

Edge shape
----------
{
    "from":  str,   # source node id
    "to":    str,   # target node id
    "label": str,   # relationship type: "references", "contains", "links_to", ...
}
"""

from typing import Any, Dict, List, Optional, TypedDict


class NodeDict(TypedDict, total=False):
    id: str
    label: str
    content: Optional[str]
    attributes: Optional[Dict[str, Any]]


class EdgeDict(TypedDict):
    id_from: str  # stored as "from" in dicts — use make_edge() helper
    id_to: str
    label: str


GraphDict = Dict[str, List[Any]]  # {"nodes": [...], "edges": [...]}


def make_node(
    id: str,
    label: str,
    content: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a node dict."""
    node: Dict[str, Any] = {"id": id, "label": label}
    if content is not None:
        node["content"] = content
    if attributes:
        node["attributes"] = attributes
    return node


def make_edge(from_id: str, to_id: str, label: str) -> Dict[str, Any]:
    """Build an edge dict."""
    return {"from": from_id, "to": to_id, "label": label}
