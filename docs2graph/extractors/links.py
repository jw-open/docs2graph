"""
Cross-document link detection.

Scans each document's content for mentions of other documents' titles
and adds "mentions" edges between them. This gives PPR the graph structure
it needs for multi-hop reasoning — without it, PPR degrades to pure
token matching.

Example
-------
If the "Python" article contains "Guido van Rossum":
  → edge: Python --[mentions]--> Guido van Rossum

This is the key mechanism that lets PPR propagate relevance through
connected documents for multi-hop questions.
"""

import re
from typing import Any, Dict, List, Set, Tuple

from ..types import GraphDict, make_edge


def detect_cross_doc_links(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Scan each node's content for mentions of other nodes' labels.

    Returns a list of "mentions" edges where node A's content contains
    node B's label (case-insensitive, whole-word match).

    Parameters
    ----------
    nodes : list
        List of node dicts with "id", "label", and "content".

    Returns
    -------
    list
        List of edge dicts ``{"from": ..., "to": ..., "label": "mentions"}``.
    """
    edges: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()

    # Pre-compile a pattern per node label for efficiency
    label_patterns: List[Tuple[str, str, re.Pattern]] = []
    for node in nodes:
        label = node["label"]
        if not label:
            continue
        # Escape special chars; match whole words (or multi-word phrases)
        escaped = re.escape(label)
        pattern = re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)
        label_patterns.append((node["id"], label, pattern))

    for source_node in nodes:
        content = source_node.get("content") or ""
        if not content:
            continue
        from_id = source_node["id"]

        for to_id, label, pattern in label_patterns:
            if to_id == from_id:
                continue
            edge_key = (from_id, to_id)
            if edge_key in seen:
                continue
            if pattern.search(content):
                edges.append(make_edge(from_id, to_id, "mentions"))
                seen.add(edge_key)

    return edges


def enrich_graph_with_links(graph: GraphDict) -> GraphDict:
    """
    Add cross-document mention edges to an existing graph.

    Non-destructive: appends to existing edges, does not remove any.

    Parameters
    ----------
    graph : GraphDict
        ``{"nodes": [...], "edges": [...]}``

    Returns
    -------
    GraphDict
        The same graph with additional "mentions" edges.
    """
    new_edges = detect_cross_doc_links(graph.get("nodes", []))
    return {
        "nodes": graph["nodes"],
        "edges": graph.get("edges", []) + new_edges,
    }
