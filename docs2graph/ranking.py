"""
Personalized PageRank (PPR) for doc2graph.

Same algorithm as graph2sql — scores nodes in a document knowledge graph
against a natural language query and returns the most relevant subgraph.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from .matching import soft_match_score
from .types import GraphDict


def personalized_page_rank(
    query: str,
    graph: GraphDict,
    alpha: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 50,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Compute PPR over a document knowledge graph and return the relevant subgraph.

    Parameters
    ----------
    query : str
        Natural language question.
    graph : GraphDict
        ``{"nodes": [...], "edges": [...]}``.
    alpha : float
        Damping factor (default 0.85).
    tol : float
        Convergence tolerance (default 1e-6).
    max_iter : int
        Max power-iteration steps (default 50).
    k : int
        Number of top-ranked seed nodes (default 5 — larger than graph2sql
        since document graphs tend to be bigger).

    Returns
    -------
    dict
        ``{"nodes": [...], "edges": [...]}`` — top-k nodes + 1-hop neighbours.
        Top-k nodes carry a ``"score"`` field.
        Returns empty lists if no query tokens match any node.
    """
    nodes: List[Dict[str, Any]] = graph.get("nodes", [])
    edges: List[Dict[str, Any]] = graph.get("edges", [])
    n = len(nodes)

    if n == 0:
        return {"nodes": [], "edges": []}

    node_id_to_index: Dict[str, int] = {node["id"]: i for i, node in enumerate(nodes)}

    # Build column-stochastic transition matrix.
    M = np.zeros((n, n))
    outlinks: Dict[int, List[int]] = {i: [] for i in range(n)}
    for edge in edges:
        src, tgt = edge.get("from"), edge.get("to")
        if src in node_id_to_index and tgt in node_id_to_index:
            j, i = node_id_to_index[src], node_id_to_index[tgt]
            outlinks[j].append(i)

    for j in range(n):
        if outlinks[j]:
            prob = 1.0 / len(outlinks[j])
            for i in outlinks[j]:
                M[i, j] = prob
        else:
            M[:, j] = 1.0 / n  # dangling node

    # Personalization vector via soft matching.
    p = np.array([
        soft_match_score(
            query=query,
            label=node["label"],
            content=node.get("content") or "",
            attributes=node.get("attributes"),
        )
        for node in nodes
    ])

    if p.sum() == 0:
        return {"nodes": [], "edges": []}
    p = p / p.sum()

    # Power iteration.
    r = np.ones(n) / n
    for _ in range(max_iter):
        r_new = (1 - alpha) * p + alpha * M.dot(r)
        if np.linalg.norm(r_new - r, 1) < tol:
            r = r_new
            break
        r = r_new

    # Top-k by PPR score.
    scored = sorted(
        ((r[node_id_to_index[node["id"]]], node["id"]) for node in nodes),
        key=lambda t: t[0],
        reverse=True,
    )
    top_ids = {nid for _, nid in scored[:k]}
    top_scores = {nid: float(sc) for sc, nid in scored[:k]}

    top_node_objs = [node for node in nodes if node["id"] in top_ids]
    extended_nodes, extended_edges = _extend_subgraph(top_node_objs, graph)

    id_to_label = {node["id"]: node["label"] for node in extended_nodes}

    formatted_nodes = []
    for node in extended_nodes:
        nd: Dict[str, Any] = {
            "label": node["label"],
            "content": node.get("content"),
            "attributes": node.get("attributes", {}),
        }
        if node["id"] in top_scores:
            nd["score"] = top_scores[node["id"]]
        formatted_nodes.append(nd)

    formatted_edges = [
        {"label": e.get("label", ""), "from": id_to_label[e["from"]], "to": id_to_label[e["to"]]}
        for e in extended_edges
        if e.get("from") in id_to_label and e.get("to") in id_to_label
    ]

    return {"nodes": formatted_nodes, "edges": formatted_edges}


def _extend_subgraph(
    top_nodes: List[Dict[str, Any]],
    graph: GraphDict,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extend top-k nodes with 1-hop neighbours and connecting edges."""
    top_ids = {node["id"] for node in top_nodes}
    extended_ids = set(top_ids)

    for edge in graph.get("edges", []):
        if edge.get("from") in top_ids or edge.get("to") in top_ids:
            extended_ids.add(edge.get("from"))
            extended_ids.add(edge.get("to"))

    extended_nodes = [n for n in graph.get("nodes", []) if n["id"] in extended_ids]
    extended_edges = [
        e for e in graph.get("edges", [])
        if e.get("from") in extended_ids and e.get("to") in extended_ids
    ]
    return extended_nodes, extended_edges
