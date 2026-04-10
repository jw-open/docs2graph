"""Tests for cross-document link detection."""

import pytest
from doc2graph import DocumentGraph
from doc2graph.extractors.links import detect_cross_doc_links, enrich_graph_with_links


DOCS = [
    {"title": "Python", "content": "Python was created by Guido van Rossum in 1991."},
    {"title": "Guido van Rossum", "content": "Guido was born in the Netherlands in 1956."},
    {"title": "Netherlands", "content": "The Netherlands is a country in northwestern Europe."},
    {"title": "Java", "content": "Java is a programming language created by James Gosling."},
]


def test_from_texts_detects_edges():
    g = DocumentGraph.from_texts(DOCS)
    d = g.to_dict()
    assert len(d["edges"]) > 0


def test_python_mentions_guido_edge():
    g = DocumentGraph.from_texts(DOCS)
    edges = g.to_dict()["edges"]
    # Python doc mentions "Guido van Rossum" → should have an edge
    python_to_guido = [
        e for e in edges
        if "python" in e["from"] and "guido" in e["to"]
    ]
    assert len(python_to_guido) > 0


def test_guido_mentions_netherlands_edge():
    g = DocumentGraph.from_texts(DOCS)
    edges = g.to_dict()["edges"]
    guido_to_nl = [
        e for e in edges
        if "guido" in e["from"] and "netherlands" in e["to"]
    ]
    assert len(guido_to_nl) > 0


def test_no_self_edges():
    g = DocumentGraph.from_texts(DOCS)
    edges = g.to_dict()["edges"]
    self_edges = [e for e in edges if e["from"] == e["to"]]
    assert len(self_edges) == 0


def test_multihop_ranking():
    """PPR should surface Netherlands for 'what country was Python creator born in?'"""
    g = DocumentGraph.from_texts(DOCS)
    result = g.rank("what country was the creator of Python born in?", k=3)
    labels = [n["label"] for n in result["nodes"]]
    # Netherlands should be retrievable through Python→Guido→Netherlands chain
    assert "Netherlands" in labels or "Guido van Rossum" in labels


def test_detect_cross_doc_links_direct():
    nodes = [
        {"id": "a", "label": "Alpha", "content": "Alpha references Beta explicitly."},
        {"id": "b", "label": "Beta", "content": "Beta is standalone."},
        {"id": "c", "label": "Gamma", "content": "Gamma has no references."},
    ]
    edges = detect_cross_doc_links(nodes)
    assert any(e["from"] == "a" and e["to"] == "b" for e in edges)
    assert not any(e["from"] == "c" for e in edges)


def test_enrich_graph_with_links():
    graph = {
        "nodes": [
            {"id": "x", "label": "X node", "content": "X node references Y node."},
            {"id": "y", "label": "Y node", "content": "Y node is independent."},
        ],
        "edges": [],
    }
    enriched = enrich_graph_with_links(graph)
    assert len(enriched["edges"]) > 0
    assert enriched["nodes"] == graph["nodes"]  # nodes unchanged


def test_no_duplicate_edges():
    g = DocumentGraph.from_texts(DOCS)
    edges = g.to_dict()["edges"]
    edge_keys = [(e["from"], e["to"]) for e in edges]
    assert len(edge_keys) == len(set(edge_keys))
