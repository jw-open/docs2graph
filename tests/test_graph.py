"""Tests for DocumentGraph class."""

import pytest
from docs2graph import DocumentGraph


@pytest.fixture
def simple_graph():
    g = DocumentGraph()
    g.add_node("intro", "Introduction", content="This paper introduces a graph-based approach.")
    g.add_node("method", "Methodology", content="We use Personalized PageRank to rank nodes.")
    g.add_node("results", "Results", content="Our approach achieves 80% recall on HotpotQA.")
    g.add_node("conclusion", "Conclusion", content="Graph-based ranking reduces tokens needed.")
    g.add_edge("intro", "method", "references")
    g.add_edge("method", "results", "references")
    g.add_edge("results", "conclusion", "references")
    return g


def test_repr(simple_graph):
    assert "DocumentGraph" in repr(simple_graph)
    assert "4" in repr(simple_graph)


def test_len(simple_graph):
    assert len(simple_graph) == 4


def test_to_dict(simple_graph):
    d = simple_graph.to_dict()
    assert "nodes" in d
    assert "edges" in d
    assert len(d["nodes"]) == 4
    assert len(d["edges"]) == 3


def test_from_dict_roundtrip(simple_graph):
    d = simple_graph.to_dict()
    g2 = DocumentGraph.from_dict(d)
    assert len(g2) == 4
    assert g2.to_dict() == d


def test_rank_returns_expected_keys(simple_graph):
    result = simple_graph.rank("how does ranking work?")
    assert "nodes" in result
    assert "edges" in result


def test_rank_returns_nodes(simple_graph):
    result = simple_graph.rank("PageRank methodology", k=2)
    assert len(result["nodes"]) > 0


def test_top_k_nodes_have_score(simple_graph):
    result = simple_graph.rank("methodology ranking", k=2)
    scored = [n for n in result["nodes"] if "score" in n]
    assert len(scored) <= 2


def test_duplicate_node_raises(simple_graph):
    with pytest.raises(ValueError):
        simple_graph.add_node("intro", "Duplicate")


def test_method_chaining():
    g = (
        DocumentGraph()
        .add_node("a", "Alpha", content="first node")
        .add_node("b", "Beta", content="second node")
        .add_edge("a", "b", "references")
    )
    assert len(g) == 2


def test_from_texts():
    g = DocumentGraph.from_texts([
        {"title": "Python", "content": "Python is a programming language created by Guido."},
        {"title": "Guido van Rossum", "content": "Guido created Python in the late 1980s."},
    ])
    assert len(g) == 2
    result = g.rank("who created Python?", k=2)
    labels = {n["label"] for n in result["nodes"]}
    assert "Guido van Rossum" in labels or "Python" in labels


def test_empty_query_returns_empty(simple_graph):
    result = simple_graph.rank("")
    assert result["nodes"] == []
    assert result["edges"] == []
