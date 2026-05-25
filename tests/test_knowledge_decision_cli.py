import json

from doc2graph import DocumentGraph, extract_decision_graph, extract_knowledge_graph
from doc2graph.cli import build_graph, main


PAPER = """# Abstract

This paper proposes a graph based approach for document reasoning.
We show that graph ranked context reduces token usage by 40% [1].

# Method

The method builds a knowledge graph from document sections and citations.
See https://example.com/paper for details.

# Results

Evaluation results show improved recall on multi hop questions.
"""


DECISION = """# Problem

The system needs to create graphs from documentation without requiring cloud calls.

# Option A: Static extraction

- Pro: deterministic and cheap
- Con: misses implied tradeoffs

# Option B: LLM enrichment

- Pro: summarizes ambiguous design reasoning
- Con: costs money and may leak private docs

# Decision

Choose static extraction by default and keep LLM enrichment optional.

# Consequences

Users can run locally, but inferred reasoning should be labeled separately.
"""


def test_extract_knowledge_graph_for_paper_signals():
    graph = extract_knowledge_graph(PAPER, source="paper.md")
    node_types = {n.get("attributes", {}).get("type") for n in graph["nodes"]}
    labels = {e["label"] for e in graph["edges"]}

    assert "section" in node_types
    assert "claim" in node_types
    assert "evidence" in node_types
    assert "citation" in node_types
    assert "url" in node_types
    assert "supported_by" in labels
    assert graph["current_node_id"].startswith("document:")


def test_extract_decision_graph_roles_and_edges():
    graph = extract_decision_graph(DECISION, source="adr.md")
    node_types = {n.get("attributes", {}).get("type") for n in graph["nodes"]}
    labels = {e["label"] for e in graph["edges"]}

    assert "problem" in node_types
    assert "option" in node_types
    assert "pros" in node_types
    assert "cons" in node_types
    assert "decision" in node_types
    assert "has_option" in labels
    assert "resolved_by" in labels


def test_build_graph_and_document_graph_from_document(tmp_path):
    path = tmp_path / "adr.md"
    path.write_text(DECISION, encoding="utf-8")

    raw = build_graph(str(path), graph_type="decision")
    graph = DocumentGraph.from_document(str(path), graph_type="decision")

    assert raw["nodes"]
    assert len(graph) == len(raw["nodes"])


def test_cli_writes_graph_json(tmp_path):
    source = tmp_path / "paper.md"
    output = tmp_path / "graph.json"
    source.write_text(PAPER, encoding="utf-8")

    rc = main([str(source), "--graph", "knowledge", "--output", str(output), "--pretty"])
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["nodes"]
    assert payload["edges"]
