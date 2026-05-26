"""Tests for section extractor."""

import pytest
from docs2graph.extractors.section import extract_section_graph, extract_paragraph_graph


SAMPLE_MARKDOWN = """# Introduction

This paper introduces graph-based document ranking.
We compare our approach with RAG baselines.

## Background

Traditional RAG uses vector search over chunks.
This loses document structure.

## Methodology

We apply Personalized PageRank over a section graph.
See also: Introduction for motivation.

## Results

Our method achieves 80% recall on HotpotQA at k=5.
"""


def test_extract_section_graph_node_count():
    g = extract_section_graph(SAMPLE_MARKDOWN)
    # H1 + 3 H2 = 4 sections
    assert len(g["nodes"]) == 4


def test_extract_section_graph_labels():
    g = extract_section_graph(SAMPLE_MARKDOWN)
    labels = {n["label"] for n in g["nodes"]}
    assert "Introduction" in labels
    assert "Methodology" in labels


def test_extract_section_graph_content_attached():
    g = extract_section_graph(SAMPLE_MARKDOWN)
    intro = next(n for n in g["nodes"] if n["label"] == "Introduction")
    assert "graph-based" in (intro.get("content") or "")


def test_extract_section_graph_has_edges():
    g = extract_section_graph(SAMPLE_MARKDOWN)
    assert len(g["edges"]) > 0


def test_extract_section_graph_cross_reference_edge():
    g = extract_section_graph(SAMPLE_MARKDOWN)
    # "Methodology" mentions "Introduction" → should have a references edge
    edge_pairs = {(e["from"].split("_")[0], e["label"]) for e in g["edges"]}
    # At least one "references" edge should exist
    assert any(label == "references" for _, label in edge_pairs)


def test_extract_paragraph_graph():
    text = "First paragraph about graphs.\n\nSecond paragraph about ranking.\n\nThird paragraph about results."
    g = extract_paragraph_graph(text)
    assert len(g["nodes"]) == 3
    assert len(g["edges"]) == 2  # sequential edges


def test_extract_paragraph_graph_empty():
    g = extract_paragraph_graph("")
    assert g["nodes"] == []
    assert g["edges"] == []


def test_no_headings_fallback():
    text = "Just some plain text without any headings."
    g = extract_section_graph(text, source="test.md")
    assert len(g["nodes"]) == 1
    assert g["nodes"][0]["content"] == text.strip()
