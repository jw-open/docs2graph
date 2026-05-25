import hashlib
import json
import os
import time

import pytest

from doc2graph import DocumentGraph, extract_decision_graph, extract_knowledge_graph
from doc2graph.cli import build_graph, main
from doc2graph.corpus import build_corpus_graph


PAPER = """# Abstract

This paper proposes a graph based approach for document reasoning.
We show that graph ranked context reduces token usage by 40% [1].

# Method

The method builds a knowledge graph from document sections and citations.
See https://example.com/paper for details.

# Results

Evaluation results show improved recall on multi hop questions.
"""


REFERENCED_PAPER = """# Abstract

This paper proposes graph grounded context for long documents [1, 2].
Results show recall improves by 18% when citations are preserved [2].

# References

[1] Smith, A. Graph Context for Retrieval. 2024.
[2] Lee, B. Evidence-Aware PageRank. 2025.
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


BULLET_DECISION = """# ADR: Cache repeated corpus extraction

- Problem: repeated scans of large folders waste time.
- Option A: always rebuild every file graph.
- Option B: reuse unchanged per-file graph entries.
- Pro: faster repeated local runs.
- Con: cache files can become stale if metadata is incomplete.
- Tradeoff: cache reads add complexity but keep default output deterministic.
- Decision: choose Option B for explicit cache paths only.
- Consequence: users must opt in with a cache path.
- Confidence: medium, based on deterministic file metadata.
"""


TABLE_DECISION = """# Problem

The team needs a repeatable ingestion strategy.

# Options

| Option | Pros | Cons | Tradeoff | Decision | Confidence |
| --- | --- | --- | --- | --- | --- |
| Batch rebuild | simple to reason about | slow on large corpora | no cache invalidation risk | Rejected | Low |
| Content cache | fast repeated runs | cache metadata must be correct | extra cache file management | Selected | High |
"""


def _skip_records_sha256(records):
    digest = hashlib.sha256()
    for reason, path_type, relative_path in records:
        for value in (reason, path_type, relative_path):
            digest.update(value.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
    return digest.hexdigest()


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


def test_extract_knowledge_graph_resolves_inline_citations_to_references():
    graph = extract_knowledge_graph(REFERENCED_PAPER, source="paper.md")
    node_types = {n.get("attributes", {}).get("type") for n in graph["nodes"]}
    labels = {e["label"] for e in graph["edges"]}
    references = [
        n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "reference"
    ]
    citation_to_reference = [
        e for e in graph["edges"] if e["label"] == "resolves_to"
    ]
    claim_cites = [
        e
        for e in graph["edges"]
        if e["label"] == "cites"
        and e["from"].startswith("claim:")
        and e["to"].startswith("citation:")
    ]
    evidence_cites = [
        e
        for e in graph["edges"]
        if e["label"] == "cites"
        and e["from"].startswith("evidence:")
        and e["to"].startswith("citation:")
    ]

    assert "reference" in node_types
    assert "resolves_to" in labels
    assert {n["attributes"]["key"] for n in references} == {"1", "2"}
    assert len(citation_to_reference) == 2
    assert len(claim_cites) == 2
    assert len(evidence_cites) == 1


def test_extract_knowledge_graph_adds_definition_provenance():
    text = """# Glossary

- Personalized PageRank: a graph ranking algorithm that biases traversal toward query relevant seed nodes [1].

# Architecture

Context engineering is the process of selecting structured context for a model.

# References

[1] Page, L. The PageRank Citation Ranking. 1998.
"""
    graph = extract_knowledge_graph(text, source="glossary.md", max_concepts=0)
    nodes = graph["nodes"]
    node_types = {n.get("attributes", {}).get("type") for n in nodes}
    labels = {e["label"] for e in graph["edges"]}
    definition = next(
        n
        for n in nodes
        if n.get("attributes", {}).get("type") == "definition"
        and n["attributes"]["normalized_term"] == "personalized pagerank"
    )
    concept = next(
        n
        for n in nodes
        if n.get("attributes", {}).get("type") == "concept"
        and n["label"] == "personalized pagerank"
    )
    citation = next(
        n
        for n in nodes
        if n.get("attributes", {}).get("type") == "citation"
        and n["label"] == "1"
    )

    assert "definition" in node_types
    assert {"defines", "defined_by", "resolves_to"} <= labels
    assert any(e["from"] == definition["id"] and e["to"] == concept["id"] and e["label"] == "defines" for e in graph["edges"])
    assert any(e["from"] == concept["id"] and e["to"] == definition["id"] and e["label"] == "defined_by" for e in graph["edges"])
    assert any(e["from"] == definition["id"] and e["to"] == citation["id"] and e["label"] == "cites" for e in graph["edges"])


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


def test_extract_decision_graph_from_bullet_only_adr():
    graph = extract_decision_graph(BULLET_DECISION, source="cache-adr.md")
    node_types = {n.get("attributes", {}).get("type") for n in graph["nodes"]}
    labels = {e["label"] for e in graph["edges"]}

    assert {"problem", "option", "pros", "cons", "tradeoff", "decision", "consequence", "confidence"} <= node_types
    assert {"has_option", "resolved_by", "selects", "has_consequence", "has_confidence"} <= labels


def test_decision_selects_named_option_not_last_option():
    graph = extract_decision_graph(DECISION, source="adr.md")
    option_a = next(n for n in graph["nodes"] if n["label"] == "Option A: Static extraction")
    option_b = next(n for n in graph["nodes"] if n["label"] == "Option B: LLM enrichment")
    select_edges = [e for e in graph["edges"] if e["label"] == "selects"]

    assert any(e["to"] == option_a["id"] for e in select_edges)
    assert all(e["to"] != option_b["id"] for e in select_edges)


def test_decision_consequence_links_to_selected_option_not_last_option():
    graph = extract_decision_graph(DECISION, source="adr.md")
    option_a = next(n for n in graph["nodes"] if n["label"] == "Option A: Static extraction")
    option_b = next(n for n in graph["nodes"] if n["label"] == "Option B: LLM enrichment")
    decision = next(
        n
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "decision"
        and n["label"] == "Decision"
    )
    consequence = next(n for n in graph["nodes"] if n["label"] == "Consequences")
    consequence_edges = [e for e in graph["edges"] if e["label"] == "has_consequence"]

    assert any(e["from"] == decision["id"] and e["to"] == consequence["id"] for e in consequence_edges)
    assert any(e["from"] == option_a["id"] and e["to"] == consequence["id"] for e in consequence_edges)
    assert all(
        not (e["from"] == option_b["id"] and e["to"] == consequence["id"])
        for e in consequence_edges
    )


def test_bullet_consequence_links_to_selected_option_not_last_option():
    graph = extract_decision_graph(BULLET_DECISION, source="cache-adr.md")
    option_a = next(n for n in graph["nodes"] if n["label"] == "Option A: always rebuild every file graph.")
    option_b = next(n for n in graph["nodes"] if n["label"] == "Option B: reuse unchanged per-file graph entries.")
    decision = next(
        n
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "decision"
        and n["label"] == "Decision: choose Option B for explicit cache paths only."
    )
    consequence = next(
        n
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "consequence"
        and n["label"] == "Consequence: users must opt in with a cache path."
    )
    consequence_edges = [e for e in graph["edges"] if e["label"] == "has_consequence"]

    assert any(e["from"] == decision["id"] and e["to"] == consequence["id"] for e in consequence_edges)
    assert any(e["from"] == option_b["id"] and e["to"] == consequence["id"] for e in consequence_edges)
    assert all(
        not (e["from"] == option_a["id"] and e["to"] == consequence["id"])
        for e in consequence_edges
    )


def test_extract_decision_graph_from_option_table():
    graph = extract_decision_graph(TABLE_DECISION, source="table-adr.md")
    nodes = graph["nodes"]
    node_types = {n.get("attributes", {}).get("type") for n in nodes}
    labels = {e["label"] for e in graph["edges"]}
    content_cache = next(n for n in nodes if n["label"] == "Content cache")
    selected_decision = next(
        n
        for n in nodes
        if n.get("attributes", {}).get("type") == "decision"
        and n.get("content") == "Selected"
    )
    high_confidence = next(
        n
        for n in nodes
        if n.get("attributes", {}).get("type") == "confidence"
        and n.get("content") == "High"
    )

    assert {"option", "pros", "cons", "tradeoff", "decision", "confidence"} <= node_types
    assert {"has_option", "pros", "cons", "tradeoff", "selects", "has_confidence"} <= labels
    assert any(
        e["from"] == selected_decision["id"] and e["to"] == content_cache["id"] and e["label"] == "selects"
        for e in graph["edges"]
    )
    assert any(
        e["from"] == selected_decision["id"] and e["to"] == high_confidence["id"] and e["label"] == "has_confidence"
        for e in graph["edges"]
    )
    assert high_confidence["attributes"]["confidence_value"] == "High"


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


def test_build_graph_from_directory_corpus(tmp_path):
    docs = tmp_path / "docs"
    nested = docs / "architecture"
    nested.mkdir(parents=True)
    (docs / "paper.md").write_text(PAPER, encoding="utf-8")
    (nested / "adr.md").write_text(DECISION, encoding="utf-8")
    (docs / "ignored.bin").write_bytes(b"\x00\x01")

    graph = build_graph(str(docs), graph_type="all")
    node_types = {n.get("attributes", {}).get("type") for n in graph["nodes"]}
    labels = {e["label"] for e in graph["edges"]}

    assert graph["current_node_id"].startswith("corpus:")
    assert "corpus" in node_types
    assert "folder" in node_types
    assert "file" in node_types
    assert "claim" in node_types
    assert "decision" in node_types
    assert "extracted_as" in labels


def test_directory_corpus_preserves_same_heading_sections_per_source(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# Summary\n\nThis document describes Alpha context.", encoding="utf-8")
    (docs / "b.md").write_text("# Summary\n\nThis document describes Beta context.", encoding="utf-8")

    graph = build_corpus_graph(str(docs), graph_type="knowledge")
    summary_sections = [
        n
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "section"
        and n.get("label") == "Summary"
    ]
    sources = {n["attributes"]["source"] for n in summary_sections}

    assert len(summary_sections) == 2
    assert sources == {str(docs / "a.md"), str(docs / "b.md")}
    assert all(n["attributes"]["document_id"].startswith("document:") for n in summary_sections)


def test_directory_corpus_preserves_same_decision_sections_per_source(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# Decision\n\nChoose static extraction.", encoding="utf-8")
    (docs / "b.md").write_text("# Decision\n\nChoose explicit cache paths.", encoding="utf-8")

    graph = build_corpus_graph(str(docs), graph_type="decision")
    decision_sections = [
        n
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "decision"
        and n.get("label") == "Decision"
    ]
    sources = {n["attributes"]["source"] for n in decision_sections}

    assert len(decision_sections) == 2
    assert sources == {str(docs / "a.md"), str(docs / "b.md")}
    assert all(n["attributes"]["document_id"].startswith("decision_document:") for n in decision_sections)


def test_directory_corpus_adds_cross_document_links_for_named_sections(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    alpha = docs / "alpha.md"
    beta = docs / "beta.md"
    alpha.write_text(
        "# Alpha Plan\n\nThis document should follow the Beta Strategy before launch.",
        encoding="utf-8",
    )
    beta.write_text("# Beta Strategy\n\nThis document describes release sequencing.", encoding="utf-8")

    graph = build_corpus_graph(str(docs), graph_type="knowledge")
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    beta_section = next(
        n
        for n in graph["nodes"]
        if n.get("label") == "Beta Strategy"
        and n.get("attributes", {}).get("source") == str(beta)
    )
    cross_links = [
        e
        for e in graph["edges"]
        if e["label"] == "mentions"
        and e["to"] == beta_section["id"]
        and any(
            n["id"] == e["from"] and n.get("attributes", {}).get("source") == str(alpha)
            for n in graph["nodes"]
        )
    ]

    assert manifest["attributes"]["cross_document_link_count"] >= 1
    assert cross_links


def test_directory_corpus_avoids_generic_heading_cross_document_links(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# Summary\n\nThis summary mentions Summary generically.", encoding="utf-8")
    (docs / "b.md").write_text("# Summary\n\nAnother generic summary.", encoding="utf-8")

    graph = build_corpus_graph(str(docs), graph_type="knowledge")
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    summary_ids = {
        n["id"]
        for n in graph["nodes"]
        if n.get("label") == "Summary"
        and n.get("attributes", {}).get("type") == "section"
    }
    generic_cross_links = [
        e
        for e in graph["edges"]
        if e["label"] == "mentions" and e["from"] in summary_ids and e["to"] in summary_ids
    ]

    assert manifest["attributes"]["cross_document_link_count"] == 0
    assert generic_cross_links == []


def test_directory_corpus_skips_large_files(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "small.md").write_text("# Small\n\nThis document describes something.", encoding="utf-8")
    (docs / "large.md").write_text("x" * 100, encoding="utf-8")
    max_file_bytes = (docs / "small.md").stat().st_size

    graph = build_corpus_graph(
        str(docs),
        graph_type="knowledge",
        max_file_bytes=max_file_bytes,
    )
    node_types = {n.get("attributes", {}).get("type") for n in graph["nodes"]}
    file_statuses = {
        n["attributes"]["relative_path"]: n["attributes"]
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "file"
    }

    assert "skipped_file" in node_types
    assert file_statuses["small.md"]["status"] == "extracted"
    assert file_statuses["small.md"]["cache_status"] == "disabled"
    assert file_statuses["large.md"]["status"] == "skipped"
    assert file_statuses["large.md"]["skip_reason"] == "file_too_large"


def test_directory_corpus_reports_skipped_files_and_manifest(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")
    (docs / "b.md").write_text("# B\n\nBeta document.", encoding="utf-8")
    (docs / "notes.bin").write_bytes(b"\x00\x01")

    graph = build_corpus_graph(
        str(docs),
        graph_type="knowledge",
        max_files=1,
        skip_report_limit=10,
    )
    nodes = graph["nodes"]
    manifest = next(n for n in nodes if n.get("attributes", {}).get("type") == "corpus_manifest")
    skipped = [
        n
        for n in nodes
        if n.get("attributes", {}).get("type") == "skipped_file"
        and n.get("attributes", {}).get("reason") in {"max_files_exceeded", "unsupported_extension"}
    ]

    assert manifest["attributes"]["selected_file_count"] == 1
    assert manifest["attributes"]["skipped_by_reason"] == {
        "max_files_exceeded": 1,
        "unsupported_extension": 1,
    }
    assert manifest["attributes"]["max_files_reached"] is True
    assert {n["attributes"]["reason"] for n in skipped} == {"max_files_exceeded", "unsupported_extension"}


def test_directory_corpus_can_stop_scanning_after_max_files(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")
    (docs / "b.md").write_text("# B\n\nBeta document.", encoding="utf-8")
    (docs / "c.md").write_text("# C\n\nGamma document.", encoding="utf-8")

    graph = build_corpus_graph(
        str(docs),
        graph_type="knowledge",
        max_files=1,
        stop_after_max_files=True,
        skip_report_limit=10,
    )
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    file_paths = [
        n["attributes"]["relative_path"]
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "file"
    ]
    skipped = [
        n
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "skipped_file"
        and n.get("attributes", {}).get("reason") == "max_files_exceeded"
    ]

    assert file_paths == ["a.md"]
    assert manifest["attributes"]["selected_file_count"] == 1
    assert manifest["attributes"]["stop_after_max_files"] is True
    assert manifest["attributes"]["max_files_reached"] is True
    assert manifest["attributes"]["max_files_scan_truncated"] is True
    assert manifest["attributes"]["max_scan_entries_reached"] is False
    assert manifest["attributes"]["skipped_file_count_is_complete"] is False
    assert manifest["attributes"]["skipped_file_records_sha256_is_complete"] is False
    assert manifest["attributes"]["selected_file_records_sha256_is_complete"] is False
    assert manifest["attributes"]["skipped_by_reason"] == {"max_files_exceeded": 1}
    assert skipped[0]["attributes"]["relative_path"] == "b.md"


def test_directory_corpus_records_deterministic_selected_path_digest_and_order(tmp_path):
    docs = tmp_path / "docs"
    alpha = docs / "alpha"
    beta = docs / "beta"
    alpha.mkdir(parents=True)
    beta.mkdir()
    (docs / "root.md").write_text("# Root\n\nRoot document.", encoding="utf-8")
    (alpha / "z.md").write_text("# Zed\n\nNested alpha document.", encoding="utf-8")
    (beta / "a.md").write_text("# Aye\n\nNested beta document.", encoding="utf-8")

    graph = build_corpus_graph(str(docs), graph_type="knowledge")
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    file_attrs = [
        n["attributes"]
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "file"
    ]
    selected_paths = [attrs["relative_path"] for attrs in file_attrs]
    digest = hashlib.sha256()
    record_digest = hashlib.sha256()
    selected_total_bytes = 0
    for path in selected_paths:
        digest.update(path.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        file_path = docs / path
        size = file_path.stat().st_size
        selected_total_bytes += size
        for value in (path, file_path.suffix.lower(), str(size)):
            record_digest.update(value.encode("utf-8", errors="surrogateescape"))
            record_digest.update(b"\0")

    assert selected_paths == ["alpha/z.md", "beta/a.md", "root.md"]
    assert [attrs["extraction_order"] for attrs in file_attrs] == [0, 1, 2]
    assert manifest["attributes"]["selected_file_ordering"] == "relative_path_depth_first"
    assert manifest["attributes"]["selected_file_paths_sha256"] == digest.hexdigest()
    assert manifest["attributes"]["selected_file_records_sha256"] == record_digest.hexdigest()
    assert manifest["attributes"]["selected_file_records_sha256_is_complete"] is True
    assert manifest["attributes"]["selected_total_bytes"] == selected_total_bytes
    assert manifest["attributes"]["selected_total_bytes_is_complete"] is True


def test_directory_corpus_reports_supported_files_outside_include(tmp_path):
    docs = tmp_path / "docs"
    adr = docs / "adr"
    notes = docs / "notes"
    adr.mkdir(parents=True)
    notes.mkdir()
    (adr / "accepted.md").write_text("# Accepted\n\nChosen option.", encoding="utf-8")
    (notes / "draft.md").write_text("# Draft\n\nNot part of the ADR graph.", encoding="utf-8")

    graph = build_corpus_graph(
        str(docs),
        graph_type="knowledge",
        include=["adr/**"],
        skip_report_limit=10,
    )
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    file_paths = [
        n["attributes"]["relative_path"]
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "file"
    ]
    skipped = [
        n
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "skipped_file"
        and n.get("attributes", {}).get("reason") == "include_filter_mismatch"
    ]

    assert manifest["attributes"]["include_patterns"] == ["adr/**"]
    assert manifest["attributes"]["exclude_patterns"] == []
    assert manifest["attributes"]["selected_file_count"] == 1
    assert manifest["attributes"]["skipped_by_reason"] == {"include_filter_mismatch": 1}
    assert file_paths == ["adr/accepted.md"]
    assert skipped[0]["attributes"]["relative_path"] == "notes/draft.md"


def test_directory_corpus_include_accepts_folder_name(tmp_path):
    docs = tmp_path / "docs"
    adr = docs / "adr"
    notes = docs / "notes"
    adr.mkdir(parents=True)
    notes.mkdir()
    (adr / "accepted.md").write_text("# Accepted\n\nChosen option.", encoding="utf-8")
    (notes / "draft.md").write_text("# Draft\n\nNot part of the ADR graph.", encoding="utf-8")

    graph = build_corpus_graph(
        str(docs),
        graph_type="knowledge",
        include=["adr"],
        skip_report_limit=10,
    )
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    file_paths = [
        n["attributes"]["relative_path"]
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "file"
    ]
    skipped_paths = [
        n["attributes"]["relative_path"]
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "skipped_file"
        and n.get("attributes", {}).get("reason") == "include_filter_mismatch"
    ]

    assert manifest["attributes"]["include_patterns"] == ["adr"]
    assert manifest["attributes"]["skipped_by_reason"] == {"include_filter_mismatch": 1}
    assert file_paths == ["adr/accepted.md"]
    assert skipped_paths == ["notes/draft.md"]


def test_directory_corpus_skip_report_limit_bounds_nodes(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")
    for index in range(3):
        (docs / f"ignored-{index}.bin").write_bytes(b"\x00")

    graph = build_corpus_graph(
        str(docs),
        graph_type="knowledge",
        skip_report_limit=1,
    )
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    reported_skips = [
        n
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "skipped_file"
        and n.get("attributes", {}).get("reason") == "unsupported_extension"
    ]

    assert manifest["attributes"]["skipped_by_reason"] == {"unsupported_extension": 3}
    assert manifest["attributes"]["reported_skipped_file_count"] == 1
    assert manifest["attributes"]["unreported_skipped_file_count"] == 2
    assert manifest["attributes"]["skip_report_truncated"] is True
    assert len(reported_skips) == 1


def test_directory_corpus_skip_digest_covers_unreported_scan_skips(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")
    for index in range(3):
        (docs / f"ignored-{index}.bin").write_bytes(b"\x00")

    graph = build_corpus_graph(
        str(docs),
        graph_type="knowledge",
        skip_report_limit=1,
    )
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")

    assert manifest["attributes"]["skipped_file_records_sha256"] == _skip_records_sha256(
        [
            ("unsupported_extension", "file", "ignored-0.bin"),
            ("unsupported_extension", "file", "ignored-1.bin"),
            ("unsupported_extension", "file", "ignored-2.bin"),
        ]
    )
    assert manifest["attributes"]["skipped_file_records_sha256_is_complete"] is True


def test_directory_corpus_skip_report_limit_bounds_scan_and_runtime_skips(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.bin").write_bytes(b"\x00")
    (docs / "b.md").write_text("# B\n\nBeta document.", encoding="utf-8")

    graph = build_corpus_graph(
        str(docs),
        graph_type="knowledge",
        max_file_bytes=1,
        skip_report_limit=1,
    )
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    reported_skips = [
        n
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "skipped_file"
    ]

    assert manifest["attributes"]["skipped_by_reason"] == {
        "file_too_large": 1,
        "unsupported_extension": 1,
    }
    assert manifest["attributes"]["reported_skipped_file_count"] == 1
    assert manifest["attributes"]["unreported_skipped_file_count"] == 1
    assert manifest["attributes"]["skip_report_truncated"] is True
    assert len(reported_skips) == 1
    assert reported_skips[0]["attributes"]["reason"] == "unsupported_extension"
    assert manifest["attributes"]["skipped_file_records_sha256"] == _skip_records_sha256(
        [
            ("unsupported_extension", "file", "a.bin"),
            ("file_too_large", "file", "b.md"),
        ]
    )


def test_directory_corpus_limits_total_extracted_bytes(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")
    (docs / "b.md").write_text("# B\n\nBeta document.", encoding="utf-8")

    first_size = (docs / "a.md").stat().st_size
    graph = build_corpus_graph(
        str(docs),
        graph_type="knowledge",
        max_total_bytes=first_size,
        skip_report_limit=10,
    )
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    skipped = [
        n
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "skipped_file"
        and n.get("attributes", {}).get("reason") == "max_total_bytes_exceeded"
    ]

    assert manifest["attributes"]["extracted_file_count"] == 1
    assert manifest["attributes"]["extracted_total_bytes"] == first_size
    assert manifest["attributes"]["max_total_bytes"] == first_size
    assert manifest["attributes"]["max_total_bytes_reached"] is True
    assert manifest["attributes"]["skipped_by_reason"] == {"max_total_bytes_exceeded": 1}
    assert manifest["attributes"]["skip_report_truncated"] is False
    assert skipped[0]["attributes"]["relative_path"] == "b.md"
    assert skipped[0]["attributes"]["path_type"] == "file"


def test_directory_corpus_stops_after_total_byte_budget_is_reached(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")
    (docs / "b.md").write_text("# B\n\nBeta document has enough text.", encoding="utf-8")
    (docs / "c.md").write_text("# C\n", encoding="utf-8")

    first_size = (docs / "a.md").stat().st_size
    graph = build_corpus_graph(
        str(docs),
        graph_type="knowledge",
        max_total_bytes=first_size,
        skip_report_limit=10,
    )
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    skipped_paths = [
        n["attributes"]["relative_path"]
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "skipped_file"
        and n.get("attributes", {}).get("reason") == "max_total_bytes_exceeded"
    ]

    assert manifest["attributes"]["extracted_file_count"] == 1
    assert manifest["attributes"]["extracted_total_bytes"] == first_size
    assert manifest["attributes"]["skipped_by_reason"] == {"max_total_bytes_exceeded": 2}
    assert skipped_paths == ["b.md", "c.md"]


def test_directory_corpus_limits_recursion_depth(tmp_path):
    docs = tmp_path / "docs"
    shallow = docs / "shallow"
    deep = shallow / "deep"
    deep.mkdir(parents=True)
    (docs / "root.md").write_text("# Root\n\nRoot document.", encoding="utf-8")
    (shallow / "a.md").write_text("# A\n\nShallow document.", encoding="utf-8")
    (deep / "b.md").write_text("# B\n\nDeep document.", encoding="utf-8")

    graph = build_corpus_graph(
        str(docs),
        graph_type="knowledge",
        max_depth=1,
        skip_report_limit=10,
    )
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    file_paths = [
        n["attributes"]["relative_path"]
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "file"
    ]
    skipped = [
        n
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "skipped_file"
        and n.get("attributes", {}).get("reason") == "max_depth_exceeded"
    ]

    assert manifest["attributes"]["max_depth"] == 1
    assert manifest["attributes"]["max_depth_reached"] is True
    assert manifest["attributes"]["skipped_by_reason"] == {"max_depth_exceeded": 1}
    assert file_paths == ["root.md", "shallow/a.md"]
    assert skipped[0]["attributes"]["path_type"] == "directory"
    assert skipped[0]["attributes"]["relative_path"] == "shallow/deep"


def test_directory_corpus_can_truncate_large_scans(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")
    (docs / "b.md").write_text("# B\n\nBeta document.", encoding="utf-8")
    (docs / "c.md").write_text("# C\n\nGamma document.", encoding="utf-8")

    graph = build_corpus_graph(
        str(docs),
        graph_type="knowledge",
        max_scan_entries=2,
        skip_report_limit=10,
    )
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    file_paths = [
        n["attributes"]["relative_path"]
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "file"
    ]
    skipped = [
        n
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "skipped_file"
        and n.get("attributes", {}).get("reason") == "max_scan_entries_exceeded"
    ]

    assert file_paths == ["a.md", "b.md"]
    assert manifest["attributes"]["max_scan_entries"] == 2
    assert manifest["attributes"]["max_scan_entries_reached"] is True
    assert manifest["attributes"]["scanned_entry_count"] == 2
    assert manifest["attributes"]["skipped_file_count_is_complete"] is False
    assert manifest["attributes"]["skipped_file_records_sha256_is_complete"] is False
    assert manifest["attributes"]["skipped_by_reason"] == {"max_scan_entries_exceeded": 1}
    assert skipped[0]["attributes"]["relative_path"] == "c.md"


def test_cli_accepts_max_scan_entries(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    output = tmp_path / "graph.json"
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")
    (docs / "b.md").write_text("# B\n\nBeta document.", encoding="utf-8")

    exit_code = main(
        [
            str(docs),
            "--graph",
            "knowledge",
            "--max-scan-entries",
            "1",
            "--output",
            str(output),
        ]
    )

    graph = json.loads(output.read_text(encoding="utf-8"))
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")

    assert exit_code == 0
    assert manifest["attributes"]["max_scan_entries"] == 1
    assert manifest["attributes"]["max_scan_entries_reached"] is True


def test_cli_accepts_stop_after_max_files(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    output = tmp_path / "graph.json"
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")
    (docs / "b.md").write_text("# B\n\nBeta document.", encoding="utf-8")

    exit_code = main(
        [
            str(docs),
            "--graph",
            "knowledge",
            "--max-files",
            "1",
            "--stop-after-max-files",
            "--output",
            str(output),
        ]
    )

    graph = json.loads(output.read_text(encoding="utf-8"))
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")

    assert exit_code == 0
    assert manifest["attributes"]["max_files"] == 1
    assert manifest["attributes"]["stop_after_max_files"] is True
    assert manifest["attributes"]["max_files_scan_truncated"] is True


def test_document_graph_from_directory_accepts_corpus_options(tmp_path):
    docs = tmp_path / "docs"
    deep = docs / "deep"
    deep.mkdir(parents=True)
    (docs / "root.md").write_text("# Root\n\nRoot document.", encoding="utf-8")
    (deep / "hidden.md").write_text("# Hidden\n\nDeep document.", encoding="utf-8")

    graph = DocumentGraph.from_directory(str(docs), graph_type="knowledge", max_depth=0).to_dict()
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    file_paths = [
        n["attributes"]["relative_path"]
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "file"
    ]

    assert manifest["attributes"]["max_depth_reached"] is True
    assert file_paths == ["root.md"]


def test_non_recursive_directory_corpus_reports_skipped_subdirectories(tmp_path):
    docs = tmp_path / "docs"
    nested = docs / "nested"
    nested.mkdir(parents=True)
    (docs / "root.md").write_text("# Root\n\nRoot document.", encoding="utf-8")
    (nested / "hidden.md").write_text("# Hidden\n\nNested document.", encoding="utf-8")

    graph = build_corpus_graph(
        str(docs),
        graph_type="knowledge",
        recursive=False,
        skip_report_limit=10,
    )
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    file_paths = [
        n["attributes"]["relative_path"]
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "file"
    ]
    skipped = [
        n
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "skipped_file"
        and n.get("attributes", {}).get("reason") == "non_recursive_directory"
    ]

    assert manifest["attributes"]["recursive"] is False
    assert manifest["attributes"]["skipped_by_reason"] == {"non_recursive_directory": 1}
    assert file_paths == ["root.md"]
    assert skipped[0]["attributes"]["path_type"] == "directory"
    assert skipped[0]["attributes"]["relative_path"] == "nested"


def test_directory_scan_prunes_default_ignored_directories(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "visible.md").write_text("# Visible\n", encoding="utf-8")
    generated = docs / "node_modules" / "pkg"
    generated.mkdir(parents=True)
    (generated / "hidden.md").write_text("# Hidden\n", encoding="utf-8")

    graph = build_corpus_graph(str(docs), graph_type="knowledge")
    file_paths = {
        n["attributes"]["relative_path"]
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "file"
    }

    assert file_paths == {"visible.md"}


def test_directory_scan_ignores_only_relative_generated_names(tmp_path):
    generated_parent = tmp_path / "build"
    docs = generated_parent / "docs"
    docs.mkdir(parents=True)
    (docs / "visible.md").write_text("# Visible\n\nParent path should not matter.", encoding="utf-8")

    graph = build_corpus_graph(str(docs), graph_type="knowledge", skip_report_limit=10)
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    file_paths = [
        n["attributes"]["relative_path"]
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "file"
    ]

    assert file_paths == ["visible.md"]
    assert manifest["attributes"]["skipped_by_reason"] == {}


def test_directory_scan_prunes_doc2graph_and_packaging_artifacts(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "visible.md").write_text("# Visible\n", encoding="utf-8")
    (docs / "DOC2GRAPH_PROGRESS.md").write_text("# Generated progress\n", encoding="utf-8")
    (docs / "DOC2GRAPH_NEXT_PROMPT.md").write_text("# Generated prompt\n", encoding="utf-8")
    (docs / ".doc2graph-cache.json").write_text("{}", encoding="utf-8")
    runs = docs / ".doc2graph-runs"
    runs.mkdir()
    (runs / "doc2graph.all.json").write_text("{}", encoding="utf-8")
    egg_info = docs / "doc2graph.egg-info"
    egg_info.mkdir()
    (egg_info / "SOURCES.txt").write_text("doc2graph/corpus.py\n", encoding="utf-8")

    graph = build_corpus_graph(str(docs), graph_type="knowledge", skip_report_limit=10)
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    file_paths = [
        n["attributes"]["relative_path"]
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "file"
    ]
    skipped_paths = {
        n["attributes"]["relative_path"]
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "skipped_file"
        and n.get("attributes", {}).get("reason") == "default_ignore_match"
    }

    assert file_paths == ["visible.md"]
    assert manifest["attributes"]["selected_file_count"] == 1
    assert manifest["attributes"]["skipped_by_reason"] == {"default_ignore_match": 5}
    assert skipped_paths == {
        ".doc2graph-cache.json",
        ".doc2graph-runs",
        "DOC2GRAPH_NEXT_PROMPT.md",
        "DOC2GRAPH_PROGRESS.md",
        "doc2graph.egg-info",
    }


def test_directory_corpus_reports_default_ignored_directories(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "visible.md").write_text("# Visible\n", encoding="utf-8")
    generated = docs / "node_modules" / "pkg"
    generated.mkdir(parents=True)
    (generated / "hidden.md").write_text("# Hidden\n", encoding="utf-8")

    graph = build_corpus_graph(str(docs), graph_type="knowledge", skip_report_limit=10)
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    skipped = [
        n
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "skipped_file"
        and n.get("attributes", {}).get("reason") == "default_ignore_match"
    ]

    assert manifest["attributes"]["skipped_by_reason"] == {"default_ignore_match": 1}
    assert skipped[0]["attributes"]["path_type"] == "directory"
    assert skipped[0]["attributes"]["relative_path"] == "node_modules"


def test_directory_corpus_reports_user_excluded_paths(tmp_path):
    docs = tmp_path / "docs"
    adr = docs / "adr"
    notes = docs / "notes"
    adr.mkdir(parents=True)
    notes.mkdir()
    (adr / "accepted.md").write_text("# Accepted\n\nChosen option.", encoding="utf-8")
    (notes / "draft.md").write_text("# Draft\n\nExcluded notes.", encoding="utf-8")

    graph = build_corpus_graph(
        str(docs),
        graph_type="knowledge",
        exclude=["notes"],
        skip_report_limit=10,
    )
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    file_paths = [
        n["attributes"]["relative_path"]
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "file"
    ]
    skipped = [
        n
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "skipped_file"
        and n.get("attributes", {}).get("reason") == "exclude_filter_match"
    ]

    assert manifest["attributes"]["exclude_patterns"] == ["notes"]
    assert manifest["attributes"]["skipped_by_reason"] == {"exclude_filter_match": 1}
    assert file_paths == ["adr/accepted.md"]
    assert skipped[0]["attributes"]["path_type"] == "directory"
    assert skipped[0]["attributes"]["relative_path"] == "notes"


def test_directory_corpus_reports_symlinked_directories_without_descending(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "visible.md").write_text("# Visible\n", encoding="utf-8")
    loop = docs / "loop"
    try:
        loop.symlink_to(docs, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    graph = build_corpus_graph(str(docs), graph_type="knowledge", skip_report_limit=10)
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    skipped = [
        n
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "skipped_file"
        and n.get("attributes", {}).get("reason") == "symlink_directory"
    ]
    file_paths = [
        n["attributes"]["relative_path"]
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "file"
    ]

    assert manifest["attributes"]["skipped_by_reason"] == {"symlink_directory": 1}
    assert skipped[0]["attributes"]["path_type"] == "directory"
    assert file_paths == ["visible.md"]


def test_directory_corpus_can_reuse_explicit_cache(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    cache = tmp_path / "doc2graph-cache.json"
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")
    (docs / "b.md").write_text("# B\n\nBeta document.", encoding="utf-8")

    first = build_corpus_graph(str(docs), graph_type="knowledge", cache_path=cache)
    second = build_corpus_graph(str(docs), graph_type="knowledge", cache_path=cache)
    refreshed = build_corpus_graph(
        str(docs),
        graph_type="knowledge",
        cache_path=cache,
        refresh_cache=True,
    )

    first_manifest = next(n for n in first["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    second_manifest = next(n for n in second["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    refreshed_manifest = next(
        n for n in refreshed["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest"
    )

    assert cache.exists()
    assert first_manifest["attributes"]["cache_enabled"] is True
    assert first_manifest["attributes"]["cache_hits"] == 0
    assert first_manifest["attributes"]["cache_misses"] == 2
    assert first_manifest["attributes"]["cache_writes"] == 2
    assert first_manifest["attributes"]["cache_content_digest_hits"] == 0
    assert first_manifest["attributes"]["cache_content_digest_misses"] == 2
    assert first_manifest["attributes"]["cache_file_updated"] is True
    assert second_manifest["attributes"]["cache_hits"] == 2
    assert second_manifest["attributes"]["cache_misses"] == 0
    assert second_manifest["attributes"]["cache_writes"] == 0
    assert second_manifest["attributes"]["cache_content_digest_hits"] == 2
    assert second_manifest["attributes"]["cache_content_digest_misses"] == 0
    assert second_manifest["attributes"]["cache_file_updated"] is False
    assert refreshed_manifest["attributes"]["cache_refresh"] is True
    assert refreshed_manifest["attributes"]["cache_hits"] == 0
    assert refreshed_manifest["attributes"]["cache_misses"] == 2
    assert refreshed_manifest["attributes"]["cache_writes"] == 2
    assert refreshed_manifest["attributes"]["cache_content_digest_hits"] == 2
    assert refreshed_manifest["attributes"]["cache_content_digest_misses"] == 0
    assert refreshed_manifest["attributes"]["cache_file_updated"] is False
    assert first_manifest["attributes"]["cache_load_status"] == "missing"
    assert first_manifest["attributes"]["cache_entry_count_before"] == 0
    assert first_manifest["attributes"]["cache_entry_count_after"] == 2
    assert second_manifest["attributes"]["cache_load_status"] == "loaded"
    assert second_manifest["attributes"]["cache_entry_count_before"] == 2
    assert second_manifest["attributes"]["cache_entry_count_after"] == 2
    assert {
        n["attributes"]["relative_path"]: n["attributes"]["cache_status"]
        for n in second["nodes"]
        if n.get("attributes", {}).get("type") == "file"
    } == {"a.md": "hit", "b.md": "hit"}
    assert {
        n["attributes"]["relative_path"]: n["attributes"]["cache_status"]
        for n in refreshed["nodes"]
        if n.get("attributes", {}).get("type") == "file"
    } == {"a.md": "refresh", "b.md": "refresh"}


def test_directory_corpus_reuses_cached_content_digests_for_unchanged_files(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    cache = tmp_path / "doc2graph-cache.json"
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")
    (docs / "b.md").write_text("# B\n\nBeta document.", encoding="utf-8")

    from doc2graph import corpus

    original_file_sha256 = corpus._file_sha256
    calls = []

    def track_file_sha256(path):
        calls.append(path.name)
        return original_file_sha256(path)

    monkeypatch.setattr(corpus, "_file_sha256", track_file_sha256)

    build_corpus_graph(str(docs), graph_type="knowledge", cache_path=cache)
    first_hashes = list(calls)
    calls.clear()
    second = build_corpus_graph(str(docs), graph_type="knowledge", cache_path=cache)

    manifest = next(n for n in second["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")

    assert sorted(first_hashes) == ["a.md", "b.md"]
    assert calls == []
    assert manifest["attributes"]["cache_hits"] == 2
    assert manifest["attributes"]["cache_content_digest_hits"] == 2
    assert manifest["attributes"]["cache_content_digest_misses"] == 0


def test_directory_corpus_reports_invalid_cache_json_and_rebuilds(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    cache = tmp_path / "doc2graph-cache.json"
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")
    cache.write_text("{not-json", encoding="utf-8")

    graph = build_corpus_graph(str(docs), graph_type="knowledge", cache_path=cache)

    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    payload = json.loads(cache.read_text(encoding="utf-8"))

    assert manifest["attributes"]["cache_load_status"] == "invalid_json"
    assert manifest["attributes"]["cache_entry_count_before"] == 0
    assert manifest["attributes"]["cache_entry_count_after"] == 1
    assert manifest["attributes"]["cache_hits"] == 0
    assert manifest["attributes"]["cache_misses"] == 1
    assert manifest["attributes"]["cache_writes"] == 1
    assert payload["version"] == 1
    assert len(payload["entries"]) == 1


def test_directory_corpus_reports_invalid_cache_schema_and_rebuilds(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    cache = tmp_path / "doc2graph-cache.json"
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")
    cache.write_text(json.dumps({"version": 999, "entries": []}), encoding="utf-8")

    graph = build_corpus_graph(str(docs), graph_type="knowledge", cache_path=cache)

    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    payload = json.loads(cache.read_text(encoding="utf-8"))

    assert manifest["attributes"]["cache_load_status"] == "invalid_schema"
    assert manifest["attributes"]["cache_entry_count_before"] == 0
    assert manifest["attributes"]["cache_entry_count_after"] == 1
    assert payload["version"] == 1
    assert len(payload["entries"]) == 1


def test_directory_corpus_reports_cache_write_error_without_failing(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    blocker = tmp_path / "not-a-directory"
    cache = blocker / "doc2graph-cache.json"
    blocker.write_text("blocks cache directory creation", encoding="utf-8")
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")

    graph = build_corpus_graph(str(docs), graph_type="knowledge", cache_path=cache)

    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    file_attrs = next(n["attributes"] for n in graph["nodes"] if n.get("attributes", {}).get("type") == "file")

    assert manifest["attributes"]["extracted_file_count"] == 1
    assert manifest["attributes"]["cache_load_status"] == "missing"
    assert manifest["attributes"]["cache_entry_count_after"] == 1
    assert manifest["attributes"]["cache_file_updated"] is False
    assert manifest["attributes"]["cache_write_status"] == "write_error"
    assert manifest["attributes"]["cache_write_error"]
    assert file_attrs["status"] == "extracted"
    assert file_attrs["cache_status"] == "miss"


def test_directory_corpus_marks_failed_file_status_and_manifest(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    good = docs / "good.md"
    bad = docs / "bad.md"
    good.write_text("# Good\n\nExtractable document.", encoding="utf-8")
    bad.write_text("# Bad\n\nBroken document.", encoding="utf-8")

    from doc2graph import cli

    original_build_file_graph = cli.build_file_graph

    def fail_bad_file(path, graph_type="knowledge"):
        if path == str(bad):
            raise RuntimeError("boom")
        return original_build_file_graph(path, graph_type)

    monkeypatch.setattr(cli, "build_file_graph", fail_bad_file)

    graph = build_corpus_graph(str(docs), graph_type="knowledge", skip_report_limit=10)
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    file_attrs = {
        n["attributes"]["relative_path"]: n["attributes"]
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "file"
    }
    error = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "load_error")

    assert manifest["attributes"]["extracted_file_count"] == 1
    assert manifest["attributes"]["failed_file_count"] == 1
    assert manifest["attributes"]["skipped_by_reason"] == {"load_error": 1}
    assert file_attrs["good.md"]["status"] == "extracted"
    assert file_attrs["bad.md"]["status"] == "failed"
    assert file_attrs["bad.md"]["error_type"] == "RuntimeError"
    assert file_attrs["bad.md"]["error_message"] == "boom"
    assert error["attributes"]["suffix"] == ".md"
    assert error["attributes"]["size_bytes"] == bad.stat().st_size


def test_directory_corpus_invalidates_cache_when_extraction_fingerprint_changes(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    cache = tmp_path / "doc2graph-cache.json"
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")

    first = build_corpus_graph(str(docs), graph_type="knowledge", cache_path=cache)
    payload = json.loads(cache.read_text(encoding="utf-8"))
    entry = next(iter(payload["entries"].values()))
    entry["metadata"]["extraction_fingerprint"] = "outdated"
    cache.write_text(json.dumps(payload), encoding="utf-8")

    second = build_corpus_graph(str(docs), graph_type="knowledge", cache_path=cache)
    first_manifest = next(
        n for n in first["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest"
    )
    second_manifest = next(
        n for n in second["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest"
    )
    refreshed_payload = json.loads(cache.read_text(encoding="utf-8"))
    refreshed_entry = next(iter(refreshed_payload["entries"].values()))

    assert first_manifest["attributes"]["cache_extraction_fingerprint"]
    assert first_manifest["attributes"]["cache_misses"] == 1
    assert second_manifest["attributes"]["cache_hits"] == 0
    assert second_manifest["attributes"]["cache_misses"] == 1
    assert second_manifest["attributes"]["cache_writes"] == 1
    assert refreshed_entry["metadata"]["extraction_fingerprint"] == second_manifest["attributes"][
        "cache_extraction_fingerprint"
    ]


def test_directory_corpus_cache_uses_content_digest_not_only_stat_metadata(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    cache = tmp_path / "doc2graph-cache.json"
    source = docs / "a.md"
    original = "# A\n\nThis document describes Alpha graph cache."
    updated = "# A\n\nThis document describes Bravo graph cache."
    assert len(original.encode("utf-8")) == len(updated.encode("utf-8"))
    source.write_text(original, encoding="utf-8")

    first = build_corpus_graph(str(docs), graph_type="knowledge", cache_path=cache)
    original_stat = source.stat()
    time.sleep(0.05)
    source.write_text(updated, encoding="utf-8")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    second = build_corpus_graph(str(docs), graph_type="knowledge", cache_path=cache)

    first_manifest = next(
        n for n in first["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest"
    )
    second_manifest = next(
        n for n in second["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest"
    )
    payload = json.loads(cache.read_text(encoding="utf-8"))
    refreshed_entry = next(iter(payload["entries"].values()))
    section = next(
        n
        for n in second["nodes"]
        if n.get("attributes", {}).get("type") == "section"
        and n.get("attributes", {}).get("source") == str(source)
    )

    assert first_manifest["attributes"]["cache_validation"] == "content_sha256"
    assert second_manifest["attributes"]["cache_hits"] == 0
    assert second_manifest["attributes"]["cache_misses"] == 1
    assert second_manifest["attributes"]["cache_writes"] == 1
    assert second_manifest["attributes"]["cache_content_digest_hits"] == 0
    assert second_manifest["attributes"]["cache_content_digest_misses"] == 1
    assert refreshed_entry["metadata"]["content_sha256"]
    assert "Bravo graph cache" in section["content"]


def test_directory_corpus_prunes_stale_cache_entries(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    cache = tmp_path / "doc2graph-cache.json"
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")
    stale = docs / "b.md"
    stale.write_text("# B\n\nBeta document.", encoding="utf-8")

    build_corpus_graph(str(docs), graph_type="knowledge", cache_path=cache)
    stale.unlink()
    second = build_corpus_graph(str(docs), graph_type="knowledge", cache_path=cache)

    manifest = next(n for n in second["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    payload = json.loads(cache.read_text(encoding="utf-8"))
    cached_paths = {
        entry["metadata"]["relative_path"]
        for entry in payload["entries"].values()
        if entry.get("metadata", {}).get("root") == str(docs.resolve())
    }

    assert manifest["attributes"]["cache_hits"] == 1
    assert manifest["attributes"]["cache_pruned"] == 1
    assert manifest["attributes"]["cache_file_updated"] is True
    assert cached_paths == {"a.md"}


def test_directory_corpus_prunes_cache_entries_outside_current_include(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    cache = tmp_path / "doc2graph-cache.json"
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")
    (docs / "b.md").write_text("# B\n\nBeta document.", encoding="utf-8")

    build_corpus_graph(str(docs), graph_type="knowledge", cache_path=cache)
    filtered = build_corpus_graph(
        str(docs),
        graph_type="knowledge",
        include=["a.md"],
        cache_path=cache,
    )

    manifest = next(n for n in filtered["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    payload = json.loads(cache.read_text(encoding="utf-8"))
    cached_paths = {
        entry["metadata"]["relative_path"]
        for entry in payload["entries"].values()
        if entry.get("metadata", {}).get("root") == str(docs.resolve())
    }

    assert manifest["attributes"]["selected_file_count"] == 1
    assert manifest["attributes"]["cache_hits"] == 1
    assert manifest["attributes"]["cache_pruned"] == 1
    assert cached_paths == {"a.md"}


def test_directory_corpus_keeps_cache_entries_outside_max_files_run(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    cache = tmp_path / "doc2graph-cache.json"
    for name in ("a.md", "b.md", "c.md"):
        (docs / name).write_text(f"# {name}\n\nCached document.", encoding="utf-8")

    build_corpus_graph(str(docs), graph_type="knowledge", cache_path=cache)
    limited = build_corpus_graph(
        str(docs),
        graph_type="knowledge",
        max_files=1,
        cache_path=cache,
    )

    manifest = next(n for n in limited["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    payload = json.loads(cache.read_text(encoding="utf-8"))
    cached_paths = {
        entry["metadata"]["relative_path"]
        for entry in payload["entries"].values()
        if entry.get("metadata", {}).get("root") == str(docs.resolve())
    }

    assert manifest["attributes"]["selected_file_count"] == 1
    assert manifest["attributes"]["cache_hits"] == 1
    assert manifest["attributes"]["cache_pruned"] == 0
    assert manifest["attributes"]["cache_prune_status"] == "skipped_bounded_selection"
    assert cached_paths == {"a.md", "b.md", "c.md"}


def test_directory_corpus_keeps_cache_entries_outside_total_byte_budget(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    cache = tmp_path / "doc2graph-cache.json"
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")
    (docs / "b.md").write_text("# B\n\nBeta document.", encoding="utf-8")

    build_corpus_graph(str(docs), graph_type="knowledge", cache_path=cache)
    first_size = (docs / "a.md").stat().st_size
    limited = build_corpus_graph(
        str(docs),
        graph_type="knowledge",
        max_total_bytes=first_size,
        cache_path=cache,
    )

    manifest = next(n for n in limited["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    payload = json.loads(cache.read_text(encoding="utf-8"))
    cached_paths = {
        entry["metadata"]["relative_path"]
        for entry in payload["entries"].values()
        if entry.get("metadata", {}).get("root") == str(docs.resolve())
    }

    assert manifest["attributes"]["extracted_file_count"] == 1
    assert manifest["attributes"]["cache_hits"] == 1
    assert manifest["attributes"]["cache_pruned"] == 0
    assert manifest["attributes"]["cache_prune_status"] == "skipped_bounded_extraction"
    assert cached_paths == {"a.md", "b.md"}


def test_directory_corpus_does_not_extract_cache_file_inside_corpus(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    cache = docs / ".doc2graph-cache.json"
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")

    build_corpus_graph(str(docs), graph_type="knowledge", cache_path=cache)
    second = build_corpus_graph(str(docs), graph_type="knowledge", cache_path=cache)

    manifest = next(n for n in second["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    file_paths = [
        n["attributes"]["relative_path"]
        for n in second["nodes"]
        if n.get("attributes", {}).get("type") == "file"
    ]
    skipped = [
        n
        for n in second["nodes"]
        if n.get("attributes", {}).get("type") == "skipped_file"
        and n.get("attributes", {}).get("reason") == "reserved_output_file"
    ]
    payload = json.loads(cache.read_text(encoding="utf-8"))
    cached_paths = {
        entry["metadata"]["relative_path"]
        for entry in payload["entries"].values()
        if entry.get("metadata", {}).get("root") == str(docs.resolve())
    }

    assert manifest["attributes"]["selected_file_count"] == 1
    assert manifest["attributes"]["skipped_by_reason"] == {"reserved_output_file": 1}
    assert manifest["attributes"]["cache_hits"] == 1
    assert file_paths == ["a.md"]
    assert skipped[0]["attributes"]["relative_path"] == ".doc2graph-cache.json"
    assert cached_paths == {"a.md"}


def test_cli_does_not_extract_existing_output_file_inside_corpus(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    output = docs / "corpus.md"
    (docs / "a.md").write_text("# A\n\nAlpha document.", encoding="utf-8")
    output.write_text("# Previous output\n\nOld generated graph artifact.", encoding="utf-8")

    exit_code = main([str(docs), "--graph", "knowledge", "--output", str(output)])

    graph = json.loads(output.read_text(encoding="utf-8"))
    manifest = next(n for n in graph["nodes"] if n.get("attributes", {}).get("type") == "corpus_manifest")
    file_paths = [
        n["attributes"]["relative_path"]
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "file"
    ]
    skipped = [
        n
        for n in graph["nodes"]
        if n.get("attributes", {}).get("type") == "skipped_file"
        and n.get("attributes", {}).get("reason") == "reserved_output_file"
    ]

    assert exit_code == 0
    assert manifest["attributes"]["selected_file_count"] == 1
    assert manifest["attributes"]["skipped_by_reason"] == {"reserved_output_file": 1}
    assert file_paths == ["a.md"]
    assert skipped[0]["attributes"]["relative_path"] == "corpus.md"
