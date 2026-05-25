import json

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


def test_directory_corpus_skips_large_files(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "small.md").write_text("# Small\n\nThis document describes something.", encoding="utf-8")
    (docs / "large.md").write_text("x" * 20, encoding="utf-8")

    graph = build_corpus_graph(
        str(docs),
        graph_type="knowledge",
        max_file_bytes=10,
    )
    node_types = {n.get("attributes", {}).get("type") for n in graph["nodes"]}

    assert "skipped_file" in node_types


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
    assert len(reported_skips) == 1


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
    assert len(reported_skips) == 1
    assert reported_skips[0]["attributes"]["reason"] == "unsupported_extension"


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
    assert skipped[0]["attributes"]["relative_path"] == "b.md"


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
    assert second_manifest["attributes"]["cache_hits"] == 2
    assert second_manifest["attributes"]["cache_misses"] == 0
    assert second_manifest["attributes"]["cache_writes"] == 0
    assert refreshed_manifest["attributes"]["cache_refresh"] is True
    assert refreshed_manifest["attributes"]["cache_hits"] == 0
    assert refreshed_manifest["attributes"]["cache_misses"] == 2
    assert refreshed_manifest["attributes"]["cache_writes"] == 2


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
