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
