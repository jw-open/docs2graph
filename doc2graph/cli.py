"""Command line interface for doc2graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .extractors.decision import extract_decision_graph
from .extractors.knowledge import extract_knowledge_graph
from .extractors.media import extract_media_graph, is_media_path
from .extractors.schema import extract_schema_graph
from .corpus import build_corpus_graph
from .loaders.auto import load_document


def build_graph(path: str, graph_type: str = "knowledge") -> Dict[str, Any]:
    return build_corpus_graph(path, graph_type=graph_type)


def build_file_graph(path: str, graph_type: str = "knowledge") -> Dict[str, Any]:
    text = load_document(path)
    if graph_type == "knowledge":
        return extract_knowledge_graph(text, source=path)
    if graph_type == "decision":
        return extract_decision_graph(text, source=path)
    if graph_type == "schema":
        return extract_schema_graph(text, source=path)
    if graph_type == "media":
        return extract_media_graph(path, text=text)
    if graph_type == "all":
        graphs = [
            extract_knowledge_graph(text, source=path),
            extract_decision_graph(text, source=path),
            extract_schema_graph(text, source=path),
        ]
        if is_media_path(path):
            graphs.append(extract_media_graph(path, text=text))
        return _merge_graphs(graphs)
    raise ValueError(f"Unsupported graph type: {graph_type}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract graphs from papers, documentation, and document corpora.")
    parser.add_argument("path", help="Document path, URL, or directory")
    parser.add_argument(
        "--graph",
        choices=["knowledge", "decision", "schema", "media", "all"],
        default="knowledge",
        help="Graph type to extract",
    )
    parser.add_argument("--output", "-o", help="Write JSON graph to this path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument("--no-recursive", action="store_true", help="Do not recurse into subdirectories")
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help=(
            "Extract symlinked files inside directory corpora; symlinked "
            "directories are still skipped to avoid traversal loops"
        ),
    )
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process from a directory")
    parser.add_argument(
        "--stop-after-max-files",
        action="store_true",
        help=(
            "Stop scanning as soon as --max-files is exceeded; faster for very "
            "large trees but skipped-file counts beyond that point are incomplete"
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Maximum directory depth to descend when processing a directory; 0 means root files only",
    )
    parser.add_argument(
        "--max-scan-entries",
        type=int,
        help=(
            "Maximum filesystem entries to inspect during directory scanning; "
            "set -1 to disable"
        ),
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=25 * 1024 * 1024,
        help="Skip individual directory files larger than this many bytes; set -1 to disable",
    )
    parser.add_argument(
        "--max-total-bytes",
        type=int,
        help="Stop extracting directory files after this many cumulative bytes; set -1 to disable",
    )
    parser.add_argument(
        "--include",
        action="append",
        help="Directory include glob, relative to corpus root; may be repeated",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        help="Directory exclude glob or name; may be repeated",
    )
    parser.add_argument(
        "--skip-report-limit",
        type=int,
        default=100,
        help="Maximum number of skipped directory files to list as skipped_file nodes",
    )
    parser.add_argument(
        "--cache",
        help="Optional corpus cache JSON path for reusing unchanged per-file graph extraction",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rebuild cached per-file graphs instead of reading existing cache entries",
    )
    parser.add_argument(
        "--max-file-reference-links",
        type=int,
        help=(
            "Maximum explicit relative file links to add after merging a "
            "directory corpus; set -1 to disable"
        ),
    )
    parser.add_argument(
        "--max-cross-document-links",
        type=int,
        help=(
            "Maximum cross-document mention edges to add after merging a "
            "directory corpus; set -1 to disable"
        ),
    )
    args = parser.parse_args(argv)

    max_file_bytes = None if args.max_file_bytes < 0 else args.max_file_bytes
    max_total_bytes = (
        None
        if args.max_total_bytes is None or args.max_total_bytes < 0
        else args.max_total_bytes
    )
    max_scan_entries = (
        None
        if args.max_scan_entries is None or args.max_scan_entries < 0
        else args.max_scan_entries
    )
    max_cross_document_links = (
        None
        if args.max_cross_document_links is None or args.max_cross_document_links < 0
        else args.max_cross_document_links
    )
    max_file_reference_links = (
        None
        if args.max_file_reference_links is None or args.max_file_reference_links < 0
        else args.max_file_reference_links
    )
    graph = build_corpus_graph(
        args.path,
        args.graph,
        recursive=not args.no_recursive,
        follow_symlinks=args.follow_symlinks,
        max_files=args.max_files,
        stop_after_max_files=args.stop_after_max_files,
        max_depth=args.max_depth,
        max_scan_entries=max_scan_entries,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        include=args.include,
        exclude=args.exclude,
        skip_report_limit=args.skip_report_limit,
        cache_path=args.cache,
        output_path=args.output,
        refresh_cache=args.refresh_cache,
        max_file_reference_links=max_file_reference_links,
        max_cross_document_links=max_cross_document_links,
    )
    payload = json.dumps(graph, indent=2 if args.pretty else None, sort_keys=args.pretty)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


def _merge_graphs(graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
    nodes = []
    edges = []
    seen_nodes = set()
    seen_edges = set()
    for graph in graphs:
        for node in graph.get("nodes", []):
            node_id = node.get("id")
            if node_id and node_id not in seen_nodes:
                nodes.append(node)
                seen_nodes.add(node_id)
        for edge in graph.get("edges", []):
            key = (edge.get("from"), edge.get("to"), edge.get("label"))
            if key not in seen_edges:
                edges.append(edge)
                seen_edges.add(key)
    current_node_id = graphs[0].get("current_node_id") if graphs else None
    return {"nodes": nodes, "edges": edges, "current_node_id": current_node_id}


if __name__ == "__main__":
    raise SystemExit(main())
