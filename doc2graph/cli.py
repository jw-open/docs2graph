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
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process from a directory")
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=25 * 1024 * 1024,
        help="Skip individual directory files larger than this many bytes; set -1 to disable",
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
    args = parser.parse_args(argv)

    max_file_bytes = None if args.max_file_bytes < 0 else args.max_file_bytes
    graph = build_corpus_graph(
        args.path,
        args.graph,
        recursive=not args.no_recursive,
        max_files=args.max_files,
        max_file_bytes=max_file_bytes,
        include=args.include,
        exclude=args.exclude,
        skip_report_limit=args.skip_report_limit,
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
