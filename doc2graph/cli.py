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
from .loaders.auto import load_document


def build_graph(path: str, graph_type: str = "knowledge") -> Dict[str, Any]:
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
    parser = argparse.ArgumentParser(description="Extract graphs from papers and documentation.")
    parser.add_argument("path", help="Document path")
    parser.add_argument(
        "--graph",
        choices=["knowledge", "decision", "schema", "media", "all"],
        default="knowledge",
        help="Graph type to extract",
    )
    parser.add_argument("--output", "-o", help="Write JSON graph to this path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args(argv)

    graph = build_graph(args.path, args.graph)
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
