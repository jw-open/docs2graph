"""Prompt and graph-summary helpers for autonomous doc2graph iterations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True, slots=True)
class TestResult:
    __test__ = False

    command: str
    returncode: int
    output: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def load_graph(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_ids = set()
    linked = set()
    node_types: dict[str, int] = {}
    edge_labels: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    dangling = 0

    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", ""))
            if node_id:
                node_ids.add(node_id)
            attrs = node.get("attributes", {})
            if isinstance(attrs, dict):
                node_type = str(attrs.get("type") or attrs.get("kind") or "unknown")
                node_types[node_type] = node_types.get(node_type, 0) + 1
                source = attrs.get("source") or attrs.get("relative_path")
                if source:
                    key = str(source)
                    source_counts[key] = source_counts.get(key, 0) + 1

    if isinstance(edges, list):
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            from_id = str(edge.get("from", ""))
            to_id = str(edge.get("to", ""))
            linked.update({from_id, to_id})
            if from_id not in node_ids or to_id not in node_ids:
                dangling += 1
            label = str(edge.get("label", "unknown"))
            edge_labels[label] = edge_labels.get(label, 0) + 1

    isolated = node_ids - linked
    current = graph.get("current_node_id")
    if current:
        isolated.discard(str(current))

    return {
        "node_count": len(nodes) if isinstance(nodes, list) else 0,
        "edge_count": len(edges) if isinstance(edges, list) else 0,
        "node_types": dict(sorted(node_types.items())),
        "edge_labels": dict(sorted(edge_labels.items())),
        "dangling_edge_count": dangling,
        "isolated_node_count": len(isolated),
        "top_sources": dict(sorted(source_counts.items(), key=lambda item: (-item[1], item[0]))[:8]),
    }


def find_previous_snapshot(output_dir: Path, current_snapshot: Path, target_name: str, graph_type: str) -> Path | None:
    pattern = f"{target_name}.{graph_type}.*.json"
    candidates = sorted(path for path in output_dir.glob(pattern) if path != current_snapshot)
    return candidates[-1] if candidates else None


def _counts_text(counts: object, limit: int = 10) -> str:
    if not isinstance(counts, dict) or not counts:
        return "- none"
    ordered = sorted(((str(k), int(v)) for k, v in counts.items()), key=lambda item: (-item[1], item[0]))
    return "\n".join(f"- {name}: {count}" for name, count in ordered[:limit])


def _test_text(result: TestResult | None) -> str:
    if result is None:
        return "- Not run."
    status = "passed" if result.passed else "failed"
    output = result.output.strip() or "(no output)"
    if len(output) > 1200:
        output = output[:1200].rstrip() + "\n... truncated ..."
    return f"- `{result.command}` {status} with exit code {result.returncode}.\n\n```text\n{output}\n```"


def _delta(current: Dict[str, Any], previous: Dict[str, Any] | None) -> str:
    if previous is None:
        return (
            f"- Nodes: {current['node_count']}\n"
            f"- Edges: {current['edge_count']}\n"
            f"- Dangling edges: {current['dangling_edge_count']}\n"
            f"- Isolated nodes: {current['isolated_node_count']}"
        )
    return (
        f"- Nodes: {current['node_count']} ({int(current['node_count']) - int(previous['node_count']):+})\n"
        f"- Edges: {current['edge_count']} ({int(current['edge_count']) - int(previous['edge_count']):+})\n"
        f"- Dangling edges: {current['dangling_edge_count']} "
        f"({int(current['dangling_edge_count']) - int(previous['dangling_edge_count']):+})\n"
        f"- Isolated nodes: {current['isolated_node_count']} "
        f"({int(current['isolated_node_count']) - int(previous['isolated_node_count']):+})"
    )


def build_iteration_prompt(
    *,
    target_path: Path,
    graph_type: str,
    snapshot: Path,
    summary: Dict[str, Any],
    previous_snapshot: Path | None,
    previous_summary: Dict[str, Any] | None,
    test_result: TestResult | None,
) -> str:
    warnings = []
    if int(summary.get("dangling_edge_count", 0)):
        warnings.append(f"Fix {summary['dangling_edge_count']} dangling edges.")
    if test_result is not None and not test_result.passed:
        warnings.append("Fix failing tests before expanding extraction.")
    if not warnings:
        warnings.append("No blocking graph health issue detected.")

    next_steps = [
        "Improve corpus-scale extraction for very large folder trees: limits, deterministic ordering, skipped-file reporting, and update/caching behavior.",
        "Improve knowledge graph quality for personalized PageRank: better concepts, claims, evidence, citations, cross-document links, and node provenance.",
        "Improve decision graph quality: extract problems, options, pros, cons, tradeoffs, decisions, consequences, and confidence without requiring an LLM.",
        "Improve document format support and resilience: PDFs, Office files, HTML, CSV, images/OCR, URLs, and mixed directories should fail per-file instead of failing the run.",
        "Add optional enrichment interfaces only behind explicit flags; default graph generation must remain deterministic and non-LLM.",
    ]

    return (
        "# doc2graph Next Iteration Prompt\n\n"
        "You are working only in the `doc2graph` repository. Continue improving a Python package that reads "
        "single documents, multiple documents, URLs, media files, and document folders, then emits OhWise-compatible "
        "graph JSON for personalized PageRank/context engineering.\n\n"
        "## Current Snapshot\n\n"
        f"- Target path: `{target_path}`\n"
        f"- Graph type: `{graph_type}`\n"
        f"- Current snapshot: `{snapshot}`\n"
        f"- Previous snapshot: `{previous_snapshot if previous_snapshot else 'none'}`\n\n"
        "## Graph Delta\n\n"
        f"{_delta(summary, previous_summary)}\n\n"
        "## Node Types\n\n"
        f"{_counts_text(summary.get('node_types'))}\n\n"
        "## Edge Labels\n\n"
        f"{_counts_text(summary.get('edge_labels'))}\n\n"
        "## Top Sources\n\n"
        f"{_counts_text(summary.get('top_sources'))}\n\n"
        "## Issues And Bugs To Check\n\n"
        + "\n".join(f"- {warning}" for warning in warnings)
        + "\n\n"
        "## Tests\n\n"
        f"{_test_text(test_result)}\n\n"
        "## Recommended Next Steps\n\n"
        + "\n".join(f"{index}. {step}" for index, step in enumerate(next_steps, start=1))
        + "\n\n"
        "## Commit Discipline\n\n"
        "- Use `jw-open <176761431+jw-open@users.noreply.github.com>`.\n"
        "- Push to `jwpublic:jw-open/doc2graph.git` `main`.\n"
        "- Commit only source, tests, docs, and packaging changes that improve the tool.\n"
        "- Do not commit generated snapshots or timestamp-only progress files.\n"
    )


def write_iteration_prompt(path: Path, prompt: str) -> None:
    path.write_text(prompt, encoding="utf-8")


def build_action_prompt(context_prompt: str) -> str:
    return (
        "You are an autonomous developer working on `doc2graph`, a pure-Python package that extracts graph JSON "
        "from documents and document corpora for OhWise personalized PageRank/context engineering.\n\n"
        "Your working directory is the `doc2graph` repo root.\n\n"
        "## Current Analysis Context\n\n"
        + context_prompt.strip()
        + """

## Your Task

1. Pick the ONE recommended step that adds the most concrete value right now.
2. Implement real package improvements in `doc2graph/`, `tests/`, docs, or packaging.
3. Keep default graph generation deterministic and non-LLM.
4. Run `python -m pytest -q` and fix failures.
5. Commit ONLY source/test/doc/package files. Never commit `.doc2graph-runs/*.json`,
   `DOC2GRAPH_PROGRESS.md`, or `DOC2GRAPH_NEXT_PROMPT.md`.
6. Use git identity `jw-open <176761431+jw-open@users.noreply.github.com>`.
7. Push to `origin main`.

Do not create snapshot-only commits. Each iteration must improve the actual doc2graph tool.
"""
    )
