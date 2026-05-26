import argparse
from pathlib import Path

from docs2graph.iterate import _committable_changed_paths
from docs2graph.loop import _build_command
from docs2graph.prompt import TestResult, build_iteration_prompt, summarize_graph


def test_summarize_graph_reports_health_and_types():
    graph = {
        "nodes": [
            {"id": "doc", "label": "Doc", "attributes": {"type": "document", "source": "a.md"}},
            {"id": "claim", "label": "Claim", "attributes": {"type": "claim", "source": "a.md"}},
        ],
        "edges": [{"from": "doc", "to": "claim", "label": "contains"}],
        "current_node_id": "doc",
    }

    summary = summarize_graph(graph)

    assert summary["node_count"] == 2
    assert summary["edge_count"] == 1
    assert summary["dangling_edge_count"] == 0
    assert summary["node_types"]["document"] == 1
    assert summary["edge_labels"]["contains"] == 1


def test_iteration_prompt_describes_doc2graph_goals(tmp_path):
    prompt = build_iteration_prompt(
        target_path=tmp_path,
        graph_type="all",
        snapshot=tmp_path / "snapshot.json",
        summary={
            "node_count": 3,
            "edge_count": 2,
            "dangling_edge_count": 0,
            "isolated_node_count": 0,
            "node_types": {"document": 1},
            "edge_labels": {"contains": 2},
            "top_sources": {"README.md": 1},
        },
        previous_snapshot=None,
        previous_summary=None,
        test_result=TestResult("python -m pytest -q", 0, "passed"),
    )

    assert "single documents, multiple documents, URLs, media files, and document folders" in prompt
    assert "personalized PageRank" in prompt
    assert "Do not commit generated snapshots" in prompt


def test_loop_command_invokes_doc2graph_iterate(tmp_path):
    args = argparse.Namespace(
        path=str(tmp_path),
        graph="all",
        interval_minutes=20.0,
        output_dir=".doc2graph-runs",
        report_file="DOC2GRAPH_PROGRESS.md",
        prompt_file="DOC2GRAPH_NEXT_PROMPT.md",
        test_command="python -m pytest -q",
        commit_push=True,
        codex=True,
        codex_bin="/tmp/codex",
        codex_timeout_seconds=900,
        report_command=None,
        discord_webhook_url=None,
    )

    command = _build_command(args)

    assert command[:3] == [command[0], "-m", "docs2graph.iterate"]
    assert "--interval-minutes" in command
    assert "--commit-push" in command
    assert "--codex" in command


def test_committable_paths_exclude_runtime_files(monkeypatch, tmp_path):
    def fake_run(command, cwd):
        class Result:
            stdout = (
                " M DOC2GRAPH_PROGRESS.md\n"
                " M DOC2GRAPH_NEXT_PROMPT.md\n"
                " M .doc2graph-runs/demo.json\n"
                " M docs2graph/cli.py\n"
                " M tests/test_cli.py\n"
            )

        return Result()

    monkeypatch.setattr("docs2graph.iterate._run", fake_run)

    paths = _committable_changed_paths(Path("/repo"))

    assert Path("docs2graph/cli.py") in paths
    assert Path("tests/test_cli.py") in paths
    assert Path("DOC2GRAPH_PROGRESS.md") not in paths
    assert not any(str(path).startswith(".doc2graph-runs") for path in paths)
