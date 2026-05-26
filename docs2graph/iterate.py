"""Run repeated autonomous doc2graph improvement iterations."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .cli import build_graph
from .prompt import (
    TestResult,
    build_action_prompt,
    build_iteration_prompt,
    find_previous_snapshot,
    load_graph,
    summarize_graph,
    write_iteration_prompt,
)

GRAPH_TYPES = ("knowledge", "decision", "schema", "media", "all")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=False, text=True, capture_output=True)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_report(report_file: Path, summary: dict[str, object], snapshot: Path) -> str:
    line = (
        f"- {_utc_now()} analyzed `{snapshot.name}` with "
        f"{summary['node_count']} nodes and {summary['edge_count']} edges."
    )
    existing = report_file.read_text(encoding="utf-8") if report_file.exists() else "# doc2graph Progress\n\n"
    report_file.write_text(existing.rstrip() + "\n" + line + "\n", encoding="utf-8")
    return line


def _run_test(command: str | None, cwd: Path) -> TestResult | None:
    if not command:
        return None
    result = subprocess.run(command, cwd=cwd, check=False, text=True, capture_output=True, shell=True)
    output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
    return TestResult(command=command, returncode=result.returncode, output=output)


def _find_codex_binary(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    for candidate in (
        shutil.which("codex"),
        str(Path.home() / ".npm-global" / "bin" / "codex"),
        str(Path.home() / ".local" / "bin" / "codex"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("codex CLI was not found; pass --codex-bin or fix PATH")


def _codex_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        [str(Path.home() / ".npm-global" / "bin"), str(Path.home() / ".local" / "bin"), env.get("PATH", "")]
    )
    return env


def _invoke_codex(action_prompt: str, repo_root: Path, log_file: Path, *, codex_bin: str | None, timeout: int) -> str:
    import tempfile

    resolved = _find_codex_binary(codex_bin)
    output_file = Path(tempfile.mktemp(prefix="doc2graph-codex-", suffix=".txt"))
    args = [
        resolved,
        "--ask-for-approval",
        "never",
        "exec",
        "--ephemeral",
        "--cd",
        str(repo_root),
        "--sandbox",
        "danger-full-access",
        "--output-last-message",
        str(output_file),
        "-",
    ]
    try:
        result = subprocess.run(
            args,
            input=action_prompt,
            cwd=repo_root,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=_codex_env(),
        )
        output = output_file.read_text(encoding="utf-8", errors="replace") if output_file.exists() else ""
        output_file.unlink(missing_ok=True)
        combined = "\n".join(filter(None, [output, result.stdout, result.stderr])).strip()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"\n--- Codex run at {_utc_now()} (exit {result.returncode}) ---\n")
            f.write(f"command: {shlex.join(args)}\n")
            f.write(combined[:8000] + "\n")
        return combined[:4000]
    except Exception as exc:  # pragma: no cover
        message = f"Codex invocation failed: {exc}"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"\n--- Codex error at {_utc_now()} ---\n{message}\n")
        return message


def _commit_and_push(repo_root: Path, message: str, paths: list[Path]) -> bool:
    _run(["git", "config", "user.name", "jw-open"], repo_root)
    _run(["git", "config", "user.email", "176761431+jw-open@users.noreply.github.com"], repo_root)
    _run(["git", "add", *[str(path) for path in paths]], repo_root)
    if _run(["git", "diff", "--cached", "--quiet"], repo_root).returncode == 0:
        return False
    if _run(["git", "commit", "-m", message], repo_root).returncode == 0:
        _run(["git", "push", "origin", "main"], repo_root)
        return True
    return False


def _committable_changed_paths(repo_root: Path) -> list[Path]:
    result = _run(["git", "status", "--porcelain"], repo_root)
    allowed_roots = {"docs2graph", "tests", "examples", "benchmarks"}
    allowed_names = {"README.md", "pyproject.toml", ".gitignore", "LICENSE"}
    blocked_names = {"DOC2GRAPH_PROGRESS.md", "DOC2GRAPH_NEXT_PROMPT.md"}
    paths: list[Path] = []
    for line in result.stdout.splitlines():
        raw_path = line[3:]
        if " -> " in raw_path:
            raw_path = raw_path.split(" -> ", 1)[1]
        path = Path(raw_path)
        if str(path).startswith(".doc2graph-runs/") or path.name in blocked_names:
            continue
        if path.parts and (path.parts[0] in allowed_roots or path.name in allowed_names):
            paths.append(path)
    return paths


def _notify(command: str | None, message: str, cwd: Path) -> None:
    if command:
        subprocess.run(command, input=message, cwd=cwd, check=False, text=True, shell=True)


def _notify_webhook(webhook_url: str | None, message: str) -> None:
    if not webhook_url:
        return
    payload = json.dumps({"content": message[:1900]}).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=10).close()
    except (urllib.error.URLError, TimeoutError):
        return


def run_once(args: argparse.Namespace, repo_root: Path) -> str:
    target = Path(args.path).resolve()
    graph = build_graph(str(target), args.graph)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = target.name or "documents"
    snapshot = Path(args.output_dir) / f"{safe_name}.{args.graph}.{stamp}.json"
    previous_snapshot = find_previous_snapshot(Path(args.output_dir), snapshot, safe_name, args.graph)
    previous_summary = summarize_graph(load_graph(previous_snapshot)) if previous_snapshot else None
    _write_json(snapshot, graph)

    summary = summarize_graph(graph)
    pre_test = _run_test(args.test_command, repo_root)
    prompt = build_iteration_prompt(
        target_path=target,
        graph_type=args.graph,
        snapshot=snapshot,
        summary=summary,
        previous_snapshot=previous_snapshot,
        previous_summary=previous_summary,
        test_result=pre_test,
    )
    prompt_file = Path(args.prompt_file)
    write_iteration_prompt(prompt_file, prompt)
    report_line = _append_report(Path(args.report_file), summary, snapshot)

    codex_summary = "codex disabled"
    committed = False
    post_test: TestResult | None = None
    if args.codex:
        codex_output = _invoke_codex(
            build_action_prompt(prompt),
            repo_root,
            repo_root / ".doc2graph-runs" / "loop.log",
            codex_bin=args.codex_bin,
            timeout=args.codex_timeout_seconds,
        )
        codex_summary = codex_output[:300].replace("\n", " ").strip() if codex_output else "(no output)"
        post_test = _run_test(args.test_command, repo_root)
        if args.commit_push and (post_test is None or post_test.passed):
            committed = _commit_and_push(repo_root, f"Improve doc2graph iteration {stamp}", _committable_changed_paths(repo_root))
    elif args.commit_push:
        committed = _commit_and_push(repo_root, f"chore: record doc2graph iteration {stamp}", [Path(args.report_file), prompt_file])

    status = " committed source changes." if committed else " no source commit created."
    if post_test is not None:
        status += " Post-tests passed." if post_test.passed else f" Post-tests failed: `{post_test.command}`."
    message = f"{report_line} Prompt: `{prompt_file}`. Codex: {codex_summary};{status}"
    _notify(args.report_command, message, repo_root)
    _notify_webhook(args.discord_webhook_url or os.environ.get("DOC2GRAPH_DISCORD_WEBHOOK_URL"), message)
    return message


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run repeated doc2graph development iterations.")
    parser.add_argument("path", help="Document, URL, or directory to analyze for iteration context.")
    parser.add_argument("--graph", default="all", choices=GRAPH_TYPES)
    parser.add_argument("--interval-minutes", type=float, default=20.0)
    parser.add_argument("--iterations", type=int, default=1, help="Number of loops. Use 0 for forever.")
    parser.add_argument("--output-dir", default=".doc2graph-runs")
    parser.add_argument("--report-file", default="DOC2GRAPH_PROGRESS.md")
    parser.add_argument("--prompt-file", default="DOC2GRAPH_NEXT_PROMPT.md")
    parser.add_argument("--report-command")
    parser.add_argument("--discord-webhook-url")
    parser.add_argument("--test-command", default="python -m pytest -q")
    parser.add_argument("--commit-push", action="store_true")
    parser.add_argument("--codex", action="store_true")
    parser.add_argument("--codex-bin")
    parser.add_argument("--codex-timeout-seconds", type=int, default=900)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    count = 0
    while args.iterations == 0 or count < args.iterations:
        try:
            print(run_once(args, repo_root), flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"doc2graph iteration failed: {exc}", file=sys.stderr, flush=True)
            _notify(args.report_command, f"doc2graph iteration failed: {exc}", repo_root)
            return 1
        count += 1
        if args.iterations == 0 or count < args.iterations:
            time.sleep(args.interval_minutes * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
