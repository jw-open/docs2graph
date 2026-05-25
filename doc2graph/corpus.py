"""Corpus and directory graph building for doc2graph."""

from __future__ import annotations

import fnmatch
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .types import make_edge, make_node

SUPPORTED_SUFFIXES = {
    ".md",
    ".markdown",
    ".mdx",
    ".txt",
    ".rst",
    ".html",
    ".htm",
    ".docx",
    ".pptx",
    ".csv",
    ".tsv",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".webp",
}

DEFAULT_IGNORE_PATTERNS = (
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "coverage",
)


def build_corpus_graph(
    path: str,
    graph_type: str = "knowledge",
    *,
    recursive: bool = True,
    max_files: int | None = None,
    max_file_bytes: int | None = 25 * 1024 * 1024,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Build one graph from a file, URL, or directory corpus."""
    from .loaders.url import is_url
    from .cli import build_file_graph, _merge_graphs

    if is_url(path):
        return build_file_graph(path, graph_type)

    root = Path(path)
    if root.is_file():
        return build_file_graph(str(root), graph_type)
    if not root.is_dir():
        raise FileNotFoundError(path)

    files = list(
        iter_document_files(
            root,
            recursive=recursive,
            include=include,
            exclude=exclude,
            max_files=max_files,
        )
    )

    root_id = _id("corpus", str(root.resolve()))
    nodes: List[Dict[str, Any]] = [
        make_node(
            root_id,
            root.name or str(root),
            attributes={
                "type": "corpus",
                "source": str(root),
                "file_count": len(files),
                "recursive": recursive,
                "extraction_method": "static",
            },
        )
    ]
    edges: List[Dict[str, Any]] = []
    seen_folders = {root_id}
    graph_parts: List[Dict[str, Any]] = []

    for file_path in files:
        rel = file_path.relative_to(root).as_posix()
        parent_id = _ensure_folder_nodes(root, file_path.parent, root_id, nodes, edges, seen_folders)
        file_id = _id("file", rel)
        size = _safe_size(file_path)
        attrs = {
            "type": "file",
            "source": str(file_path),
            "relative_path": rel,
            "suffix": file_path.suffix.lower(),
            "size_bytes": size,
            "extraction_method": "static",
        }
        nodes.append(make_node(file_id, file_path.name, attributes=attrs))
        edges.append(make_edge(parent_id, file_id, "contains"))

        if max_file_bytes is not None and size is not None and size > max_file_bytes:
            skipped_id = _id("skipped_file", rel)
            nodes.append(
                make_node(
                    skipped_id,
                    f"Skipped {file_path.name}",
                    attributes={
                        "type": "skipped_file",
                        "source": str(file_path),
                        "relative_path": rel,
                        "reason": "file_too_large",
                        "max_file_bytes": max_file_bytes,
                        "size_bytes": size,
                    },
                )
            )
            edges.append(make_edge(file_id, skipped_id, "skipped"))
            continue

        try:
            graph = build_file_graph(str(file_path), graph_type)
        except Exception as exc:  # keep batch extraction useful on mixed corpora
            error_id = _id("load_error", f"{rel}:{type(exc).__name__}:{exc}")
            nodes.append(
                make_node(
                    error_id,
                    f"{type(exc).__name__}: {file_path.name}",
                    content=str(exc),
                    attributes={
                        "type": "load_error",
                        "source": str(file_path),
                        "relative_path": rel,
                        "error_type": type(exc).__name__,
                        "extraction_method": "static",
                    },
                )
            )
            edges.append(make_edge(file_id, error_id, "failed_to_extract"))
            continue

        graph_root = graph.get("current_node_id")
        if graph_root:
            edges.append(make_edge(file_id, graph_root, "extracted_as"))
        graph_parts.append(graph)

    corpus_graph = {"nodes": nodes, "edges": edges, "current_node_id": root_id}
    return _merge_graphs([corpus_graph, *graph_parts])


def iter_document_files(
    root: Path,
    *,
    recursive: bool = True,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    max_files: int | None = None,
) -> Iterable[Path]:
    """Yield supported document files under ``root`` in deterministic order."""
    patterns = tuple(include or ())
    excludes = tuple(DEFAULT_IGNORE_PATTERNS) + tuple(exclude or ())
    candidates = root.rglob("*") if recursive else root.glob("*")
    count = 0
    for path in sorted(candidates, key=lambda p: p.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix()
        if _is_ignored(path, rel, excludes):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if patterns and not any(fnmatch.fnmatch(rel, pattern) for pattern in patterns):
            continue
        yield path
        count += 1
        if max_files is not None and count >= max_files:
            break


def _ensure_folder_nodes(
    root: Path,
    folder: Path,
    root_id: str,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    seen: set[str],
) -> str:
    if folder == root:
        return root_id

    parts = folder.relative_to(root).parts
    parent_id = root_id
    current = root
    for part in parts:
        current = current / part
        rel = current.relative_to(root).as_posix()
        folder_id = _id("folder", rel)
        if folder_id not in seen:
            nodes.append(
                make_node(
                    folder_id,
                    part,
                    attributes={
                        "type": "folder",
                        "source": str(current),
                        "relative_path": rel,
                        "extraction_method": "static",
                    },
                )
            )
            edges.append(make_edge(parent_id, folder_id, "contains"))
            seen.add(folder_id)
        parent_id = folder_id
    return parent_id


def _is_ignored(path: Path, rel: str, patterns: Sequence[str]) -> bool:
    parts = set(path.parts)
    for pattern in patterns:
        if pattern in parts or fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern):
            return True
    return False


def _safe_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:16]
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)[:80].strip("_")
    return f"{prefix}:{slug or digest}:{digest}"
