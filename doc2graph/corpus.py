"""Corpus and directory graph building for doc2graph."""

from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .types import make_edge, make_node
from .loaders.code import CODE_SUFFIXES

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
} | CODE_SUFFIXES

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
    max_depth: int | None = None,
    max_file_bytes: int | None = 25 * 1024 * 1024,
    max_total_bytes: int | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    skip_report_limit: int = 100,
    cache_path: str | Path | None = None,
    output_path: str | Path | None = None,
    refresh_cache: bool = False,
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

    reserved_paths = [
        _resolved_path(reserved)
        for reserved in (cache_path, output_path)
        if reserved is not None
    ]
    scan = scan_document_files(
        root,
        recursive=recursive,
        include=include,
        exclude=exclude,
        max_files=max_files,
        max_depth=max_depth,
        skip_report_limit=skip_report_limit,
        reserved_paths=reserved_paths,
    )
    files = scan.files
    cache = _load_cache(cache_path) if cache_path is not None else _empty_cache()
    cache_stats = {
        "enabled": cache_path is not None,
        "path": str(cache_path) if cache_path is not None else None,
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "pruned": 0,
        "refresh": refresh_cache,
    }

    root_id = _id("corpus", str(root.resolve()))
    manifest_id = _id("corpus_manifest", str(root.resolve()))
    manifest_attrs: Dict[str, Any] = {
        "type": "corpus_manifest",
        "source": str(root),
        "selected_file_count": len(files),
        "skipped_file_count": scan.skipped_count,
        "skipped_by_reason": dict(sorted(scan.skipped_by_reason.items())),
        "include_patterns": list(include or ()),
        "exclude_patterns": list(exclude or ()),
        "skip_report_limit": skip_report_limit,
        "max_files": max_files,
        "max_files_reached": scan.skipped_by_reason.get("max_files_exceeded", 0) > 0,
        "max_depth": max_depth,
        "max_depth_reached": scan.skipped_by_reason.get("max_depth_exceeded", 0) > 0,
        "max_file_bytes": max_file_bytes,
        "max_total_bytes": max_total_bytes,
        "max_total_bytes_reached": False,
        "extracted_file_count": 0,
        "extracted_total_bytes": 0,
        "recursive": recursive,
        "extraction_method": "static",
        "cache_enabled": cache_stats["enabled"],
        "cache_path": cache_stats["path"],
        "cache_hits": cache_stats["hits"],
        "cache_misses": cache_stats["misses"],
        "cache_writes": cache_stats["writes"],
        "cache_pruned": cache_stats["pruned"],
        "cache_refresh": cache_stats["refresh"],
    }
    nodes: List[Dict[str, Any]] = [
        make_node(
            root_id,
            root.name or str(root),
            attributes={
                "type": "corpus",
                "source": str(root),
                "file_count": len(files),
                "skipped_file_count": scan.skipped_count,
                "recursive": recursive,
                "extraction_method": "static",
            },
        ),
        make_node(manifest_id, "Corpus extraction manifest", attributes=manifest_attrs),
    ]
    edges: List[Dict[str, Any]] = [make_edge(root_id, manifest_id, "contains")]
    seen_folders = {root_id}
    graph_parts: List[Dict[str, Any]] = []

    for skipped in scan.skipped_samples:
        skipped_id = _id("skipped_file", f"{skipped.reason}:{skipped.relative_path}")
        nodes.append(
            make_node(
                skipped_id,
                f"Skipped {skipped.relative_path}",
                attributes={
                    "type": "skipped_file",
                    "source": str(skipped.path),
                    "relative_path": skipped.relative_path,
                    "path_type": skipped.path_type,
                    "suffix": skipped.path.suffix.lower(),
                    "reason": skipped.reason,
                    "extraction_method": "static",
                },
            )
        )
        edges.append(make_edge(manifest_id, skipped_id, "skipped"))

    runtime_skipped_by_reason: Dict[str, int] = {}
    runtime_reported_skips = 0
    extracted_file_count = 0
    extracted_total_bytes = 0
    total_report_limit = max(0, skip_report_limit)
    budget_exhausted = False
    active_cache_keys: set[str] = set()

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

        if budget_exhausted:
            runtime_reported_skips += _add_runtime_skip(
                nodes,
                edges,
                file_id,
                file_path,
                rel,
                "max_total_bytes_exceeded",
                runtime_skipped_by_reason,
                total_report_limit=total_report_limit,
                existing_reported_count=len(scan.skipped_samples) + runtime_reported_skips,
                attributes={
                    "max_total_bytes": max_total_bytes,
                    "current_total_bytes": extracted_total_bytes,
                    "size_bytes": size,
                },
            )
            continue

        if max_file_bytes is not None and size is not None and size > max_file_bytes:
            runtime_reported_skips += _add_runtime_skip(
                nodes,
                edges,
                file_id,
                file_path,
                rel,
                "file_too_large",
                runtime_skipped_by_reason,
                total_report_limit=total_report_limit,
                existing_reported_count=len(scan.skipped_samples) + runtime_reported_skips,
                attributes={"max_file_bytes": max_file_bytes, "size_bytes": size},
            )
            continue

        if (
            max_total_bytes is not None
            and size is not None
            and extracted_total_bytes + size > max_total_bytes
        ):
            budget_exhausted = True
            runtime_reported_skips += _add_runtime_skip(
                nodes,
                edges,
                file_id,
                file_path,
                rel,
                "max_total_bytes_exceeded",
                runtime_skipped_by_reason,
                total_report_limit=total_report_limit,
                existing_reported_count=len(scan.skipped_samples) + runtime_reported_skips,
                attributes={
                    "max_total_bytes": max_total_bytes,
                    "current_total_bytes": extracted_total_bytes,
                    "size_bytes": size,
                },
            )
            continue

        try:
            graph = None
            cache_key = _cache_key(root, file_path, graph_type)
            metadata = _file_metadata(root, file_path, graph_type)
            if cache_path is not None and not refresh_cache:
                cached = cache.get("entries", {}).get(cache_key)
                if (
                    isinstance(cached, dict)
                    and cached.get("metadata") == metadata
                    and isinstance(cached.get("graph"), dict)
                ):
                    graph = cached.get("graph")
                    cache_stats["hits"] += 1

            if graph is None:
                if cache_path is not None:
                    cache_stats["misses"] += 1
                graph = build_file_graph(str(file_path), graph_type)
                if cache_path is not None:
                    cache.setdefault("entries", {})[cache_key] = {
                        "metadata": metadata,
                        "graph": graph,
                    }
                    cache_stats["writes"] += 1
            active_cache_keys.add(cache_key)
        except Exception as exc:  # keep batch extraction useful on mixed corpora
            _count_skip(runtime_skipped_by_reason, "load_error")
            if len(scan.skipped_samples) + runtime_reported_skips < total_report_limit:
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
                runtime_reported_skips += 1
            continue

        graph_root = graph.get("current_node_id")
        if graph_root:
            edges.append(make_edge(file_id, graph_root, "extracted_as"))
        graph_parts.append(graph)
        extracted_file_count += 1
        if size is not None:
            extracted_total_bytes += size

    corpus_graph = {"nodes": nodes, "edges": edges, "current_node_id": root_id}
    for reason, count in runtime_skipped_by_reason.items():
        scan.skipped_by_reason[reason] = scan.skipped_by_reason.get(reason, 0) + count
        scan.skipped_count += count
    manifest_attrs["skipped_file_count"] = scan.skipped_count
    manifest_attrs["skipped_by_reason"] = dict(sorted(scan.skipped_by_reason.items()))
    manifest_attrs["reported_skipped_file_count"] = len(scan.skipped_samples) + runtime_reported_skips
    manifest_attrs["max_total_bytes_reached"] = (
        scan.skipped_by_reason.get("max_total_bytes_exceeded", 0) > 0
    )
    manifest_attrs["extracted_file_count"] = extracted_file_count
    manifest_attrs["extracted_total_bytes"] = extracted_total_bytes
    manifest_attrs["cache_hits"] = cache_stats["hits"]
    manifest_attrs["cache_misses"] = cache_stats["misses"]
    manifest_attrs["cache_writes"] = cache_stats["writes"]
    nodes[0]["attributes"]["skipped_file_count"] = scan.skipped_count
    nodes[0]["attributes"]["extracted_file_count"] = extracted_file_count
    nodes[0]["attributes"]["extracted_total_bytes"] = extracted_total_bytes
    if cache_path is not None:
        cache_stats["pruned"] = _prune_cache(cache, root, graph_type, active_cache_keys)
        manifest_attrs["cache_pruned"] = cache_stats["pruned"]
        _write_cache(cache_path, cache)
    return _merge_graphs([corpus_graph, *graph_parts])


@dataclass(frozen=True)
class SkippedFile:
    path: Path
    relative_path: str
    reason: str
    path_type: str = "file"


@dataclass
class CorpusScan:
    files: List[Path] = field(default_factory=list)
    skipped_by_reason: Dict[str, int] = field(default_factory=dict)
    skipped_count: int = 0
    skipped_samples: List[SkippedFile] = field(default_factory=list)


def scan_document_files(
    root: Path,
    *,
    recursive: bool = True,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    max_files: int | None = None,
    max_depth: int | None = None,
    skip_report_limit: int = 100,
    reserved_paths: Sequence[Path] | None = None,
) -> CorpusScan:
    """Scan ``root`` for supported documents and bounded skipped-file metadata."""
    patterns = tuple(include or ())
    excludes = tuple(DEFAULT_IGNORE_PATTERNS) + tuple(exclude or ())
    reserved = frozenset(_resolved_path(path) for path in reserved_paths or ())
    result = CorpusScan()
    report_limit = max(0, skip_report_limit)

    for path in _iter_candidate_files(
        root,
        recursive=recursive,
        excludes=excludes,
        max_depth=max_depth,
        scan=result,
        report_limit=report_limit,
    ):
        rel = path.relative_to(root).as_posix()
        if _resolved_path(path) in reserved:
            _record_skipped(result, path, rel, "reserved_output_file", report_limit)
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            _record_skipped(result, path, rel, "unsupported_extension", report_limit)
            continue
        if patterns and not any(fnmatch.fnmatch(rel, pattern) for pattern in patterns):
            _record_skipped(result, path, rel, "include_filter_mismatch", report_limit)
            continue
        if max_files is not None and len(result.files) >= max_files:
            _record_skipped(result, path, rel, "max_files_exceeded", report_limit)
            continue
        result.files.append(path)

    return result


def iter_document_files(
    root: Path,
    *,
    recursive: bool = True,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    max_files: int | None = None,
    max_depth: int | None = None,
) -> Iterable[Path]:
    """Yield supported document files under ``root`` in deterministic order."""
    scan = scan_document_files(
        root,
        recursive=recursive,
        include=include,
        exclude=exclude,
        max_files=max_files,
        max_depth=max_depth,
        skip_report_limit=0,
    )
    yield from scan.files


def _iter_candidate_files(
    root: Path,
    *,
    recursive: bool,
    excludes: Sequence[str],
    max_depth: int | None = None,
    scan: CorpusScan | None = None,
    report_limit: int = 0,
) -> Iterable[Path]:
    """Yield non-ignored files in deterministic order, pruning ignored directories."""
    try:
        entries = sorted(root.iterdir(), key=lambda p: p.name)
    except OSError:
        if scan is not None:
            _record_skipped(scan, root, ".", "directory_inaccessible", report_limit, path_type="directory")
        return

    for path in entries:
        rel = path.relative_to(root).as_posix()
        if _is_ignored(path, rel, excludes):
            continue
        path_kind = _path_kind(path)
        if path_kind == "symlink_directory":
            if recursive and scan is not None:
                _record_skipped(scan, path, rel, "symlink_directory", report_limit, path_type="directory")
            continue
        if path_kind == "directory":
            if recursive:
                if max_depth is not None and _relative_depth(root, path) > max_depth:
                    if scan is not None:
                        _record_skipped(
                            scan,
                            path,
                            rel,
                            "max_depth_exceeded",
                            report_limit,
                            path_type="directory",
                        )
                    continue
                yield from _iter_candidate_files_for_child(
                    root,
                    path,
                    excludes,
                    max_depth=max_depth,
                    scan=scan,
                    report_limit=report_limit,
                )
            continue
        if path_kind == "file":
            yield path
        elif path_kind == "inaccessible" and scan is not None:
            _record_skipped(scan, path, rel, "path_inaccessible", report_limit)
        elif path_kind == "broken_symlink" and scan is not None:
            _record_skipped(scan, path, rel, "broken_symlink", report_limit)


def _iter_candidate_files_for_child(
    root: Path,
    folder: Path,
    excludes: Sequence[str],
    *,
    max_depth: int | None = None,
    scan: CorpusScan | None = None,
    report_limit: int = 0,
) -> Iterable[Path]:
    try:
        entries = sorted(folder.iterdir(), key=lambda p: p.name)
    except OSError:
        if scan is not None:
            rel = folder.relative_to(root).as_posix()
            _record_skipped(scan, folder, rel, "directory_inaccessible", report_limit, path_type="directory")
        return

    for path in entries:
        rel = path.relative_to(root).as_posix()
        if _is_ignored(path, rel, excludes):
            continue
        path_kind = _path_kind(path)
        if path_kind == "symlink_directory":
            if scan is not None:
                _record_skipped(scan, path, rel, "symlink_directory", report_limit, path_type="directory")
            continue
        if path_kind == "directory":
            if max_depth is not None and _relative_depth(root, path) > max_depth:
                if scan is not None:
                    _record_skipped(
                        scan,
                        path,
                        rel,
                        "max_depth_exceeded",
                        report_limit,
                        path_type="directory",
                    )
                continue
            yield from _iter_candidate_files_for_child(
                root,
                path,
                excludes,
                max_depth=max_depth,
                scan=scan,
                report_limit=report_limit,
            )
        elif path_kind == "file":
            yield path
        elif path_kind == "inaccessible" and scan is not None:
            _record_skipped(scan, path, rel, "path_inaccessible", report_limit)
        elif path_kind == "broken_symlink" and scan is not None:
            _record_skipped(scan, path, rel, "broken_symlink", report_limit)


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


def _path_kind(path: Path) -> str:
    try:
        if path.is_symlink():
            if path.exists():
                return "symlink_directory" if path.is_dir() else "file"
            return "broken_symlink"
        if path.is_dir():
            return "directory"
        if path.is_file():
            return "file"
    except OSError:
        return "inaccessible"
    return "other"


def _safe_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _resolved_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _relative_depth(root: Path, path: Path) -> int:
    return len(path.relative_to(root).parts)


def _file_metadata(root: Path, path: Path, graph_type: str) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "root": str(root.resolve()),
        "relative_path": path.relative_to(root).as_posix(),
        "graph_type": graph_type,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "suffix": path.suffix.lower(),
    }


def _cache_key(root: Path, path: Path, graph_type: str) -> str:
    rel = path.relative_to(root).as_posix()
    source = f"{root.resolve()}\0{rel}\0{graph_type}".encode("utf-8")
    digest = hashlib.sha1(source).hexdigest()
    return digest


def _empty_cache() -> Dict[str, Any]:
    return {"version": 1, "entries": {}}


def _load_cache(cache_path: str | Path | None) -> Dict[str, Any]:
    if cache_path is None:
        return _empty_cache()
    path = Path(cache_path)
    if not path.exists():
        return _empty_cache()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_cache()
    if payload.get("version") != 1 or not isinstance(payload.get("entries"), dict):
        return _empty_cache()
    return payload


def _write_cache(cache_path: str | Path, cache: Dict[str, Any]) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp")
    temp.write_text(
        json.dumps(cache, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _prune_cache(
    cache: Dict[str, Any],
    root: Path,
    graph_type: str,
    active_keys: set[str],
) -> int:
    """Remove stale cache entries for this corpus root and graph type."""
    entries = cache.get("entries")
    if not isinstance(entries, dict):
        cache["entries"] = {}
        return 0

    root_value = str(root.resolve())
    stale_keys = []
    for key, entry in entries.items():
        if key in active_keys or not isinstance(entry, dict):
            continue
        metadata = entry.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if metadata.get("root") == root_value and metadata.get("graph_type") == graph_type:
            stale_keys.append(key)

    for key in stale_keys:
        del entries[key]
    return len(stale_keys)


def _add_runtime_skip(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    file_id: str,
    file_path: Path,
    rel: str,
    reason: str,
    counts: Dict[str, int],
    *,
    total_report_limit: int,
    existing_reported_count: int,
    attributes: Dict[str, Any] | None = None,
) -> int:
    _count_skip(counts, reason)
    if existing_reported_count >= total_report_limit:
        return 0
    skipped_id = _id("skipped_file", f"{reason}:{rel}")
    skip_attrs = {
        "type": "skipped_file",
        "source": str(file_path),
        "relative_path": rel,
        "reason": reason,
        "extraction_method": "static",
    }
    if attributes:
        skip_attrs.update(attributes)
    nodes.append(make_node(skipped_id, f"Skipped {file_path.name}", attributes=skip_attrs))
    edges.append(make_edge(file_id, skipped_id, "skipped"))
    return 1


def _record_skipped(
    result: CorpusScan,
    path: Path,
    rel: str,
    reason: str,
    report_limit: int,
    *,
    path_type: str = "file",
) -> None:
    _count_skip(result.skipped_by_reason, reason)
    result.skipped_count += 1
    if len(result.skipped_samples) < report_limit:
        result.skipped_samples.append(
            SkippedFile(path=path, relative_path=rel, reason=reason, path_type=path_type)
        )


def _count_skip(counts: Dict[str, int], reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1


def _id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:16]
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)[:80].strip("_")
    return f"{prefix}:{slug or digest}:{digest}"
