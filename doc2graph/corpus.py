"""Corpus and directory graph building for doc2graph."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
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

_CORPUS_LINK_SOURCE_TYPES = {
    "document",
    "decision_document",
    "media_document",
    "section",
    "problem",
    "context",
    "option",
    "pros",
    "cons",
    "tradeoff",
    "decision",
    "consequence",
    "confidence",
    "definition",
    "claim",
    "evidence",
    "ocr_text",
}

_CORPUS_LINK_TARGET_TYPES = {
    "document",
    "decision_document",
    "media_document",
    "section",
    "problem",
    "context",
    "option",
    "tradeoff",
    "decision",
    "consequence",
    "confidence",
    "definition",
    "table",
}

_GENERIC_CROSS_DOC_ALIASES = {
    "abstract",
    "background",
    "conclusion",
    "context",
    "decision",
    "document",
    "introduction",
    "method",
    "methodology",
    "overview",
    "problem",
    "references",
    "results",
    "summary",
}

DEFAULT_MAX_FILE_BYTES = 25 * 1024 * 1024

DEFAULT_IGNORE_PATTERNS = (
    ".git",
    ".hg",
    ".svn",
    ".doc2graph-runs",
    ".doc2graph-cache.json",
    "DOC2GRAPH_PROGRESS.md",
    "DOC2GRAPH_NEXT_PROMPT.md",
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
    "*.egg-info",
    ".next",
    ".nuxt",
    "coverage",
)

EXTRACTION_FINGERPRINT_PATHS = (
    "cli.py",
    "types.py",
    "loaders/auto.py",
    "loaders/code.py",
    "loaders/csv.py",
    "loaders/docx.py",
    "loaders/google.py",
    "loaders/html.py",
    "loaders/markdown.py",
    "loaders/ocr.py",
    "loaders/pdf.py",
    "loaders/pptx.py",
    "loaders/text.py",
    "loaders/url.py",
)

EXTRACTION_FINGERPRINT_BY_GRAPH_TYPE = {
    "knowledge": ("extractors/knowledge.py", "extractors/links.py"),
    "decision": ("extractors/decision.py",),
    "schema": ("extractors/schema.py",),
    "media": ("extractors/media.py",),
    "all": (
        "extractors/knowledge.py",
        "extractors/links.py",
        "extractors/decision.py",
        "extractors/schema.py",
        "extractors/media.py",
    ),
}


def build_corpus_graph(
    path: str,
    graph_type: str = "knowledge",
    *,
    recursive: bool = True,
    max_files: int | None = None,
    stop_after_max_files: bool = False,
    max_depth: int | None = None,
    max_scan_entries: int | None = None,
    max_file_bytes: int | None = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    skip_report_limit: int = 100,
    cache_path: str | Path | None = None,
    output_path: str | Path | None = None,
    refresh_cache: bool = False,
    max_cross_document_links: int | None = None,
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

    if max_cross_document_links is not None and max_cross_document_links < 0:
        max_cross_document_links = None

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
        stop_after_max_files=stop_after_max_files,
        max_depth=max_depth,
        max_scan_entries=max_scan_entries,
        skip_report_limit=skip_report_limit,
        reserved_paths=reserved_paths,
    )
    files = scan.files
    selected_file_sizes: Dict[Path, int | None] = {}
    cache_load = (
        _load_cache_with_status(cache_path)
        if cache_path is not None
        else CacheLoad(cache=_empty_cache(), status="disabled", entry_count=0)
    )
    cache = cache_load.cache
    cache_stats = {
        "enabled": cache_path is not None,
        "path": str(cache_path) if cache_path is not None else None,
        "load_status": cache_load.status,
        "entry_count_before": cache_load.entry_count,
        "entry_count_after": cache_load.entry_count,
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "content_digest_hits": 0,
        "content_digest_misses": 0,
        "pruned": 0,
        "refresh": refresh_cache,
        "write_status": None,
        "write_error": None,
    }

    root_id = _id("corpus", str(root.resolve()))
    manifest_id = _id("corpus_manifest", str(root.resolve()))
    manifest_attrs: Dict[str, Any] = {
        "type": "corpus_manifest",
        "source": str(root),
        "selected_file_count": len(files),
        "selected_file_ordering": "relative_path_depth_first",
        "selected_file_paths_sha256": _paths_sha256(
            file_path.relative_to(root).as_posix() for file_path in files
        ),
        "selected_file_records_sha256": None,
        "selected_file_records_sha256_is_complete": not scan.scan_truncated,
        "selected_total_bytes": 0,
        "selected_total_bytes_is_complete": True,
        "skipped_file_count": scan.skipped_count,
        "skipped_by_reason": dict(sorted(scan.skipped_by_reason.items())),
        "skipped_file_records_sha256": scan.skipped_records_digest.hexdigest(),
        "skipped_file_records_sha256_is_complete": not scan.scan_truncated,
        "reported_skipped_file_count": len(scan.skipped_samples),
        "unreported_skipped_file_count": max(
            0,
            scan.skipped_count - len(scan.skipped_samples),
        ),
        "skip_report_truncated": scan.skipped_count > len(scan.skipped_samples),
        "include_patterns": list(include or ()),
        "exclude_patterns": list(exclude or ()),
        "skip_report_limit": skip_report_limit,
        "max_files": max_files,
        "stop_after_max_files": stop_after_max_files,
        "max_files_reached": scan.skipped_by_reason.get("max_files_exceeded", 0) > 0,
        "max_files_scan_truncated": scan.scan_truncated_reason == "max_files",
        "max_depth": max_depth,
        "max_depth_reached": scan.skipped_by_reason.get("max_depth_exceeded", 0) > 0,
        "max_scan_entries": max_scan_entries,
        "max_scan_entries_reached": scan.scan_truncated_reason == "max_scan_entries",
        "scanned_entry_count": scan.scanned_entry_count,
        "skipped_file_count_is_complete": not scan.scan_truncated,
        "max_file_bytes": max_file_bytes,
        "max_total_bytes": max_total_bytes,
        "max_total_bytes_reached": False,
        "extracted_file_count": 0,
        "extracted_total_bytes": 0,
        "recursive": recursive,
        "extraction_method": "static",
        "cache_enabled": cache_stats["enabled"],
        "cache_path": cache_stats["path"],
        "cache_load_status": cache_stats["load_status"]
        if cache_stats["enabled"]
        else None,
        "cache_entry_count_before": cache_stats["entry_count_before"]
        if cache_stats["enabled"]
        else None,
        "cache_entry_count_after": cache_stats["entry_count_after"]
        if cache_stats["enabled"]
        else None,
        "cache_validation": "content_sha256" if cache_stats["enabled"] else None,
        "cache_extraction_fingerprint": _extraction_fingerprint(graph_type)
        if cache_stats["enabled"]
        else None,
        "cache_hits": cache_stats["hits"],
        "cache_misses": cache_stats["misses"],
        "cache_writes": cache_stats["writes"],
        "cache_content_digest_hits": cache_stats["content_digest_hits"],
        "cache_content_digest_misses": cache_stats["content_digest_misses"],
        "cache_pruned": cache_stats["pruned"],
        "cache_prune_status": None,
        "cache_file_updated": False,
        "cache_write_status": None,
        "cache_write_error": None,
        "cache_refresh": cache_stats["refresh"],
        "max_cross_document_links": max_cross_document_links,
        "cross_document_link_count": 0,
        "cross_document_link_limit_reached": False,
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
                "skipped_file_count_is_complete": not scan.scan_truncated,
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
    failed_file_count = 0
    extracted_total_bytes = 0
    total_report_limit = max(0, skip_report_limit)
    budget_exhausted = False
    active_cache_keys: set[str] = set()

    for extraction_order, file_path in enumerate(files):
        rel = file_path.relative_to(root).as_posix()
        parent_id = _ensure_folder_nodes(root, file_path.parent, root_id, nodes, edges, seen_folders)
        file_id = _id("file", rel)
        size = _safe_size(file_path)
        selected_file_sizes[file_path] = size
        attrs = {
            "type": "file",
            "source": str(file_path),
            "relative_path": rel,
            "extraction_order": extraction_order,
            "suffix": file_path.suffix.lower(),
            "size_bytes": size,
            "status": "pending",
            "cache_status": "not_attempted",
            "extraction_method": "static",
        }
        nodes.append(make_node(file_id, file_path.name, attributes=attrs))
        edges.append(make_edge(parent_id, file_id, "contains"))

        if budget_exhausted:
            attrs["status"] = "skipped"
            attrs["skip_reason"] = "max_total_bytes_exceeded"
            runtime_reported_skips += _add_runtime_skip(
                nodes,
                edges,
                file_id,
                file_path,
                rel,
                "max_total_bytes_exceeded",
                runtime_skipped_by_reason,
                scan.skipped_records_digest,
                total_report_limit=total_report_limit,
                existing_reported_count=len(scan.skipped_samples) + runtime_reported_skips,
                attributes={
                    "path_type": "file",
                    "max_total_bytes": max_total_bytes,
                    "current_total_bytes": extracted_total_bytes,
                    "size_bytes": size,
                },
            )
            continue

        if max_file_bytes is not None and size is not None and size > max_file_bytes:
            attrs["status"] = "skipped"
            attrs["skip_reason"] = "file_too_large"
            runtime_reported_skips += _add_runtime_skip(
                nodes,
                edges,
                file_id,
                file_path,
                rel,
                "file_too_large",
                runtime_skipped_by_reason,
                scan.skipped_records_digest,
                total_report_limit=total_report_limit,
                existing_reported_count=len(scan.skipped_samples) + runtime_reported_skips,
                attributes={
                    "path_type": "file",
                    "max_file_bytes": max_file_bytes,
                    "size_bytes": size,
                },
            )
            continue

        if (
            max_total_bytes is not None
            and size is not None
            and extracted_total_bytes + size > max_total_bytes
        ):
            budget_exhausted = True
            attrs["status"] = "skipped"
            attrs["skip_reason"] = "max_total_bytes_exceeded"
            runtime_reported_skips += _add_runtime_skip(
                nodes,
                edges,
                file_id,
                file_path,
                rel,
                "max_total_bytes_exceeded",
                runtime_skipped_by_reason,
                scan.skipped_records_digest,
                total_report_limit=total_report_limit,
                existing_reported_count=len(scan.skipped_samples) + runtime_reported_skips,
                attributes={
                    "path_type": "file",
                    "max_total_bytes": max_total_bytes,
                    "current_total_bytes": extracted_total_bytes,
                    "size_bytes": size,
                },
            )
            continue

        try:
            graph = None
            cache_key = _cache_key(root, file_path, graph_type)
            cached = (
                cache.get("entries", {}).get(cache_key)
                if isinstance(cache.get("entries"), dict)
                else None
            )
            cached_metadata = (
                cached.get("metadata")
                if isinstance(cached, dict) and isinstance(cached.get("metadata"), dict)
                else None
            )
            metadata = _file_metadata(
                root,
                file_path,
                graph_type,
                cached_metadata=cached_metadata,
            )
            content_sha256_reused = bool(metadata.pop("content_sha256_reused", False))
            if cache_path is not None:
                if content_sha256_reused:
                    cache_stats["content_digest_hits"] += 1
                else:
                    cache_stats["content_digest_misses"] += 1
            if cache_path is not None and not refresh_cache:
                if (
                    isinstance(cached, dict)
                    and cached.get("metadata") == metadata
                    and isinstance(cached.get("graph"), dict)
                ):
                    graph = cached.get("graph")
                    cache_stats["hits"] += 1
                    attrs["cache_status"] = "hit"

            if graph is None:
                if cache_path is not None:
                    cache_stats["misses"] += 1
                    attrs["cache_status"] = "refresh" if refresh_cache else "miss"
                else:
                    attrs["cache_status"] = "disabled"
                graph = build_file_graph(str(file_path), graph_type)
                if cache_path is not None:
                    cache.setdefault("entries", {})[cache_key] = {
                        "metadata": metadata,
                        "graph": graph,
                    }
                    cache_stats["writes"] += 1
            active_cache_keys.add(cache_key)
        except Exception as exc:  # keep batch extraction useful on mixed corpora
            attrs["status"] = "failed"
            attrs["error_type"] = type(exc).__name__
            attrs["error_message"] = str(exc)
            _count_skip(runtime_skipped_by_reason, "load_error")
            _update_skip_digest(scan.skipped_records_digest, rel, "load_error", "file")
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
                            "suffix": file_path.suffix.lower(),
                            "size_bytes": size,
                            "error_type": type(exc).__name__,
                            "extraction_method": "static",
                        },
                    )
                )
                edges.append(make_edge(file_id, error_id, "failed_to_extract"))
                runtime_reported_skips += 1
            failed_file_count += 1
            continue

        graph_root = graph.get("current_node_id")
        if graph_root:
            edges.append(make_edge(file_id, graph_root, "extracted_as"))
        graph_parts.append(graph)
        attrs["status"] = "extracted"
        extracted_file_count += 1
        if size is not None:
            extracted_total_bytes += size

    corpus_graph = {"nodes": nodes, "edges": edges, "current_node_id": root_id}
    for reason, count in runtime_skipped_by_reason.items():
        scan.skipped_by_reason[reason] = scan.skipped_by_reason.get(reason, 0) + count
        scan.skipped_count += count
    manifest_attrs["skipped_file_count"] = scan.skipped_count
    manifest_attrs["skipped_file_count_is_complete"] = not scan.scan_truncated
    manifest_attrs["skipped_by_reason"] = dict(sorted(scan.skipped_by_reason.items()))
    manifest_attrs["skipped_file_records_sha256"] = scan.skipped_records_digest.hexdigest()
    manifest_attrs["skipped_file_records_sha256_is_complete"] = not scan.scan_truncated
    manifest_attrs["selected_file_records_sha256"] = _selected_file_records_sha256(
        root,
        files,
        selected_file_sizes,
    )
    manifest_attrs["selected_file_records_sha256_is_complete"] = not scan.scan_truncated
    manifest_attrs["selected_total_bytes"] = sum(
        size for size in selected_file_sizes.values() if size is not None
    )
    manifest_attrs["selected_total_bytes_is_complete"] = all(
        size is not None for size in selected_file_sizes.values()
    )
    manifest_attrs["reported_skipped_file_count"] = len(scan.skipped_samples) + runtime_reported_skips
    manifest_attrs["unreported_skipped_file_count"] = max(
        0,
        scan.skipped_count - manifest_attrs["reported_skipped_file_count"],
    )
    manifest_attrs["skip_report_truncated"] = (
        manifest_attrs["unreported_skipped_file_count"] > 0
    )
    manifest_attrs["max_scan_entries_reached"] = (
        scan.scan_truncated_reason == "max_scan_entries"
    )
    manifest_attrs["max_files_scan_truncated"] = scan.scan_truncated_reason == "max_files"
    manifest_attrs["scanned_entry_count"] = scan.scanned_entry_count
    manifest_attrs["max_total_bytes_reached"] = (
        scan.skipped_by_reason.get("max_total_bytes_exceeded", 0) > 0
    )
    manifest_attrs["extracted_file_count"] = extracted_file_count
    manifest_attrs["failed_file_count"] = failed_file_count
    manifest_attrs["extracted_total_bytes"] = extracted_total_bytes
    manifest_attrs["cache_hits"] = cache_stats["hits"]
    manifest_attrs["cache_misses"] = cache_stats["misses"]
    manifest_attrs["cache_writes"] = cache_stats["writes"]
    manifest_attrs["cache_content_digest_hits"] = cache_stats["content_digest_hits"]
    manifest_attrs["cache_content_digest_misses"] = cache_stats["content_digest_misses"]
    nodes[0]["attributes"]["skipped_file_count"] = scan.skipped_count
    nodes[0]["attributes"]["skipped_file_count_is_complete"] = not scan.scan_truncated
    nodes[0]["attributes"]["extracted_file_count"] = extracted_file_count
    nodes[0]["attributes"]["failed_file_count"] = failed_file_count
    nodes[0]["attributes"]["extracted_total_bytes"] = extracted_total_bytes
    if cache_path is not None:
        prune_status = _cache_prune_status(
            scan,
            recursive=recursive,
            max_files=max_files,
            max_depth=max_depth,
            max_scan_entries=max_scan_entries,
            max_file_bytes=max_file_bytes,
            max_total_bytes=max_total_bytes,
        )
        cache_stats["pruned"] = (
            _prune_cache(cache, root, graph_type, active_cache_keys)
            if prune_status == "complete_selection"
            else 0
        )
        manifest_attrs["cache_pruned"] = cache_stats["pruned"]
        manifest_attrs["cache_prune_status"] = prune_status
        cache_stats["entry_count_after"] = _cache_entry_count(cache)
        manifest_attrs["cache_entry_count_after"] = cache_stats["entry_count_after"]
        cache_write = _write_cache_with_status(cache_path, cache)
        cache_stats["write_status"] = cache_write.status
        cache_stats["write_error"] = cache_write.error_message
        manifest_attrs["cache_file_updated"] = cache_write.updated
        manifest_attrs["cache_write_status"] = cache_write.status
        manifest_attrs["cache_write_error"] = cache_write.error_message
    merged = _merge_graphs([corpus_graph, *graph_parts])
    cross_document_links = _add_corpus_cross_document_links(
        merged,
        max_links=max_cross_document_links,
    )
    manifest_attrs["cross_document_link_count"] = cross_document_links.added
    manifest_attrs["cross_document_link_limit_reached"] = cross_document_links.limit_reached
    nodes[0]["attributes"]["cross_document_link_count"] = cross_document_links.added
    nodes[0]["attributes"]["cross_document_link_limit_reached"] = (
        cross_document_links.limit_reached
    )
    return merged


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
    scanned_entry_count: int = 0
    scan_truncated: bool = False
    scan_truncated_reason: str | None = None
    skipped_records_digest: Any = field(default_factory=hashlib.sha256)


@dataclass(frozen=True)
class CacheLoad:
    cache: Dict[str, Any]
    status: str
    entry_count: int


@dataclass(frozen=True)
class CacheWrite:
    updated: bool
    status: str
    error_message: str | None = None


@dataclass(frozen=True)
class CrossDocumentLinkResult:
    added: int
    limit_reached: bool = False


def scan_document_files(
    root: Path,
    *,
    recursive: bool = True,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    max_files: int | None = None,
    stop_after_max_files: bool = False,
    max_depth: int | None = None,
    max_scan_entries: int | None = None,
    skip_report_limit: int = 100,
    reserved_paths: Sequence[Path] | None = None,
) -> CorpusScan:
    """Scan ``root`` for supported documents and bounded skipped-file metadata."""
    patterns = tuple(include or ())
    default_excludes = tuple(DEFAULT_IGNORE_PATTERNS)
    user_excludes = tuple(exclude or ())
    reserved = frozenset(_resolved_path(path) for path in reserved_paths or ())
    result = CorpusScan()
    report_limit = max(0, skip_report_limit)

    for path in _iter_candidate_files(
        root,
        recursive=recursive,
        default_excludes=default_excludes,
        user_excludes=user_excludes,
        reserved_paths=reserved,
        max_depth=max_depth,
        max_scan_entries=max_scan_entries,
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
        if patterns and not _matches_include_patterns(path, rel, patterns):
            _record_skipped(result, path, rel, "include_filter_mismatch", report_limit)
            continue
        if max_files is not None and len(result.files) >= max_files:
            _record_skipped(result, path, rel, "max_files_exceeded", report_limit)
            if stop_after_max_files:
                result.scan_truncated = True
                result.scan_truncated_reason = "max_files"
                break
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
    stop_after_max_files: bool = False,
    max_depth: int | None = None,
    max_scan_entries: int | None = None,
) -> Iterable[Path]:
    """Yield supported document files under ``root`` in deterministic order."""
    scan = scan_document_files(
        root,
        recursive=recursive,
        include=include,
        exclude=exclude,
        max_files=max_files,
        stop_after_max_files=stop_after_max_files,
        max_depth=max_depth,
        max_scan_entries=max_scan_entries,
        skip_report_limit=0,
    )
    yield from scan.files


def _iter_candidate_files(
    root: Path,
    *,
    recursive: bool,
    default_excludes: Sequence[str],
    user_excludes: Sequence[str],
    reserved_paths: frozenset[Path] | None = None,
    max_depth: int | None = None,
    max_scan_entries: int | None = None,
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
        if scan is not None and scan.scan_truncated:
            return
        rel = path.relative_to(root).as_posix()
        if scan is not None and not _consume_scan_entry(
            scan,
            path,
            rel,
            max_scan_entries,
            report_limit,
        ):
            return
        if reserved_paths is not None and _resolved_path(path) in reserved_paths:
            yield path
            continue
        ignore_reason = _ignore_reason(path, rel, default_excludes, user_excludes)
        if ignore_reason is not None:
            if scan is not None:
                _record_skipped(
                    scan,
                    path,
                    rel,
                    ignore_reason,
                    report_limit,
                    path_type=_skip_path_type(path),
                )
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
                    default_excludes,
                    user_excludes,
                    reserved_paths,
                    max_depth=max_depth,
                    max_scan_entries=max_scan_entries,
                    scan=scan,
                    report_limit=report_limit,
                )
                if scan is not None and scan.scan_truncated:
                    return
            elif scan is not None:
                _record_skipped(
                    scan,
                    path,
                    rel,
                    "non_recursive_directory",
                    report_limit,
                    path_type="directory",
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
    default_excludes: Sequence[str],
    user_excludes: Sequence[str],
    reserved_paths: frozenset[Path] | None = None,
    *,
    max_depth: int | None = None,
    max_scan_entries: int | None = None,
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
        if scan is not None and scan.scan_truncated:
            return
        rel = path.relative_to(root).as_posix()
        if scan is not None and not _consume_scan_entry(
            scan,
            path,
            rel,
            max_scan_entries,
            report_limit,
        ):
            return
        if reserved_paths is not None and _resolved_path(path) in reserved_paths:
            yield path
            continue
        ignore_reason = _ignore_reason(path, rel, default_excludes, user_excludes)
        if ignore_reason is not None:
            if scan is not None:
                _record_skipped(
                    scan,
                    path,
                    rel,
                    ignore_reason,
                    report_limit,
                    path_type=_skip_path_type(path),
                )
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
                default_excludes,
                user_excludes,
                reserved_paths,
                max_depth=max_depth,
                max_scan_entries=max_scan_entries,
                scan=scan,
                report_limit=report_limit,
            )
            if scan is not None and scan.scan_truncated:
                return
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


def _add_corpus_cross_document_links(
    graph: Dict[str, Any],
    *,
    max_links: int | None = None,
) -> CrossDocumentLinkResult:
    """
    Add deterministic mention edges between nodes from different corpus files.

    Per-file extractors intentionally work in isolation. This pass reconnects
    the merged corpus graph when one document explicitly names another
    document's title, section, decision, table, or path-derived stem. The
    optional limit bounds the corpus-wide edge expansion for very large trees.
    """
    nodes = graph.get("nodes", [])
    edges = graph.setdefault("edges", [])
    existing_edges = {
        (edge.get("from"), edge.get("to"), edge.get("label"))
        for edge in edges
    }
    targets = _cross_document_targets(nodes)
    added = 0

    for source_node in nodes:
        source_attrs = source_node.get("attributes") or {}
        source_type = source_attrs.get("type")
        source_path = source_attrs.get("source")
        if source_type not in _CORPUS_LINK_SOURCE_TYPES or not source_path:
            continue
        content = source_node.get("content") or ""
        if not content:
            continue
        source_id = source_node.get("id")
        if not source_id:
            continue

        for target in targets:
            if target["source"] == source_path or target["id"] == source_id:
                continue
            if not _content_matches_any_pattern(content, target["patterns"]):
                continue
            edge_key = (source_id, target["id"], "mentions")
            if edge_key in existing_edges:
                continue
            if max_links is not None and added >= max_links:
                return CrossDocumentLinkResult(added=added, limit_reached=True)
            edges.append(make_edge(source_id, target["id"], "mentions"))
            existing_edges.add(edge_key)
            added += 1

    return CrossDocumentLinkResult(added=added)


def _cross_document_targets(nodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for node in nodes:
        attrs = node.get("attributes") or {}
        node_id = node.get("id")
        source = attrs.get("source")
        if not node_id or not source or attrs.get("type") not in _CORPUS_LINK_TARGET_TYPES:
            continue
        aliases = _cross_document_aliases(node)
        if not aliases:
            continue
        key = f"{source}\0{node_id}"
        if key in seen:
            continue
        targets.append({
            "id": node_id,
            "source": source,
            "patterns": [_alias_pattern(alias) for alias in aliases],
        })
        seen.add(key)
    return targets


def _cross_document_aliases(node: Dict[str, Any]) -> List[str]:
    attrs = node.get("attributes") or {}
    aliases: List[str] = []
    _append_alias(aliases, node.get("label", ""))

    source = attrs.get("source")
    if source and attrs.get("type") in {"document", "decision_document", "media_document"}:
        path = Path(str(source))
        _append_alias(aliases, path.name)
        _append_alias(aliases, path.stem)
        _append_alias(aliases, path.stem.replace("-", " ").replace("_", " "))

    return aliases


def _append_alias(aliases: List[str], value: str) -> None:
    alias = _normalize_alias(value)
    if alias and alias not in aliases:
        aliases.append(alias)


def _normalize_alias(value: str) -> str:
    value = Path(value).stem if "/" in value or "\\" in value else value
    value = re.sub(r"\s+", " ", value.replace("_", " ").replace("-", " ")).strip(" .,:;()[]")
    lower = value.lower()
    if not lower or lower in _GENERIC_CROSS_DOC_ALIASES:
        return ""
    if len(lower) < 4:
        return ""
    if len(lower.split()) == 1 and not any(char.isdigit() for char in lower) and len(lower) < 8:
        return ""
    return value


def _alias_pattern(alias: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.IGNORECASE)


def _content_matches_any_pattern(content: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    for pattern in patterns:
        if pattern.search(content):
            return True
    return False


def _ignore_reason(
    path: Path,
    rel: str,
    default_patterns: Sequence[str],
    user_patterns: Sequence[str],
) -> str | None:
    if _matches_ignore_patterns(path, rel, user_patterns):
        return "exclude_filter_match"
    if _matches_ignore_patterns(path, rel, default_patterns):
        return "default_ignore_match"
    return None


def _matches_ignore_patterns(path: Path, rel: str, patterns: Sequence[str]) -> bool:
    rel_parts = set(rel.split("/"))
    for pattern in patterns:
        if pattern in rel_parts or fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern):
            return True
    return False


def _matches_include_patterns(path: Path, rel: str, patterns: Sequence[str]) -> bool:
    """Return whether a selected file matches user include globs or path parts."""
    rel_parts = set(rel.split("/"))
    for pattern in patterns:
        if pattern in rel_parts or fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern):
            return True
    return False


def _skip_path_type(path: Path) -> str:
    try:
        if path.is_dir():
            return "directory"
    except OSError:
        return "path"
    return "file"


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


def _consume_scan_entry(
    scan: CorpusScan,
    path: Path,
    rel: str,
    max_scan_entries: int | None,
    report_limit: int,
) -> bool:
    """Advance the deterministic scan budget for one filesystem entry."""
    if max_scan_entries is None:
        scan.scanned_entry_count += 1
        return True
    if scan.scanned_entry_count >= max_scan_entries:
        scan.scan_truncated = True
        scan.scan_truncated_reason = "max_scan_entries"
        _record_skipped(
            scan,
            path,
            rel,
            "max_scan_entries_exceeded",
            report_limit,
            path_type=_skip_path_type(path),
        )
        return False
    scan.scanned_entry_count += 1
    return True


def _resolved_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _relative_depth(root: Path, path: Path) -> int:
    return len(path.relative_to(root).parts)


def _file_metadata(
    root: Path,
    path: Path,
    graph_type: str,
    *,
    cached_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    stat = path.stat()
    stat_metadata = {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "inode": stat.st_ino,
    }
    content_sha256 = None
    content_sha256_reused = False
    if cached_metadata and all(
        cached_metadata.get(key) == value for key, value in stat_metadata.items()
    ):
        cached_digest = cached_metadata.get("content_sha256")
        if isinstance(cached_digest, str) and cached_digest:
            content_sha256 = cached_digest
            content_sha256_reused = True
    if content_sha256 is None:
        content_sha256 = _file_sha256(path)
    return {
        "root": str(root.resolve()),
        "relative_path": path.relative_to(root).as_posix(),
        "graph_type": graph_type,
        "extraction_fingerprint": _extraction_fingerprint(graph_type),
        **stat_metadata,
        "content_sha256": content_sha256,
        "content_sha256_reused": content_sha256_reused,
        "suffix": path.suffix.lower(),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths_sha256(paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
    return digest.hexdigest()


def _selected_file_records_sha256(
    root: Path,
    files: Sequence[Path],
    sizes: Dict[Path, int | None],
) -> str:
    digest = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root).as_posix()
        size = sizes.get(path)
        for value in (rel, path.suffix.lower(), "" if size is None else str(size)):
            digest.update(value.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _extraction_fingerprint(graph_type: str) -> str:
    """Fingerprint code that can affect deterministic per-file graph extraction."""
    base = Path(__file__).resolve().parent
    relative_paths = sorted(
        set(EXTRACTION_FINGERPRINT_PATHS)
        | set(EXTRACTION_FINGERPRINT_BY_GRAPH_TYPE.get(graph_type, ()))
    )
    digest = hashlib.sha256()
    digest.update(graph_type.encode("utf-8"))
    for relative_path in relative_paths:
        path = base / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def _cache_key(root: Path, path: Path, graph_type: str) -> str:
    rel = path.relative_to(root).as_posix()
    source = f"{root.resolve()}\0{rel}\0{graph_type}".encode("utf-8")
    digest = hashlib.sha1(source).hexdigest()
    return digest


def _empty_cache() -> Dict[str, Any]:
    return {"version": 1, "entries": {}}


def _load_cache(cache_path: str | Path | None) -> Dict[str, Any]:
    return _load_cache_with_status(cache_path).cache


def _load_cache_with_status(cache_path: str | Path | None) -> CacheLoad:
    if cache_path is None:
        return CacheLoad(cache=_empty_cache(), status="disabled", entry_count=0)
    path = Path(cache_path)
    if not path.exists():
        return CacheLoad(cache=_empty_cache(), status="missing", entry_count=0)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return CacheLoad(cache=_empty_cache(), status="invalid_json", entry_count=0)
    except OSError:
        return CacheLoad(cache=_empty_cache(), status="read_error", entry_count=0)
    if payload.get("version") != 1 or not isinstance(payload.get("entries"), dict):
        return CacheLoad(cache=_empty_cache(), status="invalid_schema", entry_count=0)
    return CacheLoad(cache=payload, status="loaded", entry_count=_cache_entry_count(payload))


def _cache_entry_count(cache: Dict[str, Any]) -> int:
    entries = cache.get("entries")
    return len(entries) if isinstance(entries, dict) else 0


def _write_cache(cache_path: str | Path, cache: Dict[str, Any]) -> bool:
    result = _write_cache_with_status(cache_path, cache)
    if result.status == "write_error":
        raise OSError(result.error_message or "cache write failed")
    return result.updated


def _write_cache_with_status(cache_path: str | Path, cache: Dict[str, Any]) -> CacheWrite:
    path = Path(cache_path)
    payload = _serialize_cache(cache)
    try:
        if path.exists() and path.read_text(encoding="utf-8") == payload:
            return CacheWrite(updated=False, status="unchanged")
    except OSError:
        pass
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f"{path.name}.tmp")
        temp.write_text(payload, encoding="utf-8")
        temp.replace(path)
    except OSError as exc:
        return CacheWrite(updated=False, status="write_error", error_message=str(exc))
    return CacheWrite(updated=True, status="updated")


def _serialize_cache(cache: Dict[str, Any]) -> str:
    return json.dumps(cache, sort_keys=True, separators=(",", ":")) + "\n"


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


def _cache_prune_status(
    scan: CorpusScan,
    *,
    recursive: bool,
    max_files: int | None,
    max_depth: int | None,
    max_scan_entries: int | None,
    max_file_bytes: int | None,
    max_total_bytes: int | None,
) -> str:
    """
    Return whether stale cache pruning is safe for this corpus run.

    Cache entries are keyed by corpus root and graph type. A complete directory
    selection can safely remove absent keys because missing active keys mean the
    file disappeared or filters intentionally excluded it. Bounded traversal or
    extraction runs should keep unrelated warm entries for later full runs.
    """
    if scan.scan_truncated:
        return "skipped_truncated_scan"
    if not recursive or max_files is not None or max_depth is not None:
        return "skipped_bounded_selection"
    if max_scan_entries is not None:
        return "skipped_bounded_selection"
    if max_file_bytes not in (None, DEFAULT_MAX_FILE_BYTES) or max_total_bytes is not None:
        return "skipped_bounded_extraction"
    return "complete_selection"


def _add_runtime_skip(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    file_id: str,
    file_path: Path,
    rel: str,
    reason: str,
    counts: Dict[str, int],
    skipped_records_digest: Any,
    *,
    total_report_limit: int,
    existing_reported_count: int,
    attributes: Dict[str, Any] | None = None,
) -> int:
    _count_skip(counts, reason)
    path_type = str((attributes or {}).get("path_type") or "file")
    _update_skip_digest(skipped_records_digest, rel, reason, path_type)
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
    _update_skip_digest(result.skipped_records_digest, rel, reason, path_type)
    if len(result.skipped_samples) < report_limit:
        result.skipped_samples.append(
            SkippedFile(path=path, relative_path=rel, reason=reason, path_type=path_type)
        )


def _update_skip_digest(digest: Any, rel: str, reason: str, path_type: str) -> None:
    for value in (reason, path_type, rel):
        digest.update(value.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")


def _count_skip(counts: Dict[str, int], reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1


def _id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:16]
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)[:80].strip("_")
    return f"{prefix}:{slug or digest}:{digest}"
