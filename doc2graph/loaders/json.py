"""JSON and JSON Lines loader for doc2graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from .text import _detect_encoding


def load_json(path: str, max_items: Optional[int] = None) -> str:
    """
    Load a JSON or JSONL file and return deterministic structured text.

    JSON objects are traversed with sorted keys so corpus graph extraction is
    stable across runs. JSONL records preserve file order.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix.lower() not in {".json", ".jsonl"}:
        raise ValueError(f"Expected a .json or .jsonl file, got '{p.suffix}'")

    raw = p.read_bytes()
    encoding = _detect_encoding(raw)
    text = raw.decode(encoding, errors="replace")

    if p.suffix.lower() == ".jsonl":
        return _load_jsonl_text(text, max_items=max_items)

    data = json.loads(text)
    return _render_json_value(data, max_items=max_items)


def _load_jsonl_text(text: str, *, max_items: Optional[int]) -> str:
    parts: List[str] = []
    seen = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if max_items is not None and seen >= max_items:
            break
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL on line {line_number}: {exc.msg}") from exc
        rendered = _render_json_value(value, prefix=f"record {seen + 1}", max_items=max_items)
        if rendered:
            parts.append(rendered)
        seen += 1
    return "\n".join(parts)


def _render_json_value(
    value: Any,
    *,
    prefix: str = "",
    max_items: Optional[int] = None,
) -> str:
    lines: List[str] = []
    _append_json_lines(lines, value, prefix=prefix, max_items=max_items)
    return "\n".join(lines)


def _append_json_lines(
    lines: List[str],
    value: Any,
    *,
    prefix: str,
    max_items: Optional[int],
) -> None:
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        if max_items is not None:
            items = items[:max_items]
        for key, child in items:
            child_prefix = _join_path(prefix, str(key))
            _append_json_lines(lines, child, prefix=child_prefix, max_items=max_items)
        return

    if isinstance(value, list):
        items = value if max_items is None else value[:max_items]
        for index, child in enumerate(items):
            child_prefix = _join_path(prefix, str(index))
            _append_json_lines(lines, child, prefix=child_prefix, max_items=max_items)
        return

    if value is None:
        rendered = "null"
    elif isinstance(value, bool):
        rendered = "true" if value else "false"
    else:
        rendered = str(value)

    if prefix:
        lines.append(f"{prefix}: {rendered}")
    else:
        lines.append(rendered)


def _join_path(prefix: str, key: str) -> str:
    if not prefix:
        return key
    return f"{prefix}.{key}"
