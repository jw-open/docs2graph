"""
CSV loader for doc2graph — converts tabular CSV data to structured text.

Uses the standard library ``csv`` module — no extra dependencies.
Each row is rendered as ``column: value`` pairs so LLMs can reason about
column semantics, not just positional data.

No installation required.
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path
from typing import List, Optional


def load_csv(
    path: str,
    max_rows: Optional[int] = None,
    delimiter: str = ",",
    include_header: bool = True,
) -> str:
    """
    Load a CSV file and return its content as structured text.

    Each row is rendered as ``column_name: value, column_name: value, ...``
    so downstream models understand column semantics.  Rows with no data
    are skipped.

    Parameters
    ----------
    path : str
        Path to the ``.csv`` (or ``.tsv``) file.
    max_rows : int, optional
        Maximum number of data rows to read.  ``None`` reads all rows.
    delimiter : str
        Field delimiter.  Default ``","``; use ``"\\t"`` for TSV files.
    include_header : bool
        If ``True`` (default), a header summary line listing all column names
        is prepended to the output.

    Returns
    -------
    str
        Structured text representation of the CSV data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not ``.csv`` or ``.tsv``.

    Example
    -------
    >>> from doc2graph.loaders.csv import load_csv
    >>> text = load_csv("data.csv")
    >>> text = load_csv("data.tsv", delimiter="\\t", max_rows=100)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError(f"Expected a .csv or .tsv file, got '{p.suffix}'")

    parts: List[str] = []

    from .text import _detect_encoding
    detected_enc = _detect_encoding(p.read_bytes())

    with p.open(encoding=detected_enc, errors="replace", newline="") as f:
        reader = _csv.DictReader(f, delimiter=delimiter)
        headers = reader.fieldnames or []

        if include_header and headers:
            parts.append(f"Columns: {', '.join(headers)}\n")

        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            pairs = [f"{k}: {v}" for k, v in row.items() if isinstance(v, str) and v.strip()]
            if pairs:
                parts.append(", ".join(pairs))

    return "\n".join(parts)
