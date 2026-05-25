"""
Schema graph extraction from structured documents — pure Python, no NLP deps.

Understands two common schema documentation formats:

**Format 1 — Markdown pipe tables** (most common in data dictionaries):

    ## customers
    | Column | Type        | Notes       |
    |--------|-------------|-------------|
    | id     | INT         | Primary key |
    | name   | VARCHAR(100)| Not null    |

**Format 2 — Heading + bullet/paragraph descriptions**:

    ## orders
    - id: INT, primary key
    - customer_id: INT, references customers
    - total: DECIMAL(10,2)

The output GraphDict is compatible with ``graph2sql.SchemaGraph.from_dict()``,
so the two packages can be chained::

    from doc2graph.extractors.schema import extract_schema_graph
    from graph2sql import SchemaGraph

    graph_dict = extract_schema_graph(schema_doc_text)
    g = SchemaGraph.from_dict(graph_dict)
    context = g.rank("total revenue per customer")
"""

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from ..types import GraphDict, make_edge, make_node

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\|(.+)\|$", re.MULTILINE)
_SEPARATOR_RE = re.compile(r"^\|[\s\-:|]+\|$")


def extract_schema_graph(text: str, source: str = "") -> GraphDict:
    """
    Extract a schema graph from a schema documentation document.

    Tries markdown pipe-table format first. If no tables are found, falls
    back to heading + bullet/paragraph parsing.

    Parameters
    ----------
    text : str
        Raw text of a schema documentation file (Markdown, plain text, etc.).
    source : str, optional
        Source file path — stored as a node attribute.

    Returns
    -------
    GraphDict
        ``{"nodes": [...], "edges": [...]}"`` compatible with
        ``graph2sql.SchemaGraph.from_dict()``.

    Example
    -------
    >>> text = '''
    ... ## customers
    ... | Column | Type         |
    ... |--------|--------------|
    ... | id     | INT          |
    ... | name   | VARCHAR(100) |
    ... '''
    >>> from doc2graph.extractors.schema import extract_schema_graph
    >>> g = extract_schema_graph(text)
    >>> g["nodes"][0]["label"]
    'customers'
    """
    result = _extract_from_markdown_tables(text, source)
    if result["nodes"]:
        return _add_fk_edges(result)

    result = _extract_from_sections(text, source)
    return _add_fk_edges(result)


# ---------------------------------------------------------------------------
# Format 1: Markdown pipe tables
# ---------------------------------------------------------------------------

def _extract_from_markdown_tables(text: str, source: str) -> GraphDict:
    """Parse headings followed by markdown pipe tables."""
    nodes: List[Dict[str, Any]] = []
    headings = list(_HEADING_RE.finditer(text))

    for i, heading_match in enumerate(headings):
        table_name = heading_match.group(1).strip()
        section_start = heading_match.end()
        section_end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        section = text[section_start:section_end]

        table = _parse_pipe_table(section)
        if table is None:
            continue

        col_defs = _table_rows_to_col_defs(table)
        if not col_defs:
            continue

        attrs: Dict[str, Any] = {"type": "table"}
        if source:
            attrs["source"] = source

        nodes.append(make_node(
            id=_table_id(table_name, source),
            label=table_name,
            content=", ".join(col_defs),
            attributes=attrs,
        ))

    return {"nodes": nodes, "edges": []}


def _parse_pipe_table(text: str) -> Optional[List[List[str]]]:
    """
    Extract rows from a markdown pipe table.
    Returns list of row lists (header row first), or None if no table found.
    """
    lines = text.splitlines()
    table_lines = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if re.match(r"^\|.+\|$", stripped):
            in_table = True
            if not _SEPARATOR_RE.match(stripped):
                table_lines.append(stripped)
        elif in_table:
            break  # table ended

    if len(table_lines) < 2:  # need at least header + one data row
        return None

    rows = []
    for line in table_lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return rows


def _table_rows_to_col_defs(rows: List[List[str]]) -> List[str]:
    """
    Convert table rows to column definition strings.

    Looks for columns named "column"/"field"/"name" and "type" in the header.
    Falls back to positional: first col = name, second col = type.
    """
    if not rows:
        return []

    header = [h.lower() for h in rows[0]]

    # Locate column-name and type columns
    name_idx = _find_col(header, ("column", "field", "name", "col"))
    type_idx = _find_col(header, ("type", "datatype", "data type", "dtype"))
    note_idx = _find_col(header, ("notes", "note", "description", "desc", "constraints", "comment"))

    # Positional fallback
    if name_idx is None:
        name_idx = 0
    if type_idx is None:
        type_idx = 1 if len(header) > 1 else None

    col_defs = []
    for row in rows[1:]:
        if len(row) <= name_idx:
            continue
        col_name = row[name_idx].strip("` ")
        if not col_name or col_name.startswith("-"):
            continue

        parts = [col_name]
        if type_idx is not None and type_idx < len(row):
            col_type = row[type_idx].strip()
            if col_type:
                parts.append(col_type)

        # Extract flags and FK references from notes/description
        if note_idx is not None and note_idx < len(row):
            note = row[note_idx]
            note_lower = note.lower()
            if "primary key" in note_lower or " pk" in note_lower:
                parts.append("PK")
            if "not null" in note_lower or "required" in note_lower:
                parts.append("NOT NULL")
            # Preserve "references <table>" so FK edge detection can find it
            ref_m = re.search(r"\breferences?\s+(\w+)", note, re.IGNORECASE)
            if ref_m:
                parts.append(f"references {ref_m.group(1)}")

        col_defs.append(" ".join(parts))

    return col_defs


def _find_col(header: List[str], candidates: tuple) -> Optional[int]:
    for candidate in candidates:
        for i, h in enumerate(header):
            if candidate in h:
                return i
    return None


def _table_id(table_name: str, source: str) -> str:
    slug = table_name.lower().replace(" ", "_")
    if not source:
        return slug
    scoped_value = f"{source}\0{table_name}"
    digest = hashlib.sha1(scoped_value.encode("utf-8", errors="ignore")).hexdigest()[:10]
    scoped_slug = re.sub(r"[^a-zA-Z0-9]+", "_", scoped_value.lower()).strip("_")[:70]
    return f"{scoped_slug or slug}:{digest}"


# ---------------------------------------------------------------------------
# Format 2: Heading + bullet/paragraph descriptions
# ---------------------------------------------------------------------------

def _extract_from_sections(text: str, source: str) -> GraphDict:
    """
    Parse schema docs where headings are table names and content
    describes columns as bullets or key: value lines.

    Supports patterns like:
        - id: INT, primary key
        - customer_id: INT, references customers
        * name VARCHAR(100) not null
    """
    nodes: List[Dict[str, Any]] = []
    headings = list(_HEADING_RE.finditer(text))

    col_line_re = re.compile(
        r"^[\-\*\•]\s*`?(\w+)`?\s*[:\s]\s*(.+)$", re.MULTILINE
    )

    for i, heading_match in enumerate(headings):
        table_name = heading_match.group(1).strip()
        section_start = heading_match.end()
        section_end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        section = text[section_start:section_end]

        col_matches = list(col_line_re.finditer(section))
        if not col_matches:
            continue

        col_defs = []
        for m in col_matches:
            col_name = m.group(1)
            desc = m.group(2).strip().rstrip(",")

            # Extract type: first word that looks like a SQL type
            type_m = re.search(
                r"\b(INT|INTEGER|BIGINT|SMALLINT|VARCHAR|TEXT|CHAR|"
                r"DECIMAL|NUMERIC|FLOAT|DOUBLE|BOOLEAN|BOOL|DATE|"
                r"DATETIME|TIMESTAMP|JSON|UUID|SERIAL|BLOB)[\w(,)]*\b",
                desc, re.IGNORECASE
            )
            col_type = type_m.group(0) if type_m else ""

            flags = []
            if re.search(r"\bprimary\s+key\b|\bpk\b", desc, re.IGNORECASE):
                flags.append("PK")
            if re.search(r"\bnot\s+null\b|\brequired\b", desc, re.IGNORECASE):
                flags.append("NOT NULL")

            parts = [col_name]
            if col_type:
                parts.append(col_type)
            parts.extend(flags)
            col_defs.append(" ".join(parts))

        if not col_defs:
            continue

        attrs: Dict[str, Any] = {"type": "table"}
        if source:
            attrs["source"] = source

        nodes.append(make_node(
            id=_table_id(table_name, source),
            label=table_name,
            content=", ".join(col_defs),
            attributes=attrs,
        ))

    return {"nodes": nodes, "edges": []}


# ---------------------------------------------------------------------------
# FK edge inference (applied after both parsers)
# ---------------------------------------------------------------------------

_FK_SUFFIX_RE = re.compile(r"^(.+)_id$", re.IGNORECASE)
_REF_RE = re.compile(r"\breferences?\s+(\w+)", re.IGNORECASE)


def _add_fk_edges(graph: GraphDict) -> GraphDict:
    """
    Infer foreign-key edges from column names and content.

    Rules (in order):
    1. Explicit "references <table>" in column content.
    2. Column named ``<table>_id`` where ``<table>`` is a known table.
    """
    node_ids = {n["id"] for n in graph["nodes"]}
    label_to_id = {n["label"].lower(): n["id"] for n in graph["nodes"]}
    edges: List[Dict[str, Any]] = list(graph.get("edges", []))
    seen: set = {(e["from"], e["to"]) for e in edges}

    for node in graph["nodes"]:
        content = node.get("content") or ""
        from_id = node["id"]

        # Rule 1: explicit "references <table>" in content
        for ref_match in _REF_RE.finditer(content):
            ref_name = ref_match.group(1).lower()
            to_id = label_to_id.get(ref_name)
            if to_id and to_id != from_id and (from_id, to_id) not in seen:
                edges.append(make_edge(from_id, to_id, "foreign_key"))
                seen.add((from_id, to_id))

        # Rule 2: column named <table>_id where <table> matches a known table
        for col_def in content.split(","):
            col_name = col_def.strip().split()[0] if col_def.strip() else ""
            fk_m = _FK_SUFFIX_RE.match(col_name)
            if fk_m:
                ref_name = fk_m.group(1).lower()
                # Try: exact, plural (+ s), singular (- s)
                candidates = [ref_name, ref_name + "s", ref_name.rstrip("s")]
                to_id = next(
                    (label_to_id[c] for c in candidates if c in label_to_id), None
                )
                if to_id and to_id != from_id and (from_id, to_id) not in seen:
                    edges.append(make_edge(from_id, to_id, "foreign_key"))
                    seen.add((from_id, to_id))

    graph["edges"] = edges
    return graph
