"""Tests for doc2graph.extract_schema_graph()."""

import pytest
from doc2graph.extractors.schema import extract_schema_graph

# ---------------------------------------------------------------------------
# Format 1: markdown pipe tables
# ---------------------------------------------------------------------------

PIPE_TABLE_DOC = """
# E-Commerce Schema

## customers
| Column | Type         | Notes       |
|--------|--------------|-------------|
| id     | INT          | Primary key |
| name   | VARCHAR(100) | Not null    |
| email  | VARCHAR(200) |             |

## orders
| Column      | Type          | Notes                   |
|-------------|---------------|-------------------------|
| id          | INT           | Primary key             |
| customer_id | INT           | References customers    |
| total       | DECIMAL(10,2) |                         |

## order_items
| Column   | Type | Notes               |
|----------|------|---------------------|
| id       | INT  | Primary key         |
| order_id | INT  | References orders   |
| quantity | INT  |                     |
"""

BULLET_DOC = """
## users
- id: INT, primary key
- username: VARCHAR(50), not null
- email: VARCHAR(100)

## posts
- id: INT, primary key
- user_id: INT, references users
- title: VARCHAR(255), not null
"""


def test_pipe_table_nodes_extracted():
    g = extract_schema_graph(PIPE_TABLE_DOC)
    labels = {n["label"] for n in g["nodes"]}
    assert "customers" in labels
    assert "orders" in labels
    assert "order_items" in labels


def test_pipe_table_column_content():
    g = extract_schema_graph(PIPE_TABLE_DOC)
    nodes = {n["label"]: n for n in g["nodes"]}
    assert "name" in nodes["customers"]["content"]
    assert "email" in nodes["customers"]["content"]
    assert "total" in nodes["orders"]["content"]


def test_pipe_table_pk_flagged():
    g = extract_schema_graph(PIPE_TABLE_DOC)
    nodes = {n["label"]: n for n in g["nodes"]}
    assert "PK" in nodes["customers"]["content"]


def test_pipe_table_fk_edge_from_references_note():
    g = extract_schema_graph(PIPE_TABLE_DOC)
    edges = {(e["from"], e["to"]) for e in g["edges"]}
    assert ("orders", "customers") in edges


def test_pipe_table_fk_edge_from_col_name():
    g = extract_schema_graph(PIPE_TABLE_DOC)
    edges = {(e["from"], e["to"]) for e in g["edges"]}
    # order_id → orders inferred from column name
    assert ("order_items", "orders") in edges


def test_pipe_table_node_attributes():
    g = extract_schema_graph(PIPE_TABLE_DOC)
    nodes = {n["label"]: n for n in g["nodes"]}
    assert nodes["customers"]["attributes"]["type"] == "table"


def test_table_ids_are_source_scoped_when_source_is_present():
    left = extract_schema_graph("## customers\n- id: INT\n", source="left.md")
    right = extract_schema_graph("## customers\n- id: INT\n", source="right.md")

    assert left["nodes"][0]["label"] == "customers"
    assert right["nodes"][0]["label"] == "customers"
    assert left["nodes"][0]["id"] != right["nodes"][0]["id"]


# ---------------------------------------------------------------------------
# Format 2: heading + bullet descriptions
# ---------------------------------------------------------------------------

def test_bullet_format_nodes_extracted():
    g = extract_schema_graph(BULLET_DOC)
    labels = {n["label"] for n in g["nodes"]}
    assert "users" in labels
    assert "posts" in labels


def test_bullet_format_column_content():
    g = extract_schema_graph(BULLET_DOC)
    nodes = {n["label"]: n for n in g["nodes"]}
    assert "username" in nodes["users"]["content"]
    assert "title" in nodes["posts"]["content"]


def test_bullet_format_fk_from_references():
    g = extract_schema_graph(BULLET_DOC)
    edges = {(e["from"], e["to"]) for e in g["edges"]}
    assert ("posts", "users") in edges


def test_bullet_format_pk_flagged():
    g = extract_schema_graph(BULLET_DOC)
    nodes = {n["label"]: n for n in g["nodes"]}
    assert "PK" in nodes["users"]["content"]


# ---------------------------------------------------------------------------
# Compatibility with graph2sql
# ---------------------------------------------------------------------------

def test_output_compatible_with_graph2sql():
    """GraphDict from extract_schema_graph must be accepted by SchemaGraph.from_dict()."""
    try:
        from graph2sql import SchemaGraph
    except ImportError:
        pytest.skip("graph2sql not installed")

    g_dict = extract_schema_graph(PIPE_TABLE_DOC)
    schema = SchemaGraph.from_dict(g_dict)
    context = schema.rank("total revenue per customer", k=2)
    labels = [n["label"] for n in context["nodes"]]
    assert any(l in labels for l in ("customers", "orders"))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_text_returns_empty():
    g = extract_schema_graph("")
    assert g == {"nodes": [], "edges": []}


def test_no_tables_returns_empty():
    g = extract_schema_graph("# Introduction\nThis is a general document with no tables.")
    assert g["nodes"] == []
