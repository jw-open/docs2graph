# docs2graph

[![PyPI version](https://img.shields.io/pypi/v/doc2graph.svg)](https://pypi.org/project/doc2graph/)
[![PyPI downloads](https://img.shields.io/pypi/dm/doc2graph.svg)](https://pypi.org/project/doc2graph/)
[![Python](https://img.shields.io/pypi/pyversions/doc2graph.svg)](https://pypi.org/project/doc2graph/)
[![CI](https://github.com/jw-open/doc2graph/actions/workflows/ci.yml/badge.svg)](https://github.com/jw-open/doc2graph/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

**Turn documents into queryable knowledge graphs — no LLM required.**

When you need to answer questions over long files, reports, or multi-document corpora, naive chunking loses structure. `doc2graph` extracts entities, relationships, and context as a graph, ranks relevant nodes with Personalized PageRank, and hands you exactly what your LLM needs.

**Pure Python. No LLM dependency. Bring your own model.**

---

## Why graph-based context?

| Approach | What you lose |
|----------|--------------|
| Fixed-size chunking | Sentence boundaries, section context, cross-references |
| Embedding search | Exact match, structural relationships, citation graphs |
| **doc2graph** | Nothing — relationships are explicit edges |

The graph knows that a *claim* is supported by *evidence* in a specific *section*, which *cites* a *reference*, which is authored by a specific *person*. Flat chunks don't.

---

## Quick start

```bash
pip install docs2graph
```

```bash
# Extract a knowledge graph from a paper
docs2graph paper.md --graph knowledge --output paper.graph.json

# Extract ADR-style decisions from architecture docs
docs2graph architecture.md --graph decision --output decisions.graph.json

# Process an entire documentation corpus
docs2graph ./docs --graph all --output corpus.graph.json
```

```python
from doc2graph import DocumentGraph

# Single document
g = DocumentGraph.from_document("report.pdf", graph_type="knowledge")

# Rank nodes relevant to a query
context = g.rank("what are the key risks?", k=10)
# Pass context["nodes"] + context["edges"] to your LLM
```

---

## Installation

**Core (Markdown, plain text, HTML, JSON, CSV, code files):**
```bash
pip install docs2graph
```

**With PDF support:**
```bash
pip install "docs2graph[pdf]"
```

**With Word / PowerPoint support:**
```bash
pip install "docs2graph[docx,pptx]"
```

**With OCR (images, scanned PDFs):**
```bash
pip install "docs2graph[ocr]"
# Also requires: apt install tesseract-ocr  (Ubuntu/Debian)
#                brew install tesseract     (macOS)
```

**Everything:**
```bash
pip install "docs2graph[all]"
```

---

## How it works

```
Document / File / Corpus
         │
         ▼
   Format loader ──► Text + structure
         │
         ▼
   Extractor ──► Nodes (entities, sections, claims, ...)
         │         └─► Edges (contains, references, defines, ...)
         ▼
   Knowledge graph (plain JSON)
         │
         ▼
   query → Personalized PageRank → Ranked subgraph
         │
         ▼
   Your LLM prompt
```

1. **Load** — auto-detects format, handles encoding, extracts clean text and structure
2. **Extract** — turns structure into typed graph nodes and labeled edges
3. **Rank** — Personalized PageRank starting from query-matched nodes surfaces the most relevant subgraph
4. **Use** — pass `context["nodes"]` + `context["edges"]` to any LLM

---

## Graph types

### `knowledge` — for research papers, reports, documentation

Extracts: **documents, sections, concepts, definitions, claims, evidence, tables, citations, references, URLs**

```bash
docs2graph paper.md --graph knowledge --output paper.graph.json
```

```python
g = DocumentGraph.from_document("paper.md", graph_type="knowledge")
context = g.rank("graph-based context ranking", k=15)
```

Relationships: `contains`, `references`, `defines`, `defined_by`, `supports`, `cites`, `resolves_to`, `links_to`

Inline citations (`[1]`, `(Smith, 2024)`) are resolved to matching `# References` entries. Claim-to-evidence support links are deterministic and conservative — only same-section evidence or evidence sharing meaningful terms with the claim.

### `decision` — for ADRs and architecture documents

Extracts: **problems, context/drivers, options, pros, cons, tradeoffs, decisions, consequences, confidence**

```bash
docs2graph architecture.md --graph decision --output decisions.graph.json
```

```python
decisions = DocumentGraph.from_document("adr.md", graph_type="decision")
```

Recognizes ADR-style headings (`## Decision`, `## Options`, `## Consequences`), standalone prefixed lines (`Constraint:`, `Assumption:`, `Rationale:`), and Markdown option tables. Context bullets link to decisions with `informed_by` edges so the reasoning trail is traversable.

### `schema` — for data dictionaries and schema docs

Extracts table and entity graphs from schema documentation for text-to-SQL context.

### `media` — for images and charts

Extracts image metadata, OCR text, and chart signal nodes.

### `all` — merged graph from all extractors

```bash
docs2graph ./docs --graph all --output corpus.graph.json
```

---

## Multi-document corpora

Directory input is first-class. `doc2graph` walks supported formats, emits a corpus root with folder/file provenance nodes, records selected/extracted/failed/skipped counts by extension for large-corpus audits, resolves explicit relative links (`[ADR](adr/cache.md)`) into `links_to` edges, and adds deterministic cross-document `mentions` edges when one file explicitly names another's title, section, decision, or path-derived stem.

```bash
# Process entire knowledge base
docs2graph ./knowledge-base --graph all --output corpus.graph.json

# Filter to ADRs only
docs2graph ./knowledge-base --graph decision --include "adr/**" --output adr.graph.json

# Limit corpus size
docs2graph ./exports --graph all --max-files 500 --max-file-bytes 10485760

# Bounded traversal for huge trees
docs2graph ./exports --graph all --max-depth 2 --max-total-bytes 1073741824

# Audit corpus before extraction (no files loaded)
docs2graph ./exports --graph all --scan-only --output corpus.scan.graph.json

# Cache extraction results across runs
docs2graph ./docs --graph all --cache .doc2graph-cache.json --output corpus.graph.json
```

### Corpus limits reference

| Flag | Default | Description |
|------|---------|-------------|
| `--max-files N` | unlimited | Select at most N files; continues scanning for skip counts |
| `--stop-after-max-files` | off | Stop scanning at first file beyond `--max-files` |
| `--max-file-bytes N` | 25 MB | Skip files larger than N bytes |
| `--max-total-bytes N` | unlimited | Stop extracting after N cumulative bytes |
| `--max-depth N` | unlimited | Bound recursive descent by subdirectory depth |
| `--max-scan-entries N` | unlimited | Stop directory walk after N filesystem entries |
| `--include PATTERN` | all | Repeatable glob filter (e.g. `--include "adr/**"`) |
| `--exclude PATTERN` | none | Repeatable glob exclusion |
| `--extension EXT` | all | Repeatable suffix allowlist (e.g. `--extension md`) |
| `--scan-only` | off | Build scan graph without loading any files |
| `--follow-symlinks` | off | Extract symlinked files (symlinked dirs always skipped) |
| `--cache PATH` | none | Reuse unchanged per-file extractions across runs |
| `--refresh-cache` | off | Rebuild all cache entries |

---

## Supported formats

| Format | Extensions | Extra install |
|--------|-----------|---------------|
| Markdown | `.md`, `.mdx` | — |
| Plain text | `.txt` | — |
| HTML | `.html` | — |
| JSON / JSONL | `.json`, `.jsonl` | — |
| CSV / TSV | `.csv`, `.tsv` | — |
| Source code | `.py`, `.js`, `.ts`, `.sql`, `.yaml`, `.toml`, `.sh`, ... | — |
| PDF | `.pdf` | `pip install "docs2graph[pdf]"` |
| Word | `.docx` | `pip install "docs2graph[docx]"` |
| PowerPoint | `.pptx` | `pip install "docs2graph[pptx]"` |
| Images / OCR | `.png`, `.jpg`, `.gif`, `.tif`, `.bmp`, `.webp` | `pip install "docs2graph[ocr]"` + tesseract |
| URLs | `https://...` | — |
| Google Docs/Sheets/Slides | public export URLs | `GOOGLE_DOCS_BEARER_TOKEN` for private |

---

## Python API

### Single document

```python
from doc2graph import DocumentGraph

# Auto-detect format
g = DocumentGraph.from_document("paper.pdf", graph_type="knowledge")

# Markdown
g = DocumentGraph.from_markdown("notes.md", graph_type="all")

# Plain text
g = DocumentGraph.from_text("My text content...", graph_type="knowledge")
```

### Directory corpus

```python
g = DocumentGraph.from_directory(
    "./docs",
    graph_type="all",
    max_depth=3,
    max_files=500,
    cache=".doc2graph-cache.json",
)
```

### Query and rank

```python
context = g.rank("what are the main risks?", k=10)
# Returns {"nodes": [...], "edges": [...]}

# Pass to any LLM
prompt = f"Context:\n{context}\n\nQuestion: what are the main risks?"
```

### Build and export

```python
# Export
g.to_json("graph.json")           # plain JSON
g.to_graphml("graph.graphml")     # GraphML for Gephi / yEd
graph_dict = g.to_dict()          # raw {"nodes": [...], "edges": [...]}

# Inspect
print(len(g.nodes))
print(len(g.edges))
```

### Graph output format

```json
{
  "nodes": [
    {
      "id": "claim:this_paper_proposes_a_graph_based_approach",
      "label": "This paper proposes a graph based approach",
      "content": "This paper proposes a graph based approach...",
      "attributes": {
        "type": "claim",
        "source": "paper.md",
        "extraction_method": "static"
      }
    }
  ],
  "edges": [
    {
      "from": "section:0_abstract",
      "to": "claim:this_paper_proposes_a_graph_based_approach",
      "label": "contains"
    }
  ]
}
```

---

## CLI reference

```bash
docs2graph <source> [options]

Arguments:
  source          File path, directory path, or URL

Options:
  --graph TYPE    Graph type: knowledge, decision, schema, media, all (default: all)
  --output PATH   Output JSON file (default: stdout)
  --max-files N   Maximum files to extract from a directory
  --max-depth N   Maximum directory recursion depth
  --cache PATH    Cache file for incremental corpus runs
  --scan-only     Build scan graph without loading files
  --include GLOB  Include pattern (repeatable)
  --exclude GLOB  Exclude pattern (repeatable)
  --extension EXT File extension filter (repeatable)
  -h, --help      Show help
```

---

## Use cases

- **RAG over technical docs** — extract section/concept graph, rank on query, pass subgraph as focused context instead of raw chunks
- **Research paper analysis** — extract entity/citation graph, find what a paper claims and what evidence it cites
- **Architecture review** — extract decision graphs from ADRs, trace the reasoning behind every architectural choice
- **Contract review** — extract clause relationships, identify obligations and conditions
- **Code understanding** — combine with [code2graph](https://github.com/jw-open/code2graph) for cross-document + cross-code context
- **Text-to-SQL** — combine with [graph2sql](https://github.com/jw-open/graph2sql) for schema-aware query generation

---

## Design principles

- **Pure Python** — no LLM, no cloud service, no database required
- **No LLM dependency** — extraction is deterministic and static; LLM enrichment is opt-in and labeled `extraction_method: llm_inferred`
- **Deterministic outputs** — same input always produces the same graph, making corpus runs reproducible and diffable
- **Works with any model** — output is plain JSON; pass to GPT-4, Claude, Llama, Mistral, or any other model
- **Pluggable** — add your own loader or extractor without touching core code
- **Shared core** — same Personalized PageRank engine as [graph2sql](https://github.com/jw-open/graph2sql)

---

## Related projects

| Package | What it does |
|---------|-------------|
| [graph2sql](https://github.com/jw-open/graph2sql) | Graph-based schema analysis for text-to-SQL — same PPR core |
| [code2graph](https://github.com/jw-open/code2graph) | Code repository → knowledge graph (modules, classes, dependencies) |

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/jw-open/doc2graph
cd doc2graph
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

Apache-2.0 — see [LICENSE](LICENSE)
