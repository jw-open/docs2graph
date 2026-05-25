# doc2graph

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange)]()

**Turn documents into queryable knowledge and decision graphs — no LLM required by default.**

When you need to answer questions over long files, reports, or corpora, naively chunking and embedding loses the structure. doc2graph extracts entities, relationships, and context as a graph — so you can traverse it, rank it, and feed exactly what's relevant to your LLM.

**No LLM included. Bring your own model.**

---

## What it does

```
Document / File
     │
     ▼
Entity extraction ──► Relationship extraction
     │                        │
     └──────────┬─────────────┘
                ▼
         Knowledge graph
         (nodes + edges)
                │
                ▼
    Query → Ranked subgraph
                │
                ▼
          Your LLM prompt
```

1. Feed it a document or folder tree — PDF, Google Doc export URL, Markdown, plain text, HTML, image/chart, code file, or mixed document corpus
2. It extracts entities (people, concepts, terms, sections) as nodes
3. It extracts relationships (references, defines, depends-on, authored-by) as edges
4. You query the graph and get back only the relevant subgraph
5. Pass that focused context to any LLM

### Current graph modes

```bash
doc2graph paper.md --graph knowledge --output paper.graph.json
doc2graph architecture.md --graph decision --output decisions.graph.json
doc2graph schema.md --graph schema --output schema.graph.json
doc2graph chart.png --graph media --output chart.graph.json
doc2graph docs.md --graph all --output docs.graph.json
doc2graph ./docs --graph all --output docs-corpus.graph.json
```

- `knowledge`: document, section, concept, claim, evidence, citation, and URL nodes.
- `decision`: problem, context, option, pros, cons, tradeoff, decision, consequence, and confidence nodes.
- `schema`: table/entity graphs from schema docs and data dictionaries.
- `media`: image/chart metadata, OCR text, and chart signal nodes.
- `all`: merged graph from the supported document extractors.

### Supported sources

- Local text-like files: `.md`, `.mdx`, `.txt`, `.html`, `.csv`, `.tsv`
- Office-style files with extras: `.docx`, `.pptx`
- PDF: native embedded text via `pypdf`, with OCR fallback for scanned PDFs
- Images/charts: `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.tif`, `.tiff`, `.bmp` via OCR and media metadata
- URLs: generic text/HTML URLs and public/exportable Google Docs, Sheets, and Slides URLs
- Directories: recursive mixed-format corpora with folder/file provenance nodes

Private Google Workspace documents require either public export access or
`GOOGLE_DOCS_BEARER_TOKEN` with permission to read the document.

Directory input is first-class. doc2graph walks supported document formats,
skips common generated folders such as `.git`, `node_modules`, `dist`, and
`build`, and emits a corpus root plus folder/file nodes linked to each extracted
document graph:

```bash
doc2graph ./knowledge-base --graph all --output corpus.graph.json
doc2graph ./knowledge-base --graph decision --include "adr/**" --output adr.graph.json
doc2graph ./exports --graph all --max-files 500 --max-file-bytes 10485760
doc2graph ./exports --graph all --max-total-bytes 1073741824 --output corpus.graph.json
doc2graph ./exports --graph all --skip-report-limit 25 --output corpus.graph.json
doc2graph ./exports --graph all --cache .doc2graph-cache.json --output corpus.graph.json
```

Large corpora are handled by deterministic limits:

- `--max-files N`: stop after N supported files.
- `--max-file-bytes N`: skip very large individual files and add a `skipped_file` node.
- `--max-file-bytes -1`: disable the per-file size guard.
- `--max-total-bytes N`: stop extracting files after the cumulative extracted
  byte budget is reached and report remaining files as skipped.
- `--max-total-bytes -1`: disable the cumulative byte guard.
- `--no-recursive`: only process files directly under the directory.
- `--include` / `--exclude`: repeatable glob filters for folder subsets.
- `--skip-report-limit N`: cap the total number of omitted files listed as
  `skipped_file` or extraction-error nodes while still preserving aggregate
  skip counts.
- `--cache PATH`: opt into a JSON cache that reuses unchanged per-file graph
  extraction across repeated corpus runs.
- `--refresh-cache`: rebuild cached entries while writing an updated cache.

Every directory graph includes a `corpus_manifest` node with selected-file
counts, skipped-file counts, cache hit/miss/write counts when caching is
enabled, extracted-file byte counts, and skip reasons such as unsupported
extensions, max-file limits, cumulative byte limits, oversized files,
symlinked directories, inaccessible paths, and per-file extraction errors. This
keeps large mixed-folder runs deterministic and auditable without requiring all
omitted paths to be materialized as graph nodes.

Outputs are plain JSON:

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
  ],
  "current_node_id": "document:paper_md"
}
```

The base graph is static and deterministic. LLM enrichment should be optional
and provenance-labeled, for example `extraction_method=llm_inferred`, so
inferred reasoning is not confused with documented evidence.

---

## Planned use cases

- **RAG over technical docs** — extract section/concept graph, rank on query, pass subgraph as context instead of raw chunks
- **Code understanding** — extract class/function/module dependency graph, answer questions about structure
- **Research paper analysis** — extract entity/citation graph, find what a paper claims and what it cites
- **Contract / legal doc review** — extract clause relationships, identify obligations and conditions
- **Long-form report QA** — extract key findings and their evidence, answer without hallucinating

---

## Planned API

```python
from doc2graph import DocumentGraph

g = DocumentGraph()
g.load("report.pdf")             # or .md, .txt, .html, .py ...
g.extract()                      # builds nodes + edges

context = g.rank("what are the key risks?", k=5)
# context["nodes"] + context["edges"] → pass to your LLM
```

Current document-native API:

```python
from doc2graph import DocumentGraph, extract_knowledge_graph, extract_decision_graph

g = DocumentGraph.from_document("paper.md", graph_type="knowledge")
decisions = DocumentGraph.from_document("adr.md", graph_type="decision")
corpus = DocumentGraph.from_directory("./docs", graph_type="all")
```

### Load from multiple files

```python
g = DocumentGraph()
g.load_many(["paper1.pdf", "paper2.pdf", "notes.md"])
g.extract()
context = g.rank("how does attention mechanism work?", k=10)
```

### Export the graph

```python
g.to_dict()     # raw {"nodes": [...], "edges": [...]}
g.to_json("graph.json")
g.to_graphml("graph.graphml")
```

---

## Planned folder structure

```
doc2graph/
├── doc2graph/
│   ├── __init__.py          # public API: DocumentGraph
│   ├── graph.py             # DocumentGraph class
│   ├── loaders/
│   │   ├── pdf.py           # PDF → plain text
│   │   ├── markdown.py      # Markdown → structured sections
│   │   ├── html.py          # HTML → text + links
│   │   └── code.py          # Python/JS/etc → AST-based extraction
│   ├── extractors/
│   │   ├── entity.py        # entity extraction (rule-based + optional spacy)
│   │   ├── relation.py      # relationship extraction
│   │   └── section.py       # section/heading graph
│   ├── ranking.py           # Personalized PageRank (shared with graph2sql)
│   └── types.py             # Node / Edge types
├── tests/
│   ├── test_loaders.py
│   ├── test_extractors.py
│   └── test_ranking.py
├── examples/
│   ├── pdf_qa.py
│   └── codebase.py
├── benchmarks/
│   └── README.md            # planned: HotpotQA, MuSiQue, 2WikiMultihopQA
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Benchmark goals

Evaluation planned against multi-hop QA benchmarks:

| Benchmark | Type | Goal |
|---|---|---|
| [HotpotQA](https://hotpotqa.github.io/) | Multi-hop QA over Wikipedia | Match or exceed chunk-based RAG accuracy with fewer tokens |
| [MuSiQue](https://arxiv.org/abs/2108.00573) | Multi-hop, harder reasoning | Graph traversal outperforms flat retrieval |
| [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop) | Multi-document reasoning | Test cross-document relationship extraction |

The goal: show that graph-ranked context achieves comparable QA accuracy to full-document prompting while using significantly fewer tokens.

---

## Design goals

- Pure Python — no LLM, no cloud, no database required
- Pluggable loaders — add your own file type
- Works with any model (GPT-4, Llama, Claude, Mistral...)
- `rank()` returns a plain dict — serialize however you want
- Shares the same graph + PPR core as [graph2sql](https://github.com/jw-open/graph2sql)

---

## Status

Pre-alpha. API is being designed. Contributions and feedback welcome — open an issue.

---

## License

Apache-2.0
