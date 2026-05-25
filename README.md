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

1. Feed it a document — PDF, Google Doc export URL, Markdown, plain text, HTML, image/chart, code file
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
```

- `knowledge`: document, section, concept, claim, evidence, citation, and URL nodes.
- `decision`: problem, context, option, pros, cons, tradeoff, decision, and consequence nodes.
- `schema`: table/entity graphs from schema docs and data dictionaries.
- `media`: image/chart metadata, OCR text, and chart signal nodes.
- `all`: merged graph from the supported document extractors.

### Supported sources

- Local text-like files: `.md`, `.mdx`, `.txt`, `.html`, `.csv`, `.tsv`
- Office-style files with extras: `.docx`, `.pptx`
- PDF: native embedded text via `pypdf`, with OCR fallback for scanned PDFs
- Images/charts: `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.tif`, `.tiff`, `.bmp` via OCR and media metadata
- URLs: generic text/HTML URLs and public/exportable Google Docs, Sheets, and Slides URLs

Private Google Workspace documents require either public export access or
`GOOGLE_DOCS_BEARER_TOKEN` with permission to read the document.

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
