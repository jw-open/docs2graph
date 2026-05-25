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

1. Feed it a document or folder tree — PDF, Google Doc export URL, Markdown, plain text, HTML, JSON/JSONL, image/chart, code file, or mixed document corpus
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

- `knowledge`: document, section, concept, definition, claim, evidence, citation, reference, and URL nodes.
- `decision`: problem, context/driver/rationale, option, pros, cons, tradeoff, decision, consequence, and confidence nodes from ADR headings, status sections, bullets, and Markdown option tables.
- `schema`: table/entity graphs from schema docs and data dictionaries.
- `media`: image/chart metadata, OCR text, and chart signal nodes.
- `all`: merged graph from the supported document extractors.

### Supported sources

- Local text-like files: `.md`, `.mdx`, `.txt`, `.html`, `.csv`, `.tsv`
  with BOM-aware and best-effort legacy encoding handling
- Office-style files with extras: `.docx`, `.pptx`
- PDF: native embedded text via `pypdf`, with OCR fallback for scanned PDFs
- Images/charts: `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.tif`, `.tiff`, `.bmp` via OCR and media metadata
- URLs: generic text/HTML URLs and public/exportable Google Docs, Sheets, and Slides URLs
- Source/config files: Python, JavaScript/TypeScript, SQL, YAML, JSON, TOML, shell, and other common text code formats
- Directories: recursive mixed-format corpora with folder/file provenance nodes
- JSON/JSONL: deterministic structured-text extraction for config files, exports, and line-delimited records

Private Google Workspace documents require either public export access or
`GOOGLE_DOCS_BEARER_TOKEN` with permission to read the document.

Directory input is first-class. doc2graph walks supported document formats,
skips common generated folders inside the selected corpus root such as `.git`,
`node_modules`, `dist`, and `build`, along with doc2graph run/cache artifacts
and Python package metadata such as `.doc2graph-runs`, `.doc2graph-cache.json`,
`DOC2GRAPH_PROGRESS.md`, `DOC2GRAPH_NEXT_PROMPT.md`, and `*.egg-info`. It
emits a corpus root plus folder/file nodes linked to each extracted document
graph. It resolves explicit relative links such as `[ADR](adr/cache.md)` into
corpus `links_to` edges, then adds deterministic cross-document `mentions`
edges when one corpus file explicitly names another file's title, section,
decision, table, or path-derived stem:

```bash
doc2graph ./knowledge-base --graph all --output corpus.graph.json
doc2graph ./knowledge-base --graph decision --include "adr/**" --output adr.graph.json
doc2graph ./exports --graph all --max-files 500 --max-file-bytes 10485760
doc2graph ./exports --graph all --max-files 500 --stop-after-max-files
doc2graph ./exports --graph all --max-total-bytes 1073741824 --output corpus.graph.json
doc2graph ./exports --graph all --max-depth 2 --output corpus.graph.json
doc2graph ./exports --graph all --max-scan-entries 100000 --output corpus.graph.json
doc2graph ./exports --graph all --extension md --extension pdf --output corpus.graph.json
doc2graph ./exports --graph all --follow-symlinks --output corpus.graph.json
doc2graph ./exports --graph all --max-file-reference-links 50000 --output corpus.graph.json
doc2graph ./exports --graph all --max-cross-document-links 50000 --output corpus.graph.json
doc2graph ./exports --graph all --skip-report-limit 25 --output corpus.graph.json
doc2graph ./exports --graph all --cache .doc2graph-cache.json --output corpus.graph.json
```

Large corpora are handled by deterministic limits:

- `--max-files N`: select at most N supported files. By default doc2graph
  continues scanning to count later supported files as `max_files_exceeded`
  skips, preserving complete skipped-file counts for the visited tree.
- `--stop-after-max-files`: stop scanning at the first supported file beyond
  `--max-files`. This is useful for huge trees when bounded traversal matters
  more than complete excess-file counts; the manifest marks
  `max_files_scan_truncated: true`, `skipped_file_count_is_complete: false`,
  and `skipped_file_records_sha256_is_complete: false`.
- `--max-file-bytes N`: skip very large individual files and add a `skipped_file` node.
- `--max-file-bytes -1`: disable the per-file size guard.
- `--max-total-bytes N`: stop extracting files after the cumulative extracted
  byte budget is reached and report remaining files as skipped.
- `--max-total-bytes -1`: disable the cumulative byte guard.
- `--no-recursive`: only process files directly under the directory and
  report skipped subdirectories as `non_recursive_directory`.
- `--follow-symlinks`: extract symlinked files that appear inside a directory
  corpus. By default symlinked files are reported as `symlink_file` skips so a
  corpus run does not silently read documents through links. Symlinked
  directories are always skipped as `symlink_directory` to avoid traversal
  loops.
- `--max-depth N`: bound recursive descent by subdirectory depth; `0` keeps
  only files directly under the corpus root and reports pruned directories.
- `--max-scan-entries N`: stop the deterministic directory walk after
  inspecting N filesystem entries. This intentionally truncates traversal for
  very large trees; the manifest marks `max_scan_entries_reached` and
  `skipped_file_count_is_complete: false` because unvisited paths are not fully
  counted.
- `--include` / `--exclude`: repeatable glob filters or folder/file names for
  corpus subsets. For example, `--include adr` and `--include "adr/**"` both
  select files under `adr/`. Supported files outside an include filter are
  counted as `include_filter_mismatch` skips, with bounded sample nodes. Paths
  matched by user exclude filters are counted as `exclude_filter_match` skips.
- `--extension EXT`: repeatable suffix allowlist for supported files, such as
  `--extension md --extension pdf`. Supported files with other suffixes are
  counted as `extension_filter_mismatch` skips, while unsupported files are
  still reported separately as `unsupported_extension`.
- `--skip-report-limit N`: cap the total number of omitted files listed as
  `skipped_file` or extraction-error nodes while still preserving aggregate
  skip counts and a deterministic SHA-256 digest of skipped path records. The
  corpus manifest reports how many skips were materialized as nodes, how many
  were omitted by this cap, and whether the skip report was truncated.
- `--max-file-reference-links N`: cap explicit relative file `links_to` edges
  resolved from Markdown, HTML, and wiki-style links between selected corpus
  files. The manifest reports `file_reference_link_limit_reached` when
  additional candidate links were omitted by the cap.
- `--max-cross-document-links N`: cap the deterministic cross-document
  `mentions` edges added after per-file graphs are merged. This bounds the
  corpus-wide linking pass for very large trees while preserving deterministic
  edge order. The manifest reports `cross_document_link_limit_reached` when
  additional candidate links were omitted by the cap.
- `--cache PATH`: opt into a JSON cache that reuses unchanged per-file graph
  extraction across repeated corpus runs. If the cache file is inside the
  scanned corpus directory, doc2graph reserves it as an output artifact and
  does not extract it as a source document. Cache entries are also tied to a
  deterministic fingerprint of the loader and extractor code that can affect
  per-file graph output and a SHA-256 digest of the source file content, so
  stale entries are rebuilt after doc2graph changes or file edits even when
  size and mtime metadata are unchanged. Cache hits are validated from stable
  extraction inputs and content SHA-256, so metadata-only touches do not force
  rebuilds; when only stat fields changed, doc2graph refreshes the cached
  metadata and reports this as `cache_metadata_updates`. Cache entries for optional parser
  stacks also record the relevant installed package versions, such as
  `pypdf`, `pdf2image`, `pytesseract`, `Pillow`, `python-docx`, or
  `python-pptx`, so PDF, OCR, DOCX, and PPTX entries refresh when their
  loader dependency versions change. Text-like formats do not record unrelated
  optional dependency versions, so installing a PDF parser does not invalidate
  Markdown cache entries. Warm-cache runs leave the cache file untouched when
  its deterministic JSON payload would not change, and the manifest reports
  this as `cache_file_updated: false`. Warm entries also reuse cached content
  digests when file size, mtime, ctime, and inode match, avoiding redundant
  full-file hashing on large unchanged corpora while still falling back to a
  fresh SHA-256 when the stat signature changes.
  The manifest also reports `cache_load_status` (`missing`, `loaded`,
  `invalid_json`, `invalid_schema`, or `read_error`) plus
  `cache_entry_count_before` and `cache_entry_count_after`, so invalid cache
  files are visible instead of being silently indistinguishable from an empty
  first run. Cache writes are best-effort: if the graph extraction succeeds
  but the cache path cannot be written, the run still returns the graph and
  reports `cache_write_status: write_error` plus `cache_write_error` in the
  manifest. Cache pruning is limited to complete selections. Bounded traversal
  or extraction runs, such as `--max-files`, `--max-depth`,
  `--max-scan-entries`, custom `--max-file-bytes`, or `--max-total-bytes`,
  preserve warm entries for files outside the current run and report the
  reason in `cache_prune_status`.
- `--output PATH`: when the output file already exists inside the scanned
  corpus directory, doc2graph reserves it too, even if it has a supported
  document extension such as `.md` or `.txt`.
- `--refresh-cache`: rebuild cached entries while writing an updated cache.

Knowledge extraction also resolves numeric inline citations such as `[1]` and
author-year citations such as `(Smith, 2024)` or `(Lee et al., 2025)` to
matching entries in `# References`, `# Bibliography`, or `# Works Cited`
sections when those entries are present. Claim and evidence nodes keep their
own `cites` edges, and citation nodes connect to parsed reference entries with
`resolves_to`, preserving deterministic provenance for PageRank and context
selection.

Knowledge extraction also turns explicit definitions into graph structure.
Glossary-style lines such as `Personalized PageRank: ...` and simple sentences
such as `Context engineering is ...` become `definition` nodes connected to
their `concept` with `defines` and `defined_by` edges, while preserving section
and citation provenance.

Claim-to-evidence support edges are deterministic and conservative. Evidence
in the same section is preferred, and cross-section support is only added when
the claim and evidence share meaningful terms, which avoids noisy support paths
between unrelated claims in large documents.

Every directory graph includes a `corpus_manifest` node with selected-file
counts, total selected bytes, a deterministic selected-path ordering contract,
and SHA-256 digests for selected paths and selected file records,
skipped-file counts, a deterministic skipped-record digest, cache hit/miss/write/update counts when caching is
enabled, the content-digest and extraction fingerprint validation used for
cache entries,
the cache load status and before/after cache entry counts,
whether optional loader dependency versions were included in cache validation,
stale cache entries pruned for the current root and graph type,
whether cache pruning was safe for the current selection,
whether the cache file was actually updated,
content-digest reuse hit/miss counts for warm cache validation,
cache write status and write errors when an explicit cache path cannot be updated,
the number of deterministic cross-document mention links added and whether an
explicit cross-document link cap was reached,
the number of explicit relative file links resolved and whether an explicit
file-reference link cap was reached,
active include/exclude patterns, extracted-file byte counts, scan budget state,
whether a max-files limit intentionally truncated traversal,
failed-file counts, and skip reasons
such as unsupported extensions, include-filter mismatches, max-file limits,
cumulative byte limits, oversized files,
depth-pruned directories, default ignored generated paths, user excluded paths,
non-recursive skipped directories, symlinked directories, symlinked files,
inaccessible paths, reserved cache/output files, and per-file extraction errors. This
keeps large mixed-folder runs deterministic and auditable without requiring all
omitted paths to be materialized as graph nodes.

Selected `file` nodes also carry deterministic audit metadata. Successful files
are marked with `status: extracted`, files skipped by byte limits are marked
with `status: skipped` plus a `skip_reason`, and files that fail to load are
marked with `status: failed`, `error_type`, and `error_message` while the run
continues. Each selected file records its zero-based `extraction_order`, making
large corpus traversal reproducible and easy to compare against the manifest's
`selected_file_paths_sha256` without materializing a full path list in metadata.
When explicit caching is enabled, each file records whether its graph
came from a cache `hit`, cache `miss`, cache `refresh`, caching was `disabled`,
or extraction was skipped before the cache was `not_attempted`.

Document-local node IDs are scoped by source during file and corpus extraction,
so common headings such as `# Abstract`, `# Summary`, or `# Decision` remain
separate per file while preserving their original labels and source
provenance. This prevents corpus merges from silently dropping same-named
sections, claims, decision nodes, citations, references, URLs, or schema
tables from later files.

Decision extraction recognizes common ADR context bullets and standalone
prefixed lines such as
`Constraint:`, `Assumption:`, `Decision driver:`, and `Rationale:` as context
instead of generic decision text, then links them to the problem with
`has_context` and to later decision nodes with `informed_by`. This keeps the
reasoning trail traversable for PageRank without adding non-deterministic
inference.

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
bounded = DocumentGraph.from_directory("./docs", graph_type="all", max_depth=2)
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
│   │   └── code.py          # Python/JS/etc → tagged plain-text loading
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
