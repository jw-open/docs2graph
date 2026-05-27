# Changelog

All notable changes to `doc2graph` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.1] — 2026-05-27

### Fixed

- Bumped `requires-python` to `>=3.10` — code uses `X | None` union syntax (PEP 604) which requires Python 3.10+; removed Python 3.9 from CI and publish matrices to fix the build

---

## [0.3.0] — 2026-05-26

### Added

**Corpus processing**
- Scan-only audit mode (`--scan-only`) — builds a deterministic corpus scan graph without loading files; useful for auditing large corpora before extraction
- Extension allowlist filter (`--extension EXT`, repeatable) — restrict corpus processing to specific file suffixes
- Symlink reporting — symlinked files reported as explicit `symlink_file` skip nodes; symlinked directories always skipped to prevent traversal loops
- `--follow-symlinks` flag to opt into extracting symlinked files
- `--stop-after-max-files` — truncates directory traversal at the first file beyond `--max-files` for bounded traversal on huge trees
- `--max-scan-entries N` — stop the directory walk after N filesystem entries
- `--skip-report-limit N` — cap the number of materialized skipped-file nodes while preserving aggregate skip counts and SHA-256 digests
- `--max-file-reference-links N` — cap explicit relative `links_to` edges from Markdown/HTML corpus links
- `--max-cross-document-links N` — cap deterministic cross-document `mentions` edges
- `--refresh-cache` flag — force rebuild of all cache entries while writing an updated cache

**Knowledge extraction**
- Markdown result tables materialized as `table` nodes; numeric/result rows exposed as `evidence` nodes with table row provenance
- Author-year citation resolution — `(Smith, 2024)` and `(Lee et al., 2025)` resolved to `# References` entries
- Definition extraction — glossary-style lines (`Term: definition`) and sentences (`X is ...`) become `definition` nodes with `defines`/`defined_by` edges
- Deterministic evidence support links — claim-to-evidence edges only when evidence is same-section or shares meaningful terms with the claim; prevents noisy support paths across unrelated claims

**Decision extraction**
- ADR context bullet recognition — `Constraint:`, `Assumption:`, `Decision driver:`, `Rationale:` become `context` nodes with `has_context` and `informed_by` edges
- Improved ADR status extraction
- Standalone prefixed context lines recognized without requiring full section headings

**Corpus cache**
- Cache entries validated against loader and extractor code fingerprints — stale entries rebuilt after `doc2graph` updates, not just file content changes
- Optional parser dependency version tracking in cache — PDF, OCR, DOCX, PPTX entries refresh when their parser package versions change; text-like format entries unaffected by unrelated optional installs
- Content-digest reuse for warm cache validation — avoids full-file hashing when stat signature (size, mtime, ctime, inode) is unchanged
- Cache write status and error reporting (`cache_write_status`, `cache_write_error`) — cache write failures do not abort extraction
- `cache_load_status` in corpus manifest (`missing`, `loaded`, `invalid_json`, `invalid_schema`, `read_error`)
- `cache_entry_count_before` / `cache_entry_count_after` in corpus manifest
- Warm-cache runs leave cache file untouched when payload is identical (`cache_file_updated: false`)
- Cache pruning limited to complete selections — bounded traversal runs preserve warm entries for files outside the current selection

**Deterministic corpus manifest**
- `corpus_manifest` node includes: selected-file counts, total selected bytes, selected-path SHA-256, skipped-file counts and digest, cache hit/miss/write/update counts, content-digest reuse counts, cross-document and file-reference link counts and caps, scan budget state, failed-file counts, skip reasons

**JSON loader**
- Deterministic structured-text extraction for config files, exports, and line-delimited JSONL records

### Fixed

- Corpus include folder matching — `--include adr` and `--include "adr/**"` now both correctly select files under `adr/`
- ADR decision status extraction for non-standard status values
- Cross-document link resolution for explicit relative file references (`[ADR](adr/cache.md)`)
- Node ID scoping for same-named sections across corpus files — prevents silent node drops during corpus merges

---

## [0.2.0] — 2026-03-15

### Added

- Multi-document corpus processing with directory input
- Cross-document `mentions` edges for deterministic corpus linking
- Corpus cache for incremental extraction across repeated runs
- Include/exclude glob filters for corpus subsets
- Corpus limits: `--max-files`, `--max-file-bytes`, `--max-total-bytes`, `--max-depth`
- `schema` graph type — table/entity graphs from data dictionaries
- `media` graph type — image metadata, OCR text, chart signals
- Google Docs, Sheets, and Slides URL loader (public export + bearer token)
- PPTX loader (`python-pptx`)
- PDF loader with OCR fallback (`pypdf`, `pdf2image`, `pytesseract`)
- CLI: `doc2graph-iterate` and `doc2graph-loop` commands
- GraphML export (`g.to_graphml()`)

### Fixed

- BOM-aware and legacy encoding handling for text-like files
- Relative citation resolution for numeric `[1]` citations

---

## [0.1.0] — 2026-01-20

### Added

- `DocumentGraph` class with `from_document()`, `from_markdown()`, `from_text()`, `from_directory()`
- `knowledge` graph type: sections, concepts, definitions, claims, evidence, citations, references, URLs
- `decision` graph type: problems, options, pros/cons, tradeoffs, decisions, consequences
- Personalized PageRank (`g.rank(query, k)`) for subgraph ranking
- Loaders: Markdown, plain text, HTML, CSV/TSV, source code, DOCX, JSON/JSONL
- CLI: `doc2graph <source> --graph TYPE --output PATH`
- JSON export (`g.to_json()`, `g.to_dict()`)
- Apache-2.0 license
- CI matrix: Python 3.9, 3.10, 3.11, 3.12

---

[0.3.0]: https://github.com/jw-open/doc2graph/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jw-open/doc2graph/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jw-open/doc2graph/releases/tag/v0.1.0
