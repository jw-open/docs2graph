# Contributing to doc2graph

Thank you for your interest in contributing. This document covers how to set up a development environment, run tests, and submit changes.

---

## Development setup

```bash
git clone https://github.com/jw-open/doc2graph
cd doc2graph
pip install -e ".[dev]"
```

For optional format support during development:

```bash
pip install -e ".[dev,pdf,docx,pptx,ocr]"
```

---

## Running tests

```bash
pytest tests/ -v
```

Run a specific test file:

```bash
pytest tests/test_graph.py -v
```

Run with coverage:

```bash
pip install pytest-cov
pytest tests/ --cov=doc2graph --cov-report=term-missing
```

Tests run on Python 3.9–3.12 in CI. Please ensure your changes pass on all supported versions before submitting.

---

## Project structure

```
doc2graph/
├── doc2graph/
│   ├── __init__.py        # Public API — keep this clean
│   ├── graph.py           # DocumentGraph class
│   ├── types.py           # Node / Edge TypedDicts
│   ├── ranking.py         # Personalized PageRank
│   ├── cli.py             # CLI entry point
│   ├── corpus.py          # Directory corpus processing
│   ├── matching.py        # Query matching for PageRank seeds
│   ├── loaders/           # One file per format
│   └── extractors/        # One file per graph type
├── tests/
├── examples/
├── benchmarks/
├── pyproject.toml
├── CHANGELOG.md
└── README.md
```

---

## Adding a loader

Loaders live in `doc2graph/loaders/`. Each loader takes a file path or content string and returns clean text + optional structured metadata.

1. Create `doc2graph/loaders/myformat.py`
2. Export a `load_myformat(path: str) -> dict` function returning `{"text": str, "meta": dict}`
3. Register it in `doc2graph/loaders/auto.py` — add the extension and import
4. Add tests in `tests/test_loaders_extra.py`
5. Update `[project.optional-dependencies]` in `pyproject.toml` if extra packages are needed

---

## Adding an extractor

Extractors live in `doc2graph/extractors/`. Each extractor takes a loaded document dict and returns nodes + edges.

1. Create `doc2graph/extractors/mytype.py`
2. Export `extract_mygraph(doc: dict) -> dict` returning `{"nodes": [...], "edges": [...]}`
3. Register it in `doc2graph/graph.py` — add to `EXTRACTORS` and the `--graph` CLI choices
4. Add tests in `tests/test_extractors.py`

---

## Submitting changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make changes, add tests
4. Run `pytest tests/ -v` — all tests must pass
5. Open a pull request against `main`

**Pull request checklist:**
- [ ] Tests pass (`pytest tests/ -v`)
- [ ] New behavior has test coverage
- [ ] Public API changes are reflected in the docstring and README
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

---

## Reporting bugs

Open an issue at https://github.com/jw-open/doc2graph/issues with:
- Python version and OS
- `doc2graph` version (`pip show doc2graph`)
- Minimal reproduction (file + command or code snippet)
- Expected vs actual output

---

## Design guidelines

- **Deterministic** — same input must always produce the same graph. Avoid random seeds, dict ordering assumptions, or timestamp-dependent IDs.
- **No LLM by default** — extraction is static rule-based. LLM-inferred nodes must be labeled `extraction_method: llm_inferred`.
- **Pure Python** — no required native extensions beyond numpy. Optional format loaders (PDF, OCR) use clearly documented extras.
- **Plain output** — graph output is plain JSON dicts and lists. No custom serialization classes.
- **Minimal dependencies** — core depends only on numpy. Keep optional deps scoped to their extras.

---

## License

By contributing you agree your changes will be licensed under Apache-2.0.
