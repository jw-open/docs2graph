"""
Tests for DOCX, PPTX, HTML, and CSV loaders.
DOCX/PPTX tests mock python-docx / python-pptx so no Office install needed.
HTML/CSV tests use no external dependencies.
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from docs2graph.loaders.html import load_html, parse_html_string
from docs2graph.loaders.csv import load_csv
from docs2graph.loaders.json import load_json
from docs2graph.loaders.auto import load_document

_DOCX_MOD = "doc2graph.loaders.docx"
_PPTX_MOD = "doc2graph.loaders.pptx"


# ---------------------------------------------------------------------------
# HTML loader (no mocking needed — uses stdlib only)
# ---------------------------------------------------------------------------

class TestLoadHtml:
    def test_extracts_paragraph_text(self, tmp_path):
        f = tmp_path / "page.html"
        f.write_text("<html><body><p>Hello world</p></body></html>")
        assert "Hello world" in load_html(str(f))

    def test_strips_script_tags(self, tmp_path):
        f = tmp_path / "page.html"
        f.write_text("<html><body><script>alert('x')</script><p>Visible</p></body></html>")
        text = load_html(str(f))
        assert "alert" not in text
        assert "Visible" in text

    def test_strips_style_tags(self, tmp_path):
        f = tmp_path / "page.html"
        f.write_text("<html><head><style>body{color:red}</style></head><body><p>Content</p></body></html>")
        text = load_html(str(f))
        assert "color" not in text
        assert "Content" in text

    def test_heading_text_extracted(self, tmp_path):
        f = tmp_path / "page.html"
        f.write_text("<h1>Title</h1><h2>Subtitle</h2><p>Body</p>")
        text = load_html(str(f))
        assert "Title" in text
        assert "Subtitle" in text
        assert "Body" in text

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_html("/no/such/file.html")

    def test_wrong_extension_raises(self, tmp_path):
        f = tmp_path / "page.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match=".html"):
            load_html(str(f))

    def test_htm_extension_accepted(self, tmp_path):
        f = tmp_path / "page.htm"
        f.write_text("<p>works</p>")
        assert "works" in load_html(str(f))

    def test_parse_html_string_no_file(self):
        text = parse_html_string("<h1>Heading</h1><p>Para</p>")
        assert "Heading" in text
        assert "Para" in text

    def test_nested_tags_extracted(self):
        text = parse_html_string("<div><ul><li>Item 1</li><li>Item 2</li></ul></div>")
        assert "Item 1" in text
        assert "Item 2" in text

    def test_empty_html_returns_empty_string(self):
        assert parse_html_string("") == ""


# ---------------------------------------------------------------------------
# CSV loader (no mocking needed — uses stdlib only)
# ---------------------------------------------------------------------------

class TestLoadCsv:
    def _write_csv(self, tmp_path, content: str, name="data.csv") -> str:
        f = tmp_path / name
        f.write_text(content, encoding="utf-8")
        return str(f)

    def test_basic_csv_text(self, tmp_path):
        path = self._write_csv(tmp_path, "name,city\nAlice,NY\nBob,LA\n")
        text = load_csv(path)
        assert "Alice" in text
        assert "NY" in text
        assert "Bob" in text

    def test_header_summary_line_included(self, tmp_path):
        path = self._write_csv(tmp_path, "name,age\nAlice,30\n")
        text = load_csv(path, include_header=True)
        assert "Columns:" in text
        assert "name" in text
        assert "age" in text

    def test_header_summary_excluded(self, tmp_path):
        path = self._write_csv(tmp_path, "name,age\nAlice,30\n")
        text = load_csv(path, include_header=False)
        assert "Columns:" not in text

    def test_column_value_pairs_format(self, tmp_path):
        path = self._write_csv(tmp_path, "product,price\nWidget,9.99\n")
        text = load_csv(path, include_header=False)
        assert "product: Widget" in text
        assert "price: 9.99" in text

    def test_max_rows_limits_output(self, tmp_path):
        rows = "id,val\n" + "\n".join(f"{i},{i*2}" for i in range(100))
        path = self._write_csv(tmp_path, rows)
        text = load_csv(path, max_rows=5, include_header=False)
        lines = [l for l in text.strip().split("\n") if l]
        assert len(lines) == 5

    def test_tsv_delimiter(self, tmp_path):
        path = self._write_csv(tmp_path, "name\tcity\nAlice\tNY\n", name="data.tsv")
        text = load_csv(path, delimiter="\t")
        assert "Alice" in text
        assert "NY" in text

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/data.csv")

    def test_wrong_extension_raises(self, tmp_path):
        f = tmp_path / "data.xlsx"
        f.write_text("a,b\n1,2")
        with pytest.raises(ValueError, match=".csv"):
            load_csv(str(f))

    def test_empty_rows_skipped(self, tmp_path):
        path = self._write_csv(tmp_path, "name,val\nAlice,1\n,,\nBob,2\n")
        text = load_csv(path, include_header=False)
        assert "Alice" in text
        assert "Bob" in text
        lines = [l for l in text.strip().split("\n") if l]
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# JSON loader (no mocking needed — uses stdlib only)
# ---------------------------------------------------------------------------

class TestLoadJson:
    def test_json_object_renders_sorted_key_paths(self, tmp_path):
        f = tmp_path / "record.json"
        f.write_text(
            '{"title":"Cache ADR","metadata":{"owner":"platform"},"tags":["cache","corpus"]}',
            encoding="utf-8",
        )

        text = load_json(str(f))
        lines = text.splitlines()

        assert lines == [
            "metadata.owner: platform",
            "tags.0: cache",
            "tags.1: corpus",
            "title: Cache ADR",
        ]

    def test_auto_loader_uses_structured_json_loader(self, tmp_path):
        f = tmp_path / "record.json"
        f.write_text('{"title":"Cache ADR"}', encoding="utf-8")

        assert load_document(str(f)) == "title: Cache ADR"

    def test_jsonl_preserves_record_order(self, tmp_path):
        f = tmp_path / "events.jsonl"
        f.write_text(
            '{"event":"created","id":1}\n\n{"event":"updated","id":2}\n',
            encoding="utf-8",
        )

        text = load_json(str(f))

        assert text.splitlines() == [
            "record 1.event: created",
            "record 1.id: 1",
            "record 2.event: updated",
            "record 2.id: 2",
        ]

    def test_max_items_limits_top_level_json_items(self, tmp_path):
        f = tmp_path / "record.json"
        f.write_text('{"b":2,"a":1,"c":3}', encoding="utf-8")

        assert load_json(str(f), max_items=2).splitlines() == ["a: 1", "b: 2"]

    def test_invalid_jsonl_reports_line_number(self, tmp_path):
        f = tmp_path / "events.jsonl"
        f.write_text('{"ok":true}\nnot json\n', encoding="utf-8")

        with pytest.raises(ValueError, match="line 2"):
            load_json(str(f))

    def test_wrong_extension_raises(self, tmp_path):
        f = tmp_path / "record.txt"
        f.write_text('{"title":"Nope"}', encoding="utf-8")

        with pytest.raises(ValueError, match=".json"):
            load_json(str(f))


# ---------------------------------------------------------------------------
# DOCX loader (mocked)
# ---------------------------------------------------------------------------

class TestLoadDocx:
    def _fake_docx_module(self, paragraphs=None, tables=None):
        """Build a fake python-docx Document."""
        para_objs = []
        for text in (paragraphs or []):
            p = MagicMock()
            p.text = text
            para_objs.append(p)

        table_objs = []
        for tbl in (tables or []):
            t = MagicMock()
            rows = []
            for row_cells in tbl:
                row = MagicMock()
                cells = []
                for cell_text in row_cells:
                    c = MagicMock()
                    c.text = cell_text
                    cells.append(c)
                row.cells = cells
                rows.append(row)
            t.rows = rows
            table_objs.append(t)

        doc = MagicMock()
        doc.paragraphs = para_objs
        doc.tables = table_objs

        docx_mod = MagicMock()
        docx_mod.Document.return_value = doc
        return docx_mod

    def test_paragraphs_extracted(self, tmp_path):
        f = tmp_path / "doc.docx"
        f.write_bytes(b"fake")
        docx_mod = self._fake_docx_module(paragraphs=["Hello", "World"])

        with patch(f"{_DOCX_MOD}._require_docx", return_value=docx_mod):
            from docs2graph.loaders.docx import load_docx
            text = load_docx(str(f))

        assert "Hello" in text
        assert "World" in text

    def test_empty_paragraphs_skipped(self, tmp_path):
        f = tmp_path / "doc.docx"
        f.write_bytes(b"fake")
        docx_mod = self._fake_docx_module(paragraphs=["Good", "", "  ", "Text"])

        with patch(f"{_DOCX_MOD}._require_docx", return_value=docx_mod):
            from docs2graph.loaders.docx import load_docx
            text = load_docx(str(f))

        lines = [l for l in text.split("\n") if l]
        assert len(lines) == 2

    def test_table_cells_extracted(self, tmp_path):
        f = tmp_path / "doc.docx"
        f.write_bytes(b"fake")
        docx_mod = self._fake_docx_module(
            paragraphs=[],
            tables=[[["Name", "Age"], ["Alice", "30"]]]
        )

        with patch(f"{_DOCX_MOD}._require_docx", return_value=docx_mod):
            from docs2graph.loaders.docx import load_docx
            text = load_docx(str(f))

        assert "Name" in text
        assert "Alice" in text

    def test_tables_excluded_when_flag_false(self, tmp_path):
        f = tmp_path / "doc.docx"
        f.write_bytes(b"fake")
        docx_mod = self._fake_docx_module(
            paragraphs=["Body text"],
            tables=[[["TableCell"]]]
        )

        with patch(f"{_DOCX_MOD}._require_docx", return_value=docx_mod):
            from docs2graph.loaders.docx import load_docx
            text = load_docx(str(f), include_tables=False)

        assert "TableCell" not in text
        assert "Body text" in text

    def test_file_not_found_raises(self):
        from docs2graph.loaders.docx import load_docx
        with pytest.raises(FileNotFoundError):
            load_docx("/no/such/file.docx")

    def test_wrong_extension_raises(self, tmp_path):
        f = tmp_path / "file.pdf"
        f.write_bytes(b"data")
        from docs2graph.loaders.docx import load_docx
        with pytest.raises(ValueError, match=".docx"):
            load_docx(str(f))

    def test_missing_python_docx_raises(self):
        from docs2graph.loaders.docx import _require_docx
        with patch.dict(sys.modules, {"docx": None}):
            with pytest.raises(ImportError, match="python-docx"):
                _require_docx()


# ---------------------------------------------------------------------------
# PPTX loader (mocked)
# ---------------------------------------------------------------------------

class TestLoadPptx:
    def _fake_pptx_presentation(self, slides_text):
        """slides_text: list of list of strings (per slide, per shape)."""
        slide_objs = []
        for slide_texts in slides_text:
            slide = MagicMock()
            shapes = []
            for text in slide_texts:
                shape = MagicMock()
                shape.has_text_frame = True
                # Build paragraphs with runs
                para = MagicMock()
                run = MagicMock()
                run.text = text
                para.runs = [run]
                shape.text_frame.paragraphs = [para]
                shapes.append(shape)
            slide.shapes = shapes
            slide_objs.append(slide)

        prs = MagicMock()
        prs.slides = slide_objs

        Presentation = MagicMock(return_value=prs)
        return Presentation

    def test_slide_text_extracted(self, tmp_path):
        f = tmp_path / "deck.pptx"
        f.write_bytes(b"fake")
        Presentation = self._fake_pptx_presentation([["Title slide"], ["Second slide"]])

        with patch(f"{_PPTX_MOD}._require_pptx", return_value=Presentation):
            from docs2graph.loaders.pptx import load_pptx
            text = load_pptx(str(f))

        assert "Title slide" in text
        assert "Second slide" in text

    def test_slide_markers_present_by_default(self, tmp_path):
        f = tmp_path / "deck.pptx"
        f.write_bytes(b"fake")
        Presentation = self._fake_pptx_presentation([["Slide one"], ["Slide two"]])

        with patch(f"{_PPTX_MOD}._require_pptx", return_value=Presentation):
            from docs2graph.loaders.pptx import load_pptx
            text = load_pptx(str(f))

        assert "--- Slide 1 ---" in text
        assert "--- Slide 2 ---" in text

    def test_slide_markers_suppressed(self, tmp_path):
        f = tmp_path / "deck.pptx"
        f.write_bytes(b"fake")
        Presentation = self._fake_pptx_presentation([["Content"]])

        with patch(f"{_PPTX_MOD}._require_pptx", return_value=Presentation):
            from docs2graph.loaders.pptx import load_pptx
            text = load_pptx(str(f), slide_markers=False)

        assert "--- Slide" not in text
        assert "Content" in text

    def test_empty_slides_skipped(self, tmp_path):
        f = tmp_path / "deck.pptx"
        f.write_bytes(b"fake")
        # Second slide has no text
        Presentation = self._fake_pptx_presentation([["Real content"], []])

        with patch(f"{_PPTX_MOD}._require_pptx", return_value=Presentation):
            from docs2graph.loaders.pptx import load_pptx
            text = load_pptx(str(f))

        assert "--- Slide 2 ---" not in text

    def test_file_not_found_raises(self):
        from docs2graph.loaders.pptx import load_pptx
        with pytest.raises(FileNotFoundError):
            load_pptx("/no/such/file.pptx")

    def test_wrong_extension_raises(self, tmp_path):
        f = tmp_path / "file.pdf"
        f.write_bytes(b"data")
        from docs2graph.loaders.pptx import load_pptx
        with pytest.raises(ValueError, match=".pptx"):
            load_pptx(str(f))

    def test_missing_python_pptx_raises(self):
        from docs2graph.loaders.pptx import _require_pptx
        with patch.dict(sys.modules, {"pptx": None}):
            with pytest.raises(ImportError, match="python-pptx"):
                _require_pptx()
