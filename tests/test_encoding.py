"""
Tests for encoding detection and multi-encoding support in text/HTML/CSV loaders.
"""

import pytest
from doc2graph.loaders.text import load_text, _detect_encoding
from doc2graph.loaders.html import load_html
from doc2graph.loaders.csv import load_csv
from doc2graph.loaders.auto import load_document
from doc2graph.loaders.markdown import load_markdown


# ---------------------------------------------------------------------------
# _detect_encoding
# ---------------------------------------------------------------------------

class TestDetectEncoding:
    def test_utf8_bom_detected(self):
        raw = b"\xef\xbb\xbfHello"
        assert _detect_encoding(raw) == "utf-8-sig"

    def test_utf16_le_bom_detected(self):
        raw = b"\xff\xfeHello"
        assert _detect_encoding(raw) == "utf-16"

    def test_utf16_be_bom_detected(self):
        raw = b"\xfe\xffHello"
        assert _detect_encoding(raw) == "utf-16"

    def test_utf32_bom_detected(self):
        raw = b"\xff\xfe\x00\x00Hello"
        assert _detect_encoding(raw) == "utf-32"

    def test_plain_ascii_falls_back_gracefully(self):
        raw = b"Hello, world!"
        enc = _detect_encoding(raw)
        assert isinstance(enc, str)
        assert len(enc) > 0


# ---------------------------------------------------------------------------
# load_text encoding
# ---------------------------------------------------------------------------

class TestLoadTextEncoding:
    def test_utf8_explicit(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_bytes("Hello café".encode("utf-8"))
        text = load_text(str(f), encoding="utf-8")
        assert "café" in text

    def test_utf16_explicit(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_bytes("Hello".encode("utf-16"))
        text = load_text(str(f), encoding="utf-16")
        assert "Hello" in text

    def test_iso8859_explicit(self, tmp_path):
        f = tmp_path / "doc.txt"
        content = "Ren\xe9 Descartes"  # é in latin-1
        f.write_bytes(content.encode("iso-8859-1"))
        text = load_text(str(f), encoding="iso-8859-1")
        assert "René" in text

    def test_utf8_bom_auto_detected(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_bytes(b"\xef\xbb\xbfHello BOM")
        text = load_text(str(f))  # auto-detect
        assert "Hello BOM" in text
        assert "\ufeff" not in text  # BOM stripped by utf-8-sig

    def test_utf16_bom_auto_detected(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_bytes("UTF-16 content".encode("utf-16"))  # includes BOM
        text = load_text(str(f))  # auto-detect
        assert "UTF-16 content" in text

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_text("/no/such/file.txt")

    def test_errors_replace_by_default(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_bytes(b"Hello \xff world")  # invalid UTF-8 byte
        text = load_text(str(f))
        assert "Hello" in text
        assert "world" in text  # didn't crash


# ---------------------------------------------------------------------------
# Markdown loader auto-detection
# ---------------------------------------------------------------------------

class TestMarkdownEncoding:
    def test_utf8_bom_markdown(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_bytes(b"\xef\xbb\xbf# Title\n\nBOM content")

        text = load_markdown(str(f))

        assert text.startswith("# Title")
        assert "\ufeff" not in text

    def test_latin1_markdown_uses_text_loader_detection(self, tmp_path):
        f = tmp_path / "legacy.md"
        f.write_bytes("# R\xe9sum\xe9\n\nRen\xe9 writes notes.".encode("iso-8859-1"))

        text = load_document(str(f))

        assert "René" in text
        assert "notes" in text


# ---------------------------------------------------------------------------
# HTML loader auto-detection
# ---------------------------------------------------------------------------

class TestHtmlEncoding:
    def test_utf8_bom_html(self, tmp_path):
        f = tmp_path / "page.html"
        f.write_bytes(b"\xef\xbb\xbf<p>BOM content</p>")
        text = load_html(str(f))
        assert "BOM content" in text

    def test_latin1_html(self, tmp_path):
        f = tmp_path / "page.html"
        f.write_bytes("<p>Ren\xe9</p>".encode("iso-8859-1"))
        # With replacement errors this should still load
        text = load_html(str(f))
        assert "Ren" in text  # might be replacement char but shouldn't crash


# ---------------------------------------------------------------------------
# CSV loader auto-detection
# ---------------------------------------------------------------------------

class TestCsvEncoding:
    def test_utf8_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_bytes("name,city\nAlice,New York\n".encode("utf-8"))
        text = load_csv(str(f))
        assert "Alice" in text

    def test_utf8_bom_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_bytes(b"\xef\xbb\xbfname,city\nAlice,NY\n")
        text = load_csv(str(f))
        assert "Alice" in text

    def test_latin1_csv_no_crash(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_bytes("name,city\nRen\xe9,Paris\n".encode("iso-8859-1"))
        # Should not crash (errors="replace")
        text = load_csv(str(f))
        assert "Paris" in text
