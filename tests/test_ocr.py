"""
Tests for OCR loaders (doc2graph.loaders.ocr).

All tests mock pytesseract / PIL / pdf2image so no system binaries are needed.
Imports of load_* functions are done at module level to avoid numpy re-import
issues on Python 3.14.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import at module level — the ocr module itself has no top-level heavy imports
from docs2graph.loaders.ocr import (
    _require_pdf2image,
    _require_pil,
    _require_pytesseract,
    load_image_ocr,
    load_ocr,
    load_pdf_ocr,
)

_MODULE = "docs2graph.loaders.ocr"


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _fake_tess(text: str = "hello world") -> MagicMock:
    m = MagicMock()
    m.image_to_string.return_value = text
    return m


def _fake_pil_image_mod() -> MagicMock:
    """Return a fake PIL.Image module (open returns a MagicMock image)."""
    mod = MagicMock()
    mod.open.return_value = MagicMock()
    return mod


# ---------------------------------------------------------------------------
# _require_* helpers
# ---------------------------------------------------------------------------

class TestRequireHelpers:
    def test_missing_pytesseract_gives_helpful_message(self):
        with patch.dict(sys.modules, {"pytesseract": None}):
            with pytest.raises(ImportError, match="pytesseract"):
                _require_pytesseract()

    def test_missing_pil_gives_helpful_message(self):
        with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            with pytest.raises(ImportError, match="Pillow"):
                _require_pil()

    def test_missing_pdf2image_gives_helpful_message(self):
        with patch.dict(sys.modules, {"pdf2image": None}):
            with pytest.raises(ImportError, match="pdf2image"):
                _require_pdf2image()


# ---------------------------------------------------------------------------
# load_image_ocr
# ---------------------------------------------------------------------------

class TestLoadImageOcr:
    def test_returns_extracted_text(self, tmp_path):
        img = tmp_path / "scan.png"
        img.write_bytes(b"fake-png")
        tess = _fake_tess("Invoice total: $42.00")
        pil_mod = _fake_pil_image_mod()

        with patch(f"{_MODULE}._require_pytesseract", side_effect=lambda: tess), \
             patch(f"{_MODULE}._require_pil", side_effect=lambda: pil_mod):
            text = load_image_ocr(str(img))

        assert "42.00" in text

    def test_strips_leading_trailing_whitespace(self, tmp_path):
        img = tmp_path / "scan.jpg"
        img.write_bytes(b"fake-jpg")
        tess = _fake_tess("  some text   \n\n")
        pil_mod = _fake_pil_image_mod()

        with patch(f"{_MODULE}._require_pytesseract", side_effect=lambda: tess), \
             patch(f"{_MODULE}._require_pil", side_effect=lambda: pil_mod):
            text = load_image_ocr(str(img))

        assert text == "some text"

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image_ocr("/nonexistent/path/file.png")

    def test_unsupported_extension_raises(self, tmp_path):
        bad = tmp_path / "file.docx"
        bad.write_bytes(b"data")
        with pytest.raises(ValueError, match="Unsupported image format"):
            load_image_ocr(str(bad))

    def test_lang_passed_to_tesseract(self, tmp_path):
        img = tmp_path / "scan.png"
        img.write_bytes(b"fake")
        tess = _fake_tess("bonjour")
        pil_mod = _fake_pil_image_mod()

        with patch(f"{_MODULE}._require_pytesseract", side_effect=lambda: tess), \
             patch(f"{_MODULE}._require_pil", side_effect=lambda: pil_mod):
            load_image_ocr(str(img), lang="fra")

        kwargs = tess.image_to_string.call_args[1]
        assert kwargs["lang"] == "fra"

    def test_dpi_included_in_config_string(self, tmp_path):
        img = tmp_path / "scan.tiff"
        img.write_bytes(b"fake")
        tess = _fake_tess("text")
        pil_mod = _fake_pil_image_mod()

        with patch(f"{_MODULE}._require_pytesseract", side_effect=lambda: tess), \
             patch(f"{_MODULE}._require_pil", side_effect=lambda: pil_mod):
            load_image_ocr(str(img), dpi=400)

        config = tess.image_to_string.call_args[1]["config"]
        assert "400" in config

    @pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"])
    def test_all_supported_extensions_accepted(self, tmp_path, ext):
        img = tmp_path / f"file{ext}"
        img.write_bytes(b"fake")
        tess = _fake_tess("text")
        pil_mod = _fake_pil_image_mod()

        with patch(f"{_MODULE}._require_pytesseract", side_effect=lambda: tess), \
             patch(f"{_MODULE}._require_pil", side_effect=lambda: pil_mod):
            result = load_image_ocr(str(img))

        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# load_pdf_ocr
# ---------------------------------------------------------------------------

class TestLoadPdfOcr:
    def _run(self, pdf_path, pages_texts, pages=None):
        fake_imgs = [MagicMock() for _ in pages_texts]
        fake_convert = MagicMock(return_value=fake_imgs)
        tess = MagicMock()
        tess.image_to_string.side_effect = list(pages_texts)
        pil_mod = _fake_pil_image_mod()

        with patch(f"{_MODULE}._require_pdf2image", side_effect=lambda: fake_convert), \
             patch(f"{_MODULE}._require_pytesseract", side_effect=lambda: tess), \
             patch(f"{_MODULE}._require_pil", side_effect=lambda: pil_mod):
            kwargs = {"pages": pages} if pages is not None else {}
            return load_pdf_ocr(str(pdf_path), **kwargs)

    def test_returns_text_from_all_pages(self, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-fake")
        text = self._run(pdf, ["Page one text", "Page two text"])
        assert "Page one text" in text
        assert "Page two text" in text
        assert "--- Page 1 ---" in text
        assert "--- Page 2 ---" in text

    def test_single_page_selection(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-fake")
        text = self._run(pdf, ["page one only"], pages=1)
        assert "page one only" in text

    def test_empty_pages_omitted_from_output(self, tmp_path):
        pdf = tmp_path / "sparse.pdf"
        pdf.write_bytes(b"%PDF-fake")
        text = self._run(pdf, ["", "real content"])
        assert "--- Page 1 ---" not in text
        assert "real content" in text

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_pdf_ocr("/no/such/file.pdf")

    def test_non_pdf_raises(self, tmp_path):
        bad = tmp_path / "image.png"
        bad.write_bytes(b"data")
        with pytest.raises(ValueError, match="Expected a PDF"):
            load_pdf_ocr(str(bad))

    def test_page_markers_present(self, tmp_path):
        pdf = tmp_path / "multi.pdf"
        pdf.write_bytes(b"%PDF-fake")
        text = self._run(pdf, ["alpha", "beta", "gamma"])
        for marker in ["--- Page 1 ---", "--- Page 2 ---", "--- Page 3 ---"]:
            assert marker in text


# ---------------------------------------------------------------------------
# load_ocr (unified dispatcher)
# ---------------------------------------------------------------------------

class TestLoadOcr:
    def test_routes_png_to_image_ocr(self, tmp_path):
        img = tmp_path / "x.png"
        img.write_bytes(b"fake")
        with patch(f"{_MODULE}.load_image_ocr", return_value="img text") as mock_img, \
             patch(f"{_MODULE}.load_pdf_ocr", return_value="pdf text"):
            result = load_ocr(str(img))
        mock_img.assert_called_once()
        assert result == "img text"

    def test_routes_jpg_to_image_ocr(self, tmp_path):
        img = tmp_path / "x.jpg"
        img.write_bytes(b"fake")
        with patch(f"{_MODULE}.load_image_ocr", return_value="img text") as mock_img, \
             patch(f"{_MODULE}.load_pdf_ocr", return_value="pdf text"):
            result = load_ocr(str(img))
        mock_img.assert_called_once()
        assert result == "img text"

    def test_routes_pdf_to_pdf_ocr(self, tmp_path):
        pdf = tmp_path / "x.pdf"
        pdf.write_bytes(b"%PDF")
        with patch(f"{_MODULE}.load_pdf_ocr", return_value="pdf text") as mock_pdf, \
             patch(f"{_MODULE}.load_image_ocr", return_value="img text"):
            result = load_ocr(str(pdf))
        mock_pdf.assert_called_once()
        assert result == "pdf text"

    def test_unsupported_type_raises(self, tmp_path):
        bad = tmp_path / "file.xlsx"
        bad.write_bytes(b"data")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_ocr(str(bad))

    def test_pages_forwarded_to_pdf_loader(self, tmp_path):
        pdf = tmp_path / "x.pdf"
        pdf.write_bytes(b"%PDF")
        with patch(f"{_MODULE}.load_pdf_ocr", return_value="text") as mock_pdf, \
             patch(f"{_MODULE}.load_image_ocr", return_value="text"):
            load_ocr(str(pdf), pages=[1, 3])
        call_kwargs = mock_pdf.call_args[1]
        assert call_kwargs.get("pages") == [1, 3]
