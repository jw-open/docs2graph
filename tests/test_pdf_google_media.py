from unittest.mock import MagicMock, patch

import pytest

from docs2graph.cli import build_graph
from docs2graph.extractors.media import extract_media_graph
from docs2graph.loaders.google import _to_export_url, load_google_doc
from docs2graph.loaders.pdf import load_pdf
from docs2graph.loaders.url import load_url


def test_google_doc_url_converts_to_text_export():
    url = "https://docs.google.com/document/d/doc123/edit"
    assert _to_export_url(url) == "https://docs.google.com/document/d/doc123/export?format=txt"


def test_google_sheet_url_converts_to_csv_export():
    url = "https://docs.google.com/spreadsheets/d/sheet123/edit#gid=0"
    assert _to_export_url(url) == "https://docs.google.com/spreadsheets/d/sheet123/export?format=csv"


def test_google_loader_decodes_plain_text_response():
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = b"Architecture notes"
    response.headers.get.return_value = "text/plain; charset=utf-8"

    with patch("docs2graph.loaders.google.urlopen", return_value=response):
        text = load_google_doc("https://docs.google.com/document/d/doc123/edit")

    assert text == "Architecture notes"


def test_generic_url_loader_parses_html_response():
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = b"<h1>Title</h1><script>x()</script><p>Body</p>"
    response.headers.get.return_value = "text/html; charset=utf-8"

    with patch("docs2graph.loaders.url.urlopen", return_value=response):
        text = load_url("https://example.com/page")

    assert "Title" in text
    assert "Body" in text
    assert "x()" not in text


def test_pdf_loader_uses_native_text(tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF")
    page = MagicMock()
    page.extract_text.return_value = "Native PDF text"
    reader = MagicMock()
    reader.pages = [page]
    pypdf = MagicMock()
    pypdf.PdfReader.return_value = reader

    with patch("docs2graph.loaders.pdf._require_pypdf", return_value=pypdf):
        text = load_pdf(str(pdf))

    assert "Native PDF text" in text
    assert "--- Page 1 ---" in text


def test_pdf_loader_falls_back_to_ocr_when_native_empty(tmp_path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF")
    page = MagicMock()
    page.extract_text.return_value = ""
    reader = MagicMock()
    reader.pages = [page]
    pypdf = MagicMock()
    pypdf.PdfReader.return_value = reader

    with patch("docs2graph.loaders.pdf._require_pypdf", return_value=pypdf), \
         patch("docs2graph.loaders.ocr.load_pdf_ocr", return_value="OCR text"):
        text = load_pdf(str(pdf))

    assert text == "OCR text"


def test_media_graph_records_ocr_and_chart_signals(tmp_path):
    image = tmp_path / "chart.png"
    image.write_bytes(b"fake")

    graph = extract_media_graph(str(image), text="Revenue trend by quarter with legend and y-axis")
    labels = {node["label"] for node in graph["nodes"]}
    types = {node.get("attributes", {}).get("type") for node in graph["nodes"]}

    assert "OCR text" in labels
    assert "Legend" in labels
    assert "chart_signal" in types


def test_cli_media_graph_uses_ocr_text(tmp_path):
    image = tmp_path / "chart.png"
    image.write_bytes(b"fake")

    with patch("docs2graph.loaders.ocr.load_ocr", return_value="Legend revenue by quarter"):
        graph = build_graph(str(image), graph_type="media")

    assert graph["current_node_id"].startswith("media:")
    assert any(node["label"] == "OCR text" for node in graph["nodes"])


def test_cli_accepts_media_graph_choice(tmp_path):
    image = tmp_path / "chart.png"
    image.write_bytes(b"fake")

    with patch("docs2graph.loaders.ocr.load_ocr", return_value="chart text"):
        graph = build_graph(str(image), graph_type="all")

    assert any(node.get("attributes", {}).get("type") == "media_document" for node in graph["nodes"])
