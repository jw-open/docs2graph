"""
doc2graph — extract knowledge graphs from documents.

Query what's in your files without dumping the whole text into a prompt.
"""

from .graph import DocumentGraph
from .ranking import personalized_page_rank
from .types import make_node, make_edge
from .extractors.schema import extract_schema_graph
from .loaders.ocr import load_ocr, load_image_ocr, load_pdf_ocr
from .loaders.docx import load_docx
from .loaders.pptx import load_pptx
from .loaders.html import load_html, parse_html_string
from .loaders.csv import load_csv

__version__ = "0.3.0"
__all__ = [
    "DocumentGraph",
    "personalized_page_rank",
    "make_node",
    "make_edge",
    "extract_schema_graph",
    "load_ocr",
    "load_image_ocr",
    "load_pdf_ocr",
    "load_docx",
    "load_pptx",
    "load_html",
    "parse_html_string",
    "load_csv",
]
