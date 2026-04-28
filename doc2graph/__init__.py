"""
doc2graph — extract knowledge graphs from documents.

Query what's in your files without dumping the whole text into a prompt.
"""

from .graph import DocumentGraph
from .ranking import personalized_page_rank
from .types import make_node, make_edge
from .extractors.schema import extract_schema_graph
from .loaders.ocr import load_ocr, load_image_ocr, load_pdf_ocr

__version__ = "0.2.0"
__all__ = [
    "DocumentGraph",
    "personalized_page_rank",
    "make_node",
    "make_edge",
    "extract_schema_graph",
    "load_ocr",
    "load_image_ocr",
    "load_pdf_ocr",
]
