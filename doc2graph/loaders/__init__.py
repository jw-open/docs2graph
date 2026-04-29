from .text import load_text
from .markdown import load_markdown
from .ocr import load_ocr, load_image_ocr, load_pdf_ocr
from .docx import load_docx
from .pptx import load_pptx
from .html import load_html, parse_html_string
from .csv import load_csv

__all__ = [
    "load_text",
    "load_markdown",
    "load_ocr",
    "load_image_ocr",
    "load_pdf_ocr",
    "load_docx",
    "load_pptx",
    "load_html",
    "parse_html_string",
    "load_csv",
]
