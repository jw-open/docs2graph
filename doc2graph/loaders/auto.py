"""File loader dispatch for common documentation formats."""

from __future__ import annotations

from pathlib import Path

from .text import load_text


def load_document(path: str) -> str:
    """
    Load a documentation file as plain text.

    The loader is intentionally dependency-light. Optional formats such as
    DOCX/PPTX work when their extras are installed; unsupported extensions
    fall back to text loading.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in {".md", ".markdown", ".mdx"}:
        from .markdown import load_markdown

        return load_markdown(path)
    if suffix in {".html", ".htm"}:
        from .html import load_html

        return load_html(path)
    if suffix == ".docx":
        from .docx import load_docx

        return load_docx(path)
    if suffix == ".pptx":
        from .pptx import load_pptx

        return load_pptx(path)
    if suffix in {".csv", ".tsv"}:
        from .csv import load_csv

        return load_csv(path)
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".pdf"}:
        from .ocr import load_ocr

        return load_ocr(path)

    return load_text(path)
