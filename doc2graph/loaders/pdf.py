"""PDF loader with native text extraction and OCR fallback."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union


def _require_pypdf():
    try:
        import pypdf

        return pypdf
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for native PDF text extraction. "
            "Install it with: pip install doc2graph[pdf]"
        ) from exc


def load_pdf(
    path: str,
    pages: Optional[Union[int, List[int]]] = None,
    ocr_fallback: bool = True,
) -> str:
    """
    Extract text from a PDF.

    Native embedded text is extracted with ``pypdf`` first. If the PDF appears
    scanned or image-only and ``ocr_fallback`` is true, the loader falls back to
    the existing Tesseract OCR path.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got '{p.suffix}'")

    text = _load_pdf_native(path, pages=pages)
    if text.strip() or not ocr_fallback:
        return text.strip()

    from .ocr import load_pdf_ocr

    return load_pdf_ocr(path, pages=pages)


def _load_pdf_native(path: str, pages: Optional[Union[int, List[int]]] = None) -> str:
    pypdf = _require_pypdf()
    reader = pypdf.PdfReader(path)
    page_indexes = _page_indexes(len(reader.pages), pages)

    parts: List[str] = []
    for page_index in page_indexes:
        page = reader.pages[page_index]
        page_text = (page.extract_text() or "").strip()
        if page_text:
            parts.append(f"--- Page {page_index + 1} ---\n\n{page_text}")
    return "\n\n".join(parts)


def _page_indexes(count: int, pages: Optional[Union[int, List[int]]]) -> List[int]:
    if pages is None:
        return list(range(count))
    requested = [pages] if isinstance(pages, int) else sorted(set(pages))
    indexes = []
    for page in requested:
        if page < 1 or page > count:
            continue
        indexes.append(page - 1)
    return indexes
