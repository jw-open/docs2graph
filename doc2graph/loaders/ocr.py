"""
OCR loader for doc2graph — extracts text from images and scanned PDFs.

Uses `pytesseract` (Apache-2.0) as the OCR engine, which wraps the open-source
Tesseract binary.  PDF support additionally requires `pdf2image` and the
`poppler` system library.

Installation
------------
Minimal (images only)::

    pip install doc2graph[ocr]
    # also install the Tesseract binary:
    # Ubuntu/Debian: sudo apt install tesseract-ocr
    # macOS:         brew install tesseract

PDF pages::

    pip install doc2graph[ocr,pdf]
    # Ubuntu/Debian: sudo apt install tesseract-ocr poppler-utils
    # macOS:         brew install tesseract poppler

Supported file types
--------------------
* Images  : PNG, JPG/JPEG, TIFF, BMP, GIF, WebP
* PDF     : each page is converted to an image and OCR'd sequentially
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Union

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}
_PDF_SUFFIX = ".pdf"


def _require_pytesseract():
    try:
        import pytesseract  # noqa: F401
        return pytesseract
    except ImportError:
        raise ImportError(
            "pytesseract is required for OCR. "
            "Install it with: pip install doc2graph[ocr]  "
            "(and ensure the Tesseract binary is installed on your system)"
        )


def _require_pil():
    try:
        from PIL import Image  # noqa: F401
        return Image
    except ImportError:
        raise ImportError(
            "Pillow is required for OCR. Install it with: pip install doc2graph[ocr]"
        )


def _require_pdf2image():
    try:
        from pdf2image import convert_from_path  # noqa: F401
        return convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image is required for PDF OCR. "
            "Install it with: pip install doc2graph[pdf]  "
            "(and ensure poppler-utils is installed on your system)"
        )


def load_image_ocr(
    path: str,
    lang: str = "eng",
    dpi: int = 300,
    tesseract_config: str = "",
) -> str:
    """
    Extract text from an image file using Tesseract OCR.

    Parameters
    ----------
    path : str
        Path to the image file (PNG, JPG, TIFF, BMP, GIF, or WebP).
    lang : str
        Tesseract language code(s).  Default ``"eng"`` (English).
        Multiple languages: ``"eng+fra"``.
    dpi : int
        Resolution hint passed to Tesseract.  Higher values improve accuracy
        on small text but increase processing time.  Default 300.
    tesseract_config : str
        Extra Tesseract CLI flags, e.g. ``"--psm 6"`` for uniform block layout.

    Returns
    -------
    str
        Raw extracted text.  Returns an empty string if no text is detected.

    Raises
    ------
    ImportError
        If pytesseract or Pillow are not installed.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not a supported image format.

    Example
    -------
    >>> from doc2graph.loaders.ocr import load_image_ocr
    >>> text = load_image_ocr("scan.png")
    >>> text = load_image_ocr("receipt.jpg", lang="eng+fra", dpi=400)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix.lower() not in _IMAGE_SUFFIXES:
        raise ValueError(
            f"Unsupported image format '{p.suffix}'. "
            f"Supported: {', '.join(sorted(_IMAGE_SUFFIXES))}"
        )

    pytesseract = _require_pytesseract()
    Image = _require_pil()

    img = Image.open(str(p))
    config = tesseract_config.strip()
    if dpi and "--dpi" not in config:
        config = f"--dpi {dpi} {config}".strip()

    text: str = pytesseract.image_to_string(img, lang=lang, config=config)
    return text.strip()


def load_pdf_ocr(
    path: str,
    lang: str = "eng",
    dpi: int = 300,
    pages: Optional[Union[int, List[int]]] = None,
    tesseract_config: str = "",
) -> str:
    """
    Extract text from a scanned PDF by converting each page to an image and
    running Tesseract OCR.

    Requires ``pdf2image`` and the ``poppler`` system library in addition to
    ``pytesseract``.

    Parameters
    ----------
    path : str
        Path to the PDF file.
    lang : str
        Tesseract language code(s).  Default ``"eng"``.
    dpi : int
        Rendering resolution for each PDF page.  Higher = better accuracy,
        slower processing.  Default 300.
    pages : int or list[int], optional
        1-based page number(s) to extract.  ``None`` extracts all pages.
        Examples: ``pages=1`` (first page only), ``pages=[1, 3, 5]``.
    tesseract_config : str
        Extra Tesseract CLI flags (e.g. ``"--psm 6"``).

    Returns
    -------
    str
        Concatenated OCR text from all requested pages, separated by
        ``"\\n\\n--- Page N ---\\n\\n"`` markers.

    Raises
    ------
    ImportError
        If pytesseract, Pillow, or pdf2image are not installed.
    FileNotFoundError
        If the PDF file does not exist.
    ValueError
        If the file is not a PDF.

    Example
    -------
    >>> from doc2graph.loaders.ocr import load_pdf_ocr
    >>> text = load_pdf_ocr("report.pdf")
    >>> first_page = load_pdf_ocr("report.pdf", pages=1)
    >>> selected = load_pdf_ocr("report.pdf", pages=[1, 2, 5])
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix.lower() != _PDF_SUFFIX:
        raise ValueError(f"Expected a PDF file, got '{p.suffix}'")

    convert_from_path = _require_pdf2image()
    pytesseract = _require_pytesseract()
    _require_pil()

    # Build page subset list (1-based)
    if pages is None:
        page_subset = None  # convert all
    elif isinstance(pages, int):
        page_subset = [pages]
    else:
        page_subset = sorted(set(pages))

    # Convert PDF pages to PIL images
    if page_subset is not None:
        images = convert_from_path(str(p), dpi=dpi, first_page=min(page_subset), last_page=max(page_subset))
        # pdf2image returns consecutive pages starting at first_page
        # Re-filter to exactly the requested page numbers
        first = min(page_subset)
        images = [img for i, img in enumerate(images, start=first) if i in page_subset]
        page_numbers = page_subset
    else:
        images = convert_from_path(str(p), dpi=dpi)
        page_numbers = list(range(1, len(images) + 1))

    config = tesseract_config.strip()
    if dpi and "--dpi" not in config:
        config = f"--dpi {dpi} {config}".strip()

    parts: List[str] = []
    for page_num, img in zip(page_numbers, images):
        page_text: str = pytesseract.image_to_string(img, lang=lang, config=config).strip()
        if page_text:
            parts.append(f"--- Page {page_num} ---\n\n{page_text}")

    return "\n\n".join(parts)


def load_ocr(
    path: str,
    lang: str = "eng",
    dpi: int = 300,
    pages: Optional[Union[int, List[int]]] = None,
    tesseract_config: str = "",
) -> str:
    """
    Unified OCR loader — auto-detects whether the input is an image or PDF.

    Routes to :func:`load_image_ocr` for images and :func:`load_pdf_ocr` for PDFs.

    Parameters
    ----------
    path : str
        Path to an image (PNG/JPG/TIFF/BMP/GIF/WebP) or PDF file.
    lang : str
        Tesseract language code(s).  Default ``"eng"``.
    dpi : int
        Rendering DPI.  Default 300.
    pages : int or list[int], optional
        Page selection for PDFs (ignored for images).
    tesseract_config : str
        Extra Tesseract CLI flags.

    Returns
    -------
    str
        Extracted text.

    Raises
    ------
    ValueError
        If the file extension is not supported.

    Example
    -------
    >>> from doc2graph.loaders.ocr import load_ocr
    >>> text = load_ocr("invoice.png")
    >>> text = load_ocr("contract.pdf", pages=[1, 2])
    """
    suffix = Path(path).suffix.lower()
    if suffix == _PDF_SUFFIX:
        return load_pdf_ocr(path, lang=lang, dpi=dpi, pages=pages, tesseract_config=tesseract_config)
    elif suffix in _IMAGE_SUFFIXES:
        return load_image_ocr(path, lang=lang, dpi=dpi, tesseract_config=tesseract_config)
    else:
        supported = sorted(_IMAGE_SUFFIXES | {_PDF_SUFFIX})
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: {', '.join(supported)}"
        )
