"""
Plain text loader for doc2graph.

Supports explicit encoding selection (UTF-8, UTF-16, ISO-8859-1, etc.) and
optional auto-detection via ``charset-normalizer`` or ``chardet``.

No mandatory extra dependencies — auto-detection is an optional enhancement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def _detect_encoding(raw: bytes) -> str:
    """
    Try to detect the encoding of raw bytes.

    Uses ``charset-normalizer`` (preferred) or ``chardet`` if installed,
    falling back to UTF-8 with replacement.
    """
    # BOM-based detection first (reliable)
    if raw[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"
    # Check UTF-32 before UTF-16: UTF-32 LE BOM starts with \xff\xfe\x00\x00
    # which would be mistaken for UTF-16 LE (\xff\xfe) if checked first.
    if raw[:4] in (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff"):
        return "utf-32"
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return "utf-16"

    # Library-based detection
    try:
        from charset_normalizer import from_bytes
        result = from_bytes(raw).best()
        if result is not None:
            return str(result.encoding)
    except ImportError:
        pass

    try:
        import chardet
        detected = chardet.detect(raw)
        if detected and detected.get("encoding"):
            return detected["encoding"]
    except ImportError:
        pass

    return "utf-8"


def load_text(
    path: str,
    encoding: Optional[str] = None,
    errors: str = "replace",
) -> str:
    """
    Read a plain text file and return its contents as a string.

    Parameters
    ----------
    path : str
        Path to the text file.  Any extension is accepted.
    encoding : str, optional
        Character encoding to use (e.g. ``"utf-8"``, ``"utf-16"``,
        ``"iso-8859-1"``, ``"latin-1"``).  When ``None`` (default),
        the encoding is auto-detected: BOM inspection first, then
        ``charset-normalizer`` / ``chardet`` if installed, falling back
        to UTF-8 with replacement characters.
    errors : str
        Error handling strategy passed to ``open()``.  Default ``"replace"``
        (unknown bytes become ``\ufffd``).  Other options: ``"strict"``,
        ``"ignore"``.

    Returns
    -------
    str
        File contents as a Unicode string.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    Example
    -------
    >>> from doc2graph.loaders.text import load_text
    >>> text = load_text("doc.txt")                          # auto-detect
    >>> text = load_text("doc.txt", encoding="utf-16")       # explicit
    >>> text = load_text("legacy.txt", encoding="iso-8859-1")
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if encoding is None:
        raw = p.read_bytes()
        encoding = _detect_encoding(raw)

    return p.read_text(encoding=encoding, errors=errors)
