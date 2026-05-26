"""
HTML loader for doc2graph — extracts readable text from HTML files or strings.

Uses the standard library ``html.parser`` — no extra dependencies.
Script, style, and meta tags are stripped; visible text is returned in
reading order.

No installation required.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import List


# Tags whose content should be completely discarded
_SKIP_TAGS = {"script", "style", "head", "meta", "link", "noscript"}

# Tags that imply a paragraph break
_BLOCK_TAGS = {
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "tr", "td", "th", "article", "section",
    "header", "footer", "nav", "aside", "blockquote", "pre",
    "br", "hr",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []
        self._skip_depth: int = 0
        self._current_tag: str = ""

    def handle_starttag(self, tag: str, attrs) -> None:
        self._current_tag = tag.lower()
        if tag.lower() in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag.lower() in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag.lower() in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse excessive whitespace / blank lines
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def load_html(path: str) -> str:
    """
    Extract visible text from an HTML file.

    Script and style blocks are discarded.  Block-level tags produce
    paragraph breaks.  The result is plain text suitable for downstream
    graph extraction.

    Parameters
    ----------
    path : str
        Path to the ``.html`` or ``.htm`` file.

    Returns
    -------
    str
        Readable plain text extracted from the HTML.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not ``.html`` or ``.htm``.

    Example
    -------
    >>> from docs2graph.loaders.html import load_html
    >>> text = load_html("page.html")
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix.lower() not in {".html", ".htm"}:
        raise ValueError(f"Expected an .html/.htm file, got '{p.suffix}'")

    from .text import _detect_encoding
    raw = p.read_bytes()
    enc = _detect_encoding(raw)
    raw_html = raw.decode(enc, errors="replace")
    return parse_html_string(raw_html)


def parse_html_string(html: str) -> str:
    """
    Extract visible text from an HTML string.

    Useful when the HTML content is already in memory (e.g. fetched from a URL).

    Parameters
    ----------
    html : str
        Raw HTML content.

    Returns
    -------
    str
        Plain text.

    Example
    -------
    >>> from docs2graph.loaders.html import parse_html_string
    >>> text = parse_html_string("<h1>Hello</h1><p>World</p>")
    >>> assert "Hello" in text and "World" in text
    """
    extractor = _TextExtractor()
    extractor.feed(html)
    return extractor.get_text()
