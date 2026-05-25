"""Generic URL loader for text and HTML sources."""

from __future__ import annotations

from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .google import is_google_workspace_url, load_google_doc
from .html import parse_html_string


def is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def load_url(source: str, timeout: float = 30.0) -> str:
    if is_google_workspace_url(source):
        return load_google_doc(source, timeout=timeout)

    request = Request(source, headers={"User-Agent": "doc2graph/0.3"})
    with urlopen(request, timeout=timeout) as response:
        body = response.read()
        content_type = response.headers.get("content-type", "")

    text = body.decode(_guess_encoding(content_type), errors="replace")
    if "html" in content_type or source.lower().endswith((".html", ".htm")):
        return parse_html_string(text)
    return text.strip()


def _guess_encoding(content_type: str) -> str:
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("charset="):
            return part.split("=", 1)[1]
    return "utf-8"
