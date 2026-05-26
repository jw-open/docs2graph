"""Google Docs/Sheets/Slides loader for public or token-authorized URLs."""

from __future__ import annotations

import os
import re
from typing import Optional
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from .html import parse_html_string

_DOC_RE = re.compile(r"/document/d/([^/]+)")
_SHEET_RE = re.compile(r"/spreadsheets/d/([^/]+)")
_SLIDES_RE = re.compile(r"/presentation/d/([^/]+)")


def is_google_workspace_url(source: str) -> bool:
    host = urlparse(source).netloc.lower()
    return host.endswith("docs.google.com")


def load_google_doc(source: str, timeout: float = 30.0) -> str:
    """
    Load text from a Google Workspace URL.

    Public/exportable documents work without credentials. Private documents can
    be fetched when ``GOOGLE_DOCS_BEARER_TOKEN`` is set to a token with access.
    """
    export_url = _to_export_url(source)
    body, content_type = _fetch(export_url, timeout=timeout)
    if "html" in content_type:
        return parse_html_string(body.decode("utf-8", errors="replace"))
    return body.decode(_guess_encoding(content_type), errors="replace").strip()


def _to_export_url(source: str) -> str:
    parsed = urlparse(source)
    if "export" in parsed.path and parse_qs(parsed.query).get("format"):
        return source

    if match := _DOC_RE.search(parsed.path):
        return f"https://docs.google.com/document/d/{match.group(1)}/export?format=txt"
    if match := _SHEET_RE.search(parsed.path):
        return f"https://docs.google.com/spreadsheets/d/{match.group(1)}/export?format=csv"
    if match := _SLIDES_RE.search(parsed.path):
        return f"https://docs.google.com/presentation/d/{match.group(1)}/export/txt"

    raise ValueError(f"Unsupported Google Workspace URL: {source}")


def _fetch(url: str, timeout: float) -> tuple[bytes, str]:
    headers = {"User-Agent": "docs2graph/0.3"}
    token = os.environ.get("GOOGLE_DOCS_BEARER_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read(), response.headers.get("content-type", "")
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise PermissionError(
                "Google document is not public/exportable. Share it publicly "
                "or set GOOGLE_DOCS_BEARER_TOKEN with access."
            ) from exc
        raise


def _guess_encoding(content_type: Optional[str]) -> str:
    if not content_type:
        return "utf-8"
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("charset="):
            return part.split("=", 1)[1]
    return "utf-8"
