"""Markdown file loader — returns raw text."""

from pathlib import Path


def load_markdown(path: str) -> str:
    """Read a Markdown file and return its contents as a string."""
    return Path(path).read_text(encoding="utf-8")
