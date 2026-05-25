"""Markdown file loader -- returns raw text with text-loader encoding support."""

from .text import load_text


def load_markdown(path: str) -> str:
    """Read a Markdown file and return its contents as a string."""
    return load_text(path)
