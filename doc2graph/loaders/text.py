"""Plain text file loader — returns raw text."""

from pathlib import Path


def load_text(path: str) -> str:
    """Read a plain text file and return its contents as a string."""
    return Path(path).read_text(encoding="utf-8")
