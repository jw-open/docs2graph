"""
Source code loader for doc2graph.

Loads source code files as plain text with optional language-tagged formatting
so LLMs understand the language context.  Uses only the standard library.

For **semantic code graph extraction** (AST parsing, call graphs, import
dependency graphs, class hierarchies) see the forthcoming ``code2graph``
package — that problem requires AST-level analysis and is out of scope for
doc2graph.

No extra dependencies required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .text import load_text

# Map file extension -> language name (for tagging)
_EXT_TO_LANG: dict = {
    ".py":    "python",
    ".js":    "javascript",
    ".ts":    "typescript",
    ".jsx":   "javascript",
    ".tsx":   "typescript",
    ".java":  "java",
    ".go":    "go",
    ".rs":    "rust",
    ".cpp":   "cpp",
    ".cc":    "cpp",
    ".c":     "c",
    ".h":     "c",
    ".hpp":   "cpp",
    ".cs":    "csharp",
    ".rb":    "ruby",
    ".php":   "php",
    ".swift": "swift",
    ".kt":    "kotlin",
    ".scala": "scala",
    ".r":     "r",
    ".sql":   "sql",
    ".sh":    "bash",
    ".bash":  "bash",
    ".zsh":   "bash",
    ".yaml":  "yaml",
    ".yml":   "yaml",
    ".json":  "json",
    ".toml":  "toml",
    ".xml":   "xml",
    ".md":    "markdown",
}

CODE_SUFFIXES = frozenset(_EXT_TO_LANG)


def detect_language(path: str) -> Optional[str]:
    """
    Return the language name for a file based on its extension.

    Returns ``None`` if the extension is not recognised.

    Example
    -------
    >>> detect_language("main.py")
    'python'
    >>> detect_language("app.ts")
    'typescript'
    >>> detect_language("unknown.xyz")  # None
    """
    return _EXT_TO_LANG.get(Path(path).suffix.lower())


def load_code(
    path: str,
    encoding: Optional[str] = None,
    tag_language: bool = True,
) -> str:
    """
    Load a source code file and return its content as a string.

    Optionally prepends a language tag line so downstream LLMs can identify
    the language without inspecting the content (e.g. ``# language: python``).

    This function loads code **as text** — it does not parse ASTs, extract
    call graphs, or build dependency graphs.  For semantic code graph
    extraction use the forthcoming ``code2graph`` package.

    Parameters
    ----------
    path : str
        Path to the source file.  Any extension is accepted; recognised
        extensions (see :func:`detect_language`) produce a language tag.
    encoding : str, optional
        Character encoding.  ``None`` triggers auto-detection (BOM + library).
    tag_language : bool
        If ``True`` (default) and the language is detected, prepend a comment
        line ``# language: <lang>`` to help LLMs identify the language.

    Returns
    -------
    str
        Source code as a Unicode string, optionally with a language tag.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    Example
    -------
    >>> from doc2graph.loaders.code import load_code
    >>> text = load_code("main.py")
    >>> text = load_code("app.js", tag_language=False)
    >>> text = load_code("legacy.py", encoding="latin-1")
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    content = load_text(str(p), encoding=encoding)

    if tag_language:
        lang = detect_language(str(p))
        if lang:
            content = f"# language: {lang}\n\n{content}"

    return content
