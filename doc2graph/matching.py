"""
Soft token matching for doc2graph.

Handles plurals, compound words, and underscored identifiers without
requiring external NLP libraries. Same approach as graph2sql.matching.
"""

import re
from typing import Dict, List, Optional, Set


_SUFFIXES = ("ing", "tion", "ations", "ation", "ies", "es", "s", "ed")


def stem(token: str) -> str:
    """Minimal suffix-stripping stemmer for document/section matching."""
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: len(token) - len(suffix)]
    return token


def tokenize(text: str) -> List[str]:
    """Split on non-word characters and underscores; lowercase."""
    return [t for t in re.split(r"[\W_]+", text.lower()) if t]


def stemmed_tokens(text: str) -> Set[str]:
    """Return a set of stemmed tokens from text."""
    return {stem(t) for t in tokenize(text)}


def soft_match_score(
    query: str,
    label: str,
    content: str = "",
    attributes: Optional[Dict] = None,
) -> float:
    """
    Soft match score between a query and a document node.

    - Label (title/heading) matches: weighted 2x
    - Content (body text) matches: weighted 1x, capped to avoid long-doc bias
    - Attribute string values (url, type, alias): weighted 1x

    Returns a float >= 0.
    """
    query_stems = stemmed_tokens(query)
    if not query_stems:
        return 0.0

    score = 0.0

    # Label match — 2x weight
    label_stems = stemmed_tokens(label)
    score += sum(1 for t in label_stems if t in query_stems) * 2.0

    # Content match — 1x, use first 500 chars to avoid biasing long docs
    if content:
        preview = content[:500]
        content_stems = stemmed_tokens(preview)
        score += sum(1 for t in content_stems if t in query_stems) * 1.0

    # Attribute string values
    if attributes:
        for val in attributes.values():
            if isinstance(val, str):
                score += sum(1 for t in stemmed_tokens(val) if t in query_stems) * 1.0

    return score
