"""Knowledge graph extraction for papers and documentation.

This module is deterministic and model-free. It extracts evidence that is
explicitly present in text: sections, concepts, claims, citations, and links.
LLM enrichment can be layered on top later, but the base graph should remain
repeatable and offline.
"""

from __future__ import annotations

import re
import hashlib
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

from ..types import GraphDict, make_edge, make_node

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_URL_RE = re.compile(r"https?://[^\s)>\]]+")
_BRACKET_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_AUTHOR_YEAR_RE = re.compile(r"\(([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?,\s*(?:19|20)\d{2})\)")
_REFERENCE_ENTRY_RE = re.compile(r"^\s*\[(\d+)\]\s+(.+)$")
_PHRASE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*){1,4}\b")

_STOP_PHRASES = {
    "this paper",
    "this document",
    "the paper",
    "the document",
    "our approach",
    "we propose",
    "we show",
    "we present",
}

_CLAIM_CUES = (
    "we propose",
    "we present",
    "we introduce",
    "we show",
    "we find",
    "we demonstrate",
    "this paper proposes",
    "this document describes",
    "the contribution",
)

_EVIDENCE_CUES = (
    "results show",
    "evaluation",
    "benchmark",
    "experiment",
    "achieves",
    "improves",
    "reduces",
    "accuracy",
    "recall",
    "precision",
    "%",
)


def extract_knowledge_graph(
    text: str,
    source: str = "",
    max_concepts: int = 40,
    max_claims: int = 30,
) -> GraphDict:
    """
    Extract a document knowledge graph from paper or documentation text.

    Nodes include the document, sections, concepts, claims, evidence snippets,
    citations, and URLs. Edges preserve provenance, for example section
    ``contains`` claim, claim ``supported_by`` evidence, section ``mentions``
    concept, and section ``cites`` citation.
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seen_nodes = set()
    seen_edges = set()

    doc_id = _node_id("document", source or "document")
    _add_node(nodes, seen_nodes, doc_id, source or "Document", text[:5000], {
        "type": "document",
        "source": source,
        "extraction_method": "static",
    })

    sections = _split_sections(text)
    reference_ids: Dict[str, str] = {}
    for reference in _extract_reference_entries(sections):
        reference_id = _source_node_id("reference", f"{reference['key']}-{reference['content']}", source)
        reference_ids.setdefault(reference["key"], reference_id)
        _add_node(nodes, seen_nodes, reference_id, reference["label"], reference["content"], {
            "type": "reference",
            "key": reference["key"],
            "source": source,
            "document_id": doc_id,
            "section": reference["section"],
            "section_index": reference["section_index"],
            "extraction_method": "static",
        })
        _add_edge(edges, seen_edges, doc_id, reference_id, "contains")

    for index, section in enumerate(sections):
        section_id = _source_node_id("section", f"{index}-{section['title']}", source)
        attrs = {
            "type": "section",
            "level": section["level"],
            "index": index,
            "source": source,
            "document_id": doc_id,
            "extraction_method": "static",
        }
        _add_node(nodes, seen_nodes, section_id, section["title"], section["content"], attrs)
        _add_edge(edges, seen_edges, doc_id, section_id, "contains")

        for url in _extract_urls(section["content"]):
            url_id = _source_node_id("url", url, source)
            _add_node(nodes, seen_nodes, url_id, url, attributes={
                "type": "url",
                "source": source,
                "document_id": doc_id,
                "extraction_method": "static",
            })
            _add_edge(edges, seen_edges, section_id, url_id, "links_to")

        _add_citation_edges(
            nodes,
            edges,
            seen_nodes,
            seen_edges,
            section_id,
            section["content"],
            source,
            doc_id,
            reference_ids,
        )

    concept_counts = _extract_concepts(text)
    for concept, count in concept_counts[:max_concepts]:
        concept_id = _node_id("concept", concept)
        _add_node(nodes, seen_nodes, concept_id, concept, attributes={
            "type": "concept",
            "frequency": count,
            "source": source,
            "extraction_method": "static",
        })
        for index, section in enumerate(sections):
            if concept.lower() in section["content"].lower():
                section_id = _source_node_id("section", f"{index}-{section['title']}", source)
                _add_edge(edges, seen_edges, section_id, concept_id, "mentions")

    claim_count = 0
    evidence_nodes: List[Tuple[str, str, str]] = []
    for index, section in enumerate(sections):
        section_id = _source_node_id("section", f"{index}-{section['title']}", source)
        for sentence in _sentences(section["content"]):
            lower = sentence.lower()
            if claim_count < max_claims and any(cue in lower for cue in _CLAIM_CUES):
                claim_id = _source_node_id("claim", sentence, source)
                _add_node(nodes, seen_nodes, claim_id, _label(sentence), sentence, {
                    "type": "claim",
                    "source": source,
                    "document_id": doc_id,
                    "section": section["title"],
                    "section_index": index,
                    "extraction_method": "static",
                })
                _add_edge(edges, seen_edges, section_id, claim_id, "contains")
                _add_citation_edges(
                    nodes,
                    edges,
                    seen_nodes,
                    seen_edges,
                    claim_id,
                    sentence,
                    source,
                    doc_id,
                    reference_ids,
                )
                claim_count += 1
            if any(cue in lower for cue in _EVIDENCE_CUES):
                evidence_id = _source_node_id("evidence", sentence, source)
                evidence_nodes.append((evidence_id, section_id, sentence))
                _add_node(nodes, seen_nodes, evidence_id, _label(sentence), sentence, {
                    "type": "evidence",
                    "source": source,
                    "document_id": doc_id,
                    "section": section["title"],
                    "section_index": index,
                    "extraction_method": "static",
                })
                _add_edge(edges, seen_edges, section_id, evidence_id, "contains")
                _add_citation_edges(
                    nodes,
                    edges,
                    seen_nodes,
                    seen_edges,
                    evidence_id,
                    sentence,
                    source,
                    doc_id,
                    reference_ids,
                )

    claim_ids = [n["id"] for n in nodes if n.get("attributes", {}).get("type") == "claim"]
    for evidence_id, _, _ in evidence_nodes:
        for claim_id in claim_ids[:5]:
            _add_edge(edges, seen_edges, claim_id, evidence_id, "supported_by")

    return {"nodes": nodes, "edges": edges, "current_node_id": doc_id}


def _split_sections(text: str) -> List[Dict[str, Any]]:
    headings = list(_HEADING_RE.finditer(text))
    if not headings:
        return [{"title": "Document", "level": 1, "content": text.strip()}]

    sections: List[Dict[str, Any]] = []
    preface = text[: headings[0].start()].strip()
    if preface:
        sections.append({"title": "Preface", "level": 1, "content": preface})

    for index, match in enumerate(headings):
        title = match.group(2).strip()
        start = match.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        sections.append({
            "title": title,
            "level": len(match.group(1)),
            "content": text[start:end].strip(),
        })
    return sections


def _extract_concepts(text: str) -> List[Tuple[str, int]]:
    counter: Counter[str] = Counter()
    for phrase in _PHRASE_RE.findall(text):
        normalized = " ".join(phrase.split()).strip(" .,:;()[]").lower()
        words = normalized.split()
        if len(words) < 2 or normalized in _STOP_PHRASES:
            continue
        if words[0] in {"this", "that", "these", "those", "there", "where", "when"}:
            continue
        if len(normalized) < 8:
            continue
        counter[normalized] += 1
    return counter.most_common()


def _extract_urls(text: str) -> Iterable[str]:
    return [u.rstrip(".,") for u in _URL_RE.findall(text)]


def _extract_citations(text: str) -> Iterable[str]:
    citations = []
    for match in _BRACKET_CITATION_RE.finditer(text):
        citations.extend(part.strip() for part in match.group(1).split(","))
    citations.extend(match.group(1) for match in _AUTHOR_YEAR_RE.finditer(text))
    return citations


def _extract_reference_entries(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    references: List[Dict[str, Any]] = []
    for index, section in enumerate(sections):
        if section["title"].strip().lower() not in {"references", "bibliography", "works cited"}:
            continue
        current: Dict[str, Any] | None = None
        for raw_line in section["content"].splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = _REFERENCE_ENTRY_RE.match(line)
            if match:
                if current is not None:
                    references.append(current)
                key = match.group(1).strip()
                content = match.group(2).strip()
                current = {
                    "key": key,
                    "label": f"[{key}] {_label(content, limit=80)}",
                    "content": content,
                    "section": section["title"],
                    "section_index": index,
                }
            elif current is not None:
                current["content"] = f"{current['content']} {line}".strip()
                current["label"] = f"[{current['key']}] {_label(current['content'], limit=80)}"
        if current is not None:
            references.append(current)
    return references


def _add_citation_edges(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    seen_nodes: set,
    seen_edges: set,
    owner_id: str,
    text: str,
    source: str,
    document_id: str,
    reference_ids: Dict[str, str],
) -> None:
    for citation in _extract_citations(text):
        citation_id = _source_node_id("citation", citation, source)
        _add_node(nodes, seen_nodes, citation_id, citation, attributes={
            "type": "citation",
            "source": source,
            "document_id": document_id,
            "extraction_method": "static",
        })
        _add_edge(edges, seen_edges, owner_id, citation_id, "cites")
        reference_id = reference_ids.get(citation)
        if reference_id:
            _add_edge(edges, seen_edges, citation_id, reference_id, "resolves_to")


def _sentences(text: str) -> Iterable[str]:
    for sentence in _SENTENCE_RE.split(re.sub(r"\s+", " ", text).strip()):
        sentence = sentence.strip()
        if len(sentence) >= 30:
            yield sentence


def _label(text: str, limit: int = 96) -> str:
    return text.strip().replace("\n", " ")[:limit].rstrip()


def _node_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:10]
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")[:70]
    return f"{prefix}:{slug or 'item'}:{digest}"


def _source_node_id(prefix: str, value: str, source: str) -> str:
    scoped_value = f"{source}\0{value}" if source else value
    return _node_id(prefix, scoped_value)


def _add_node(
    nodes: List[Dict[str, Any]],
    seen: set,
    node_id: str,
    label: str,
    content: str | None = None,
    attributes: Dict[str, Any] | None = None,
) -> None:
    if node_id in seen:
        return
    nodes.append(make_node(node_id, label, content=content, attributes=attributes))
    seen.add(node_id)


def _add_edge(edges: List[Dict[str, Any]], seen: set, from_id: str, to_id: str, label: str) -> None:
    key = (from_id, to_id, label)
    if key in seen:
        return
    edges.append(make_edge(from_id, to_id, label))
    seen.add(key)
