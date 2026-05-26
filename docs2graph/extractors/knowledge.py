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
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ..types import GraphDict, make_edge, make_node

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_URL_RE = re.compile(r"https?://[^\s)>\]]+")
_BRACKET_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_AUTHOR_YEAR_RE = re.compile(r"\(([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?,\s*(?:19|20)\d{2})\)")
_REFERENCE_ENTRY_RE = re.compile(r"^\s*\[(\d+)\]\s+(.+)$")
_REFERENCE_AUTHOR_YEAR_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\[\d+\]\s*)?"
    r"([A-Z][A-Za-z\-]+)(\s+et\s+al\.)?"
    r"(?:,\s+[A-Z](?:\.[A-Z]\.)?\.?)?"
    r".*?\b((?:19|20)\d{2})\b"
)
_PHRASE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*){1,4}\b")
_GLOSSARY_DEFINITION_RE = re.compile(
    r"^\s*(?:[-*]\s+)?(?:\*\*)?([A-Za-z][A-Za-z0-9 /+\-]{1,80}?)"
    r"(?:\*\*)?\s*(?::|--| - )\s+(.{12,})$"
)
_SENTENCE_DEFINITION_RE = re.compile(
    r"^\s*(?:A|An|The)?\s*([A-Z][A-Za-z0-9 /+\-]{2,80}?)\s+"
    r"(?:is|are|means|refers to|is defined as|are defined as)\s+(.{12,})$",
    re.IGNORECASE,
)

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

_SUPPORT_STOPWORDS = {
    "about",
    "after",
    "also",
    "because",
    "before",
    "being",
    "between",
    "could",
    "document",
    "during",
    "from",
    "have",
    "into",
    "more",
    "paper",
    "show",
    "shows",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "using",
    "when",
    "where",
    "which",
    "with",
    "would",
}


def extract_knowledge_graph(
    text: str,
    source: str = "",
    max_concepts: int = 40,
    max_claims: int = 30,
    max_definitions: int = 30,
    max_tables: int = 20,
) -> GraphDict:
    """
    Extract a document knowledge graph from paper or documentation text.

    Nodes include the document, sections, concepts, definitions, claims,
    evidence snippets, citations, and URLs. Edges preserve provenance, for
    example section ``contains`` claim, definition ``defines`` concept, claim
    ``supported_by`` evidence, section ``mentions`` concept, and section
    ``cites`` citation.
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
        for key in reference["keys"]:
            reference_ids.setdefault(key, reference_id)
        _add_node(nodes, seen_nodes, reference_id, reference["label"], reference["content"], {
            "type": "reference",
            "key": reference["key"],
            "aliases": reference["keys"],
            "source": source,
            "document_id": doc_id,
            "section": reference["section"],
            "section_index": reference["section_index"],
            "extraction_method": "static",
        })
        _add_edge(edges, seen_edges, doc_id, reference_id, "contains")

    table_records: List[Dict[str, Any]] = []
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

        if len(table_records) < max_tables:
            for table_index, table in enumerate(_extract_markdown_tables(section["content"])):
                if len(table_records) >= max_tables:
                    break
                table_id = _source_node_id(
                    "table",
                    f"{index}-{table_index}-{table['content']}",
                    source,
                )
                _add_node(nodes, seen_nodes, table_id, f"Table: {section['title']}", table["content"], {
                    "type": "table",
                    "source": source,
                    "document_id": doc_id,
                    "section": section["title"],
                    "section_index": index,
                    "table_index": table_index,
                    "headers": table["headers"],
                    "row_count": len(table["rows"]),
                    "extraction_method": "static",
                })
                _add_edge(edges, seen_edges, section_id, table_id, "contains")
                _add_citation_edges(
                    nodes,
                    edges,
                    seen_nodes,
                    seen_edges,
                    table_id,
                    table["content"],
                    source,
                    doc_id,
                    reference_ids,
                )
                table_records.append({
                    "id": table_id,
                    "section_index": index,
                    "section_title": section["title"],
                    "table_index": table_index,
                    "content": table["content"],
                    "rows": table["rows"],
                })

    concept_counts = _extract_concepts(text)
    materialized_concepts: set[str] = set()
    for concept, count in concept_counts[:max_concepts]:
        concept_id = _node_id("concept", concept)
        _add_node(nodes, seen_nodes, concept_id, concept, attributes={
            "type": "concept",
            "frequency": count,
            "source": source,
            "extraction_method": "static",
        })
        materialized_concepts.add(concept)
        for index, section in enumerate(sections):
            if concept.lower() in section["content"].lower():
                section_id = _source_node_id("section", f"{index}-{section['title']}", source)
                _add_edge(edges, seen_edges, section_id, concept_id, "mentions")
        for table in table_records:
            if concept.lower() in table["content"].lower():
                _add_edge(edges, seen_edges, table["id"], concept_id, "mentions")

    definition_count = 0
    for index, section in enumerate(sections):
        if definition_count >= max_definitions:
            break
        section_id = _source_node_id("section", f"{index}-{section['title']}", source)
        for definition in _extract_definitions(section["content"]):
            if definition_count >= max_definitions:
                break
            term = definition["term"]
            definition_text = definition["definition"]
            normalized_term = _normalize_concept(term)
            if not normalized_term:
                continue
            concept_id = _node_id("concept", normalized_term)
            if normalized_term not in materialized_concepts:
                _add_node(nodes, seen_nodes, concept_id, normalized_term, attributes={
                    "type": "concept",
                    "frequency": _count_phrase_occurrences(text, normalized_term),
                    "source": source,
                    "extraction_method": "static",
                })
                materialized_concepts.add(normalized_term)
            definition_id = _source_node_id(
                "definition",
                f"{normalized_term}-{definition_text}",
                source,
            )
            _add_node(nodes, seen_nodes, definition_id, _label(f"{term}: {definition_text}"), definition_text, {
                "type": "definition",
                "term": term,
                "normalized_term": normalized_term,
                "source": source,
                "document_id": doc_id,
                "section": section["title"],
                "section_index": index,
                "extraction_method": "static",
            })
            _add_edge(edges, seen_edges, section_id, definition_id, "contains")
            _add_edge(edges, seen_edges, definition_id, concept_id, "defines")
            _add_edge(edges, seen_edges, concept_id, definition_id, "defined_by")
            _add_edge(edges, seen_edges, section_id, concept_id, "mentions")
            _add_citation_edges(
                nodes,
                edges,
                seen_nodes,
                seen_edges,
                definition_id,
                definition_text,
                source,
                doc_id,
                reference_ids,
            )
            definition_count += 1

    claim_count = 0
    claim_records: List[Dict[str, Any]] = []
    evidence_records: List[Dict[str, Any]] = []
    for index, section in enumerate(sections):
        section_id = _source_node_id("section", f"{index}-{section['title']}", source)
        for sentence_index, sentence in enumerate(_sentences(section["content"])):
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
                claim_records.append({
                    "id": claim_id,
                    "section_index": index,
                    "sentence_index": sentence_index,
                    "text": sentence,
                })
                claim_count += 1
            if any(cue in lower for cue in _EVIDENCE_CUES):
                evidence_id = _source_node_id("evidence", sentence, source)
                evidence_records.append({
                    "id": evidence_id,
                    "section_index": index,
                    "sentence_index": sentence_index,
                    "text": sentence,
                })
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

    for table in table_records:
        table_id = table["id"]
        for row_index, row in enumerate(table["rows"]):
            row_text = " | ".join(
                f"{key}: {value}" for key, value in row.items() if value
            )
            if not _looks_like_table_evidence(row_text):
                continue
            evidence_id = _source_node_id(
                "evidence",
                f"table-{table['section_index']}-{table['table_index']}-{row_index}-{row_text}",
                source,
            )
            evidence_records.append({
                "id": evidence_id,
                "section_index": table["section_index"],
                "sentence_index": 10_000 + row_index,
                "text": row_text,
            })
            _add_node(nodes, seen_nodes, evidence_id, _label(row_text), row_text, {
                "type": "evidence",
                "source": source,
                "document_id": doc_id,
                "section": table["section_title"],
                "section_index": table["section_index"],
                "table_index": table["table_index"],
                "row_index": row_index,
                "evidence_kind": "table_row",
                "extraction_method": "static",
            })
            _add_edge(edges, seen_edges, table_id, evidence_id, "contains")
            _add_citation_edges(
                nodes,
                edges,
                seen_nodes,
                seen_edges,
                evidence_id,
                row_text,
                source,
                doc_id,
                reference_ids,
            )

    _add_support_edges(edges, seen_edges, claim_records, evidence_records)

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
        normalized = _normalize_concept(phrase)
        if not normalized:
            continue
        counter[normalized] += 1
    return counter.most_common()


def _extract_definitions(text: str) -> List[Dict[str, str]]:
    definitions: List[Dict[str, str]] = []
    seen_terms: set[str] = set()

    for line in text.splitlines():
        match = _GLOSSARY_DEFINITION_RE.match(line.strip())
        if not match:
            continue
        term = match.group(1).strip(" *`")
        definition = match.group(2).strip()
        normalized = _normalize_concept(term)
        if not normalized or normalized in seen_terms or _looks_like_heading_noise(term):
            continue
        definitions.append({"term": term, "definition": definition})
        seen_terms.add(normalized)

    for sentence in _sentences(text):
        match = _SENTENCE_DEFINITION_RE.match(sentence)
        if not match:
            continue
        term = match.group(1).strip()
        definition = match.group(2).strip()
        normalized = _normalize_concept(term)
        if not normalized or normalized in seen_terms or _looks_like_heading_noise(term):
            continue
        definitions.append({"term": term, "definition": definition})
        seen_terms.add(normalized)

    return definitions


def _extract_markdown_tables(text: str) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines) - 1:
        header_line = lines[index].strip()
        separator_line = lines[index + 1].strip()
        if not (_is_table_row(header_line) and _is_table_separator(separator_line)):
            index += 1
            continue

        headers = [_clean_table_cell(cell) for cell in _split_table_row(header_line)]
        rows: List[Dict[str, str]] = []
        table_lines = [header_line, separator_line]
        index += 2
        while index < len(lines) and _is_table_row(lines[index].strip()):
            row_line = lines[index].strip()
            cells = [_clean_table_cell(cell) for cell in _split_table_row(row_line)]
            if any(cells):
                rows.append({
                    header or f"column_{cell_index + 1}": cells[cell_index]
                    if cell_index < len(cells)
                    else ""
                    for cell_index, header in enumerate(headers)
                })
                table_lines.append(row_line)
            index += 1

        if headers and rows:
            tables.append({
                "headers": headers,
                "rows": rows,
                "content": "\n".join(table_lines),
            })
        continue
    return tables


def _is_table_row(line: str) -> bool:
    return line.startswith("|") and line.endswith("|") and line.count("|") >= 2


def _is_table_separator(line: str) -> bool:
    if not _is_table_row(line):
        return False
    cells = _split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _split_table_row(line: str) -> List[str]:
    return line.strip().strip("|").split("|")


def _clean_table_cell(cell: str) -> str:
    return re.sub(r"<br\s*/?>", " ", cell, flags=re.IGNORECASE).strip(" `")


def _looks_like_table_evidence(row_text: str) -> bool:
    lower = row_text.lower()
    return bool(re.search(r"\d|%", row_text)) or any(cue in lower for cue in _EVIDENCE_CUES)


def _normalize_concept(value: str) -> str:
    normalized = " ".join(value.split()).strip(" .,:;()[]`*_").lower()
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    words = normalized.split()
    if len(words) < 2 or normalized in _STOP_PHRASES:
        return ""
    if words[0] in {"this", "that", "these", "those", "there", "where", "when"}:
        return ""
    if len(normalized) < 8:
        return ""
    return normalized


def _looks_like_heading_noise(term: str) -> bool:
    lower = term.strip().lower()
    return lower in {"note", "example", "warning", "tip", "todo", "references"} or bool(
        re.match(r"^option\s+[a-z0-9]+$", lower)
    )


def _count_phrase_occurrences(text: str, phrase: str) -> int:
    pattern = re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", re.IGNORECASE)
    return len(pattern.findall(text))


def _extract_urls(text: str) -> Iterable[str]:
    return [u.rstrip(".,") for u in _URL_RE.findall(text)]


def _extract_citations(text: str) -> Iterable[str]:
    citations = []
    for match in _BRACKET_CITATION_RE.finditer(text):
        citations.extend(part.strip() for part in match.group(1).split(","))
    citations.extend(match.group(1) for match in _AUTHOR_YEAR_RE.finditer(text))
    return citations


def _add_support_edges(
    edges: List[Dict[str, Any]],
    seen_edges: set,
    claim_records: Sequence[Dict[str, Any]],
    evidence_records: Sequence[Dict[str, Any]],
    *,
    max_evidence_per_claim: int = 5,
) -> None:
    """
    Link claims only to evidence with local or lexical support.

    Earlier extraction linked every evidence snippet to the first few claims,
    which made large documents look more connected than their text warranted.
    This keeps support deterministic while preferring evidence in the same
    section, then evidence with shared meaningful terms.
    """
    if not claim_records or not evidence_records:
        return

    evidence_tokens = {
        evidence["id"]: _support_tokens(evidence.get("text", ""))
        for evidence in evidence_records
    }
    for claim in claim_records:
        claim_tokens = _support_tokens(claim.get("text", ""))
        ranked: List[Tuple[int, int, int, str]] = []
        for evidence in evidence_records:
            lexical_overlap = len(claim_tokens & evidence_tokens[evidence["id"]])
            same_section = claim["section_index"] == evidence["section_index"]
            if not same_section and lexical_overlap == 0:
                continue
            section_distance = abs(claim["section_index"] - evidence["section_index"])
            sentence_distance = abs(claim["sentence_index"] - evidence["sentence_index"])
            ranked.append((
                0 if same_section else 1,
                -lexical_overlap,
                section_distance * 1000 + sentence_distance,
                evidence["id"],
            ))

        for _, _, _, evidence_id in sorted(ranked)[:max_evidence_per_claim]:
            _add_edge(edges, seen_edges, claim["id"], evidence_id, "supported_by")


def _support_tokens(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", text.lower()):
        if token not in _SUPPORT_STOPWORDS:
            tokens.add(token)
    return tokens


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
                    references.append(_finalize_reference(current))
                key = match.group(1).strip()
                content = match.group(2).strip()
                current = {
                    "key": key,
                    "label": f"[{key}] {_label(content, limit=80)}",
                    "content": content,
                    "section": section["title"],
                    "section_index": index,
                }
            elif current is None:
                key = _infer_author_year_key(line)
                if key:
                    references.append(_finalize_reference({
                        "key": key,
                        "label": f"[{key}] {_label(line, limit=80)}",
                        "content": line,
                        "section": section["title"],
                        "section_index": index,
                    }))
            elif current is not None:
                current["content"] = f"{current['content']} {line}".strip()
                current["label"] = f"[{current['key']}] {_label(current['content'], limit=80)}"
        if current is not None:
            references.append(_finalize_reference(current))
    return references


def _finalize_reference(reference: Dict[str, Any]) -> Dict[str, Any]:
    keys = [reference["key"]]
    for inferred in _infer_author_year_keys(reference["content"]):
        if inferred and inferred not in keys:
            keys.append(inferred)
    reference["keys"] = keys
    return reference


def _infer_author_year_key(text: str) -> str:
    keys = _infer_author_year_keys(text)
    return keys[0] if keys else ""


def _infer_author_year_keys(text: str) -> List[str]:
    match = _REFERENCE_AUTHOR_YEAR_RE.match(text.strip())
    if not match:
        return []
    surname = match.group(1)
    et_al = bool(match.group(2))
    year = match.group(3)
    keys = [f"{surname}, {year}"]
    if et_al:
        keys.append(f"{surname} et al., {year}")
    return keys


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
