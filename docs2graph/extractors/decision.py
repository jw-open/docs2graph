"""Decision graph extraction for architecture and design documents."""

from __future__ import annotations

import re
import hashlib
from typing import Any, Dict, List, Optional

from ..types import GraphDict, make_edge, make_node

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+)$", re.MULTILINE)
_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+)$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_PREFIXED_LINE_RE = re.compile(
    r"^\s*"
    r"("
    r"problem|challenge|issue|motivation|"
    r"context|background|goals?|requirements?|constraints?|assumptions?|"
    r"drivers?|decision drivers?|rationale|reasons?|non[- ]goals?|"
    r"options?(?:\s+[a-z0-9]+)?|alternatives?(?:\s+[a-z0-9]+)?|"
    r"approaches?(?:\s+[a-z0-9]+)?|solutions?(?:\s+[a-z0-9]+)?|"
    r"pros?|benefits?|advantages?|cons?|drawbacks?|cost|risks?|"
    r"trade-?offs?|considerations?|"
    r"decision|choose|chosen|selected|status|"
    r"consequences?|impact|outcomes?|results?|confidence|certainty|confidence level"
    r")"
    r"\s*:\s*(\S.*)$",
    re.IGNORECASE,
)

_SECTION_TYPES = {
    "problem": ("problem", "challenge", "motivation", "issue"),
    "context": (
        "context",
        "background",
        "goal",
        "goals",
        "requirement",
        "requirements",
        "constraint",
        "constraints",
        "assumption",
        "assumptions",
        "driver",
        "drivers",
        "decision driver",
        "decision drivers",
        "rationale",
        "reason",
        "reasons",
        "non goal",
        "non goals",
    ),
    "option": ("option", "alternative", "approach", "solution"),
    "pros": ("pros", "benefits", "advantages"),
    "cons": ("cons", "drawbacks", "disadvantages"),
    "tradeoff": ("tradeoff", "trade-off", "consideration"),
    "decision": (
        "decision",
        "status",
        "accepted",
        "proposed",
        "rejected",
        "deprecated",
        "superseded",
        "deferred",
        "selected",
        "chosen",
        "resolution",
    ),
    "consequence": ("consequence", "consequences", "impact", "outcome", "outcomes", "result", "risks"),
    "confidence": ("confidence", "certainty", "confidence level"),
}

_DECISION_STATUS_VALUES = (
    "accepted",
    "proposed",
    "rejected",
    "deprecated",
    "superseded",
    "deferred",
)


def extract_decision_graph(text: str, source: str = "") -> GraphDict:
    """
    Extract documented architecture/design decisions as a graph.

    Static extraction finds explicit decision evidence in ADR/RFC/design docs.
    Nodes are labelled by role: problem, context, option, pros, cons,
    tradeoff, decision, and consequence. LLM-inferred reasoning should be
    added later as separate nodes with ``extraction_method=llm_inferred``.
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seen_nodes = set()
    seen_edges = set()

    root_id = _node_id("decision_document", source or "document")
    _add_node(nodes, seen_nodes, root_id, source or "Decision Document", text[:4000], {
        "type": "decision_document",
        "source": source,
        "extraction_method": "static",
    })

    sections = _split_sections(text)
    last_problem_id = None
    current_option_id = None
    selected_option_id = None
    current_decision_id = None
    context_ids: List[str] = []
    option_records: List[Dict[str, str]] = []

    for index, section in enumerate(sections):
        kind = _classify(section["title"], section["content"])
        section_id = _source_node_id(kind, f"{index}-{section['title']}", source)
        attrs = {
            "type": kind,
            "level": section["level"],
            "source": source,
            "document_id": root_id,
            "extraction_method": "static",
        }
        if kind == "decision":
            status = _decision_status(section["title"], section["content"])
            if status:
                attrs["decision_status"] = status
        _add_node(nodes, seen_nodes, section_id, section["title"], section["content"], attrs)
        _add_edge(edges, seen_edges, root_id, section_id, "contains")

        if kind == "problem":
            last_problem_id = section_id
        elif kind == "context":
            context_ids.append(section_id)
            if last_problem_id:
                _add_edge(edges, seen_edges, last_problem_id, section_id, "has_context")
            if current_decision_id:
                _add_edge(edges, seen_edges, current_decision_id, section_id, "informed_by")
        elif kind == "option":
            current_option_id = section_id
            option_records.append({"id": section_id, "title": section["title"], "content": section["content"]})
            if last_problem_id:
                _add_edge(edges, seen_edges, last_problem_id, section_id, "has_option")
        elif kind in {"pros", "cons", "tradeoff"} and current_option_id:
            _add_edge(edges, seen_edges, current_option_id, section_id, kind)
        elif kind == "decision":
            current_decision_id = section_id
            if last_problem_id:
                _add_edge(edges, seen_edges, last_problem_id, section_id, "resolved_by")
            selected_option_id = _select_option_id(section["content"], option_records) or current_option_id
            if selected_option_id:
                _add_edge(edges, seen_edges, section_id, selected_option_id, "selects")
            for context_id in context_ids:
                _add_edge(edges, seen_edges, section_id, context_id, "informed_by")
        elif kind == "consequence":
            if current_decision_id:
                _add_edge(edges, seen_edges, current_decision_id, section_id, "has_consequence")
            if selected_option_id:
                _add_edge(edges, seen_edges, selected_option_id, section_id, "has_consequence")
            elif current_option_id:
                _add_edge(edges, seen_edges, current_option_id, section_id, "has_consequence")
        elif kind == "confidence" and current_decision_id:
            _add_edge(edges, seen_edges, current_decision_id, section_id, "has_confidence")

        for table_index, table in enumerate(_extract_markdown_tables(section["content"])):
            result = _add_decision_table(
                nodes,
                edges,
                seen_nodes,
                seen_edges,
                table,
                source=source,
                root_id=root_id,
                section_id=section_id,
                section_title=section["title"],
                section_index=index,
                table_index=table_index,
                last_problem_id=last_problem_id,
                current_option_id=current_option_id,
                current_decision_id=current_decision_id,
                option_records=option_records,
            )
            current_option_id = result.get("current_option_id") or current_option_id
            selected_option_id = result.get("selected_option_id") or selected_option_id
            current_decision_id = result.get("current_decision_id") or current_decision_id

        for line_index, item in _extract_decision_items(section["content"]):
            state = _add_decision_item(
                nodes,
                edges,
                seen_nodes,
                seen_edges,
                text=item,
                default_kind=kind,
                item_key=f"{index}-line-{line_index}-{item}",
                source=source,
                root_id=root_id,
                section_id=section_id,
                section_title=section["title"],
                last_problem_id=last_problem_id,
                current_option_id=current_option_id,
                selected_option_id=selected_option_id,
                current_decision_id=current_decision_id,
                context_ids=context_ids,
                option_records=option_records,
            )
            last_problem_id = state["last_problem_id"]
            current_option_id = state["current_option_id"]
            selected_option_id = state["selected_option_id"]
            current_decision_id = state["current_decision_id"]

    return {"nodes": nodes, "edges": edges, "current_node_id": root_id}


def _split_sections(text: str) -> List[Dict[str, Any]]:
    headings = list(_HEADING_RE.finditer(text))
    if not headings:
        return [{"title": "Decision", "level": 1, "content": text.strip()}]

    sections: List[Dict[str, Any]] = []
    for index, match in enumerate(headings):
        start = match.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        sections.append({
            "title": match.group(2).strip(),
            "level": len(match.group(1)),
            "content": text[start:end].strip(),
        })
    return sections


def _classify(title: str, content: str) -> str:
    title_lower = title.lower()
    for kind, cues in _SECTION_TYPES.items():
        if any(_contains_cue(title_lower, cue) for cue in cues):
            return kind

    text = f"{title}\n{content[:300]}".lower()
    for kind, cues in _SECTION_TYPES.items():
        if any(_contains_cue(text, cue) for cue in cues):
            return kind
    return "decision_context"


def _contains_cue(text: str, cue: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(cue) + r"(?![a-z0-9])", text) is not None


def _classify_bullet(text: str, default: str) -> str:
    lower = text.lower()
    if lower.startswith(("problem:", "challenge:", "issue:", "motivation:")):
        return "problem"
    if lower.startswith((
        "option:",
        "options:",
        "alternative:",
        "alternatives:",
        "approach:",
        "approaches:",
        "solution:",
        "solutions:",
    )):
        return "option"
    if re.match(r"^(?:option|alternative|approach|solution)\s+[a-z0-9]+[:.)-]", lower):
        return "option"
    if lower.startswith(("pro:", "pros:", "benefit:", "benefits:", "advantage:", "advantages:")):
        return "pros"
    if lower.startswith(("risk:", "risk -", "risk.")):
        return "consequence" if default == "consequence" else "cons"
    if lower.startswith(("con:", "cons:", "drawback:", "drawbacks:", "cost:")):
        return "cons"
    if "tradeoff" in lower or "trade-off" in lower or "but " in lower:
        return "tradeoff"
    if lower.startswith(("decide", "decision:", "choose", "chosen", "selected:", "status:")):
        return "decision"
    if lower.startswith((
        "context:",
        "background:",
        "goal:",
        "goals:",
        "requirement:",
        "requirements:",
        "constraint:",
        "constraints:",
        "assumption:",
        "assumptions:",
        "driver:",
        "drivers:",
        "decision driver:",
        "decision drivers:",
        "rationale:",
        "reason:",
        "reasons:",
        "non-goal:",
        "non-goals:",
        "non goal:",
        "non goals:",
    )):
        return "context"
    if _decision_status("", text):
        return "decision"
    if lower.startswith(("consequence:", "consequences:", "impact:", "outcome:", "outcomes:", "result:", "results:", "risk:")):
        return "consequence"
    if lower.startswith(("confidence:", "certainty:", "confidence level:")):
        return "confidence"
    return default


def _extract_decision_items(text: str) -> List[tuple[int, str]]:
    items: List[tuple[int, str]] = []
    for line_index, raw_line in enumerate(text.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("|", "#")):
            continue
        bullet_match = _BULLET_LINE_RE.match(line)
        if bullet_match:
            items.append((line_index, bullet_match.group(1).strip()))
            continue
        if _PREFIXED_LINE_RE.match(line):
            items.append((line_index, line))
    return items


def _add_decision_item(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    seen_nodes: set,
    seen_edges: set,
    *,
    text: str,
    default_kind: str,
    item_key: str,
    source: str,
    root_id: str,
    section_id: str,
    section_title: str,
    last_problem_id: Optional[str],
    current_option_id: Optional[str],
    selected_option_id: Optional[str],
    current_decision_id: Optional[str],
    context_ids: List[str],
    option_records: List[Dict[str, str]],
) -> Dict[str, Optional[str]]:
    item_kind = _classify_bullet(text, default=default_kind)
    item_id = _source_node_id(item_kind, item_key, source)
    attrs = {
        "type": item_kind,
        "source": source,
        "document_id": root_id,
        "section": section_title,
        "extraction_method": "static",
    }
    if item_kind == "decision":
        status = _decision_status("", text)
        if status:
            attrs["decision_status"] = status
    _add_node(nodes, seen_nodes, item_id, _label(text), text, attrs)
    _add_edge(edges, seen_edges, section_id, item_id, "contains")

    if item_kind == "problem":
        last_problem_id = item_id
    elif item_kind == "context":
        context_ids.append(item_id)
        if last_problem_id:
            _add_edge(edges, seen_edges, last_problem_id, item_id, "has_context")
        if current_decision_id:
            _add_edge(edges, seen_edges, current_decision_id, item_id, "informed_by")
    elif item_kind == "option":
        current_option_id = item_id
        option_records.append({"id": item_id, "title": text, "content": text})
        if last_problem_id:
            _add_edge(edges, seen_edges, last_problem_id, item_id, "has_option")
    elif item_kind in {"pros", "cons", "tradeoff"} and current_option_id:
        _add_edge(edges, seen_edges, current_option_id, item_id, item_kind)
    elif item_kind == "decision":
        current_decision_id = item_id
        if last_problem_id:
            _add_edge(edges, seen_edges, last_problem_id, item_id, "resolved_by")
        selected_option_id = _select_option_id(text, option_records) or current_option_id
        if selected_option_id:
            _add_edge(edges, seen_edges, item_id, selected_option_id, "selects")
        for context_id in context_ids:
            _add_edge(edges, seen_edges, item_id, context_id, "informed_by")
    elif item_kind == "consequence":
        if current_decision_id:
            _add_edge(edges, seen_edges, current_decision_id, item_id, "has_consequence")
        if selected_option_id:
            _add_edge(edges, seen_edges, selected_option_id, item_id, "has_consequence")
        elif current_option_id:
            _add_edge(edges, seen_edges, current_option_id, item_id, "has_consequence")
    elif item_kind == "confidence" and current_decision_id:
        _add_edge(edges, seen_edges, current_decision_id, item_id, "has_confidence")

    return {
        "last_problem_id": last_problem_id,
        "current_option_id": current_option_id,
        "selected_option_id": selected_option_id,
        "current_decision_id": current_decision_id,
    }


def _decision_status(title: str, content: str) -> Optional[str]:
    text = f"{title}\n{content}".lower()
    for status in _DECISION_STATUS_VALUES:
        if re.search(r"(?<![a-z0-9])" + re.escape(status) + r"(?![a-z0-9])", text):
            return status
    return None


def _extract_markdown_tables(text: str) -> List[List[Dict[str, str]]]:
    tables: List[List[Dict[str, str]]] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines) - 1:
        header_line = lines[index]
        separator_line = lines[index + 1]
        if "|" not in header_line or not _TABLE_SEPARATOR_RE.match(separator_line):
            index += 1
            continue

        headers = [_normalize_header(cell) for cell in _split_table_row(header_line)]
        rows: List[Dict[str, str]] = []
        index += 2
        while index < len(lines) and "|" in lines[index]:
            cells = _split_table_row(lines[index])
            if len(cells) < 2:
                break
            row = {
                headers[cell_index]: cells[cell_index].strip()
                for cell_index in range(min(len(headers), len(cells)))
                if headers[cell_index]
            }
            if any(row.values()):
                rows.append(row)
            index += 1
        if rows:
            tables.append(rows)
    return tables


def _split_table_row(line: str) -> List[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _normalize_header(header: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", header.lower()).strip("_")
    aliases = {
        "alternative": "option",
        "alternatives": "option",
        "approach": "option",
        "solution": "option",
        "benefit": "pros",
        "benefits": "pros",
        "advantages": "pros",
        "pro": "pros",
        "drawback": "cons",
        "drawbacks": "cons",
        "disadvantages": "cons",
        "con": "cons",
        "risk": "cons",
        "risks": "cons",
        "trade_off": "tradeoff",
        "tradeoffs": "tradeoff",
        "considerations": "tradeoff",
        "selected": "decision",
        "chosen": "decision",
        "status": "decision",
        "resolution": "decision",
        "impact": "consequence",
        "result": "consequence",
        "results": "consequence",
        "certainty": "confidence",
        "confidence_level": "confidence",
    }
    return aliases.get(normalized, normalized)


def _add_decision_table(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    seen_nodes: set,
    seen_edges: set,
    rows: List[Dict[str, str]],
    *,
    source: str,
    root_id: str,
    section_id: str,
    section_title: str,
    section_index: int,
    table_index: int,
    last_problem_id: Optional[str],
    current_option_id: Optional[str],
    current_decision_id: Optional[str],
    option_records: List[Dict[str, str]],
) -> Dict[str, Optional[str]]:
    option_column = _first_present_column(rows, ("option", "name", "title"))
    selected_option_id: Optional[str] = None
    for row_index, row in enumerate(rows):
        option_text = row.get(option_column or "", "").strip()
        if not option_text:
            option_text = row.get("decision", "").strip()
        row_key = f"{section_index}-{table_index}-{row_index}-{option_text or _row_content(row)}"

        option_id = current_option_id
        if option_text:
            option_id = _source_node_id("option", f"table-{row_key}", source)
            _add_node(nodes, seen_nodes, option_id, _label(option_text), _row_content(row), {
                "type": "option",
                "source": source,
                "document_id": root_id,
                "section": section_title,
                "section_index": section_index,
                "table_index": table_index,
                "row_index": row_index,
                "extraction_method": "static",
            })
            _add_edge(edges, seen_edges, section_id, option_id, "contains")
            if last_problem_id:
                _add_edge(edges, seen_edges, last_problem_id, option_id, "has_option")
            option_records.append({"id": option_id, "title": option_text, "content": _row_content(row)})
            current_option_id = option_id

        for kind in ("pros", "cons", "tradeoff", "consequence"):
            value = row.get(kind, "").strip()
            if not value:
                continue
            node_id = _source_node_id(kind, f"table-{row_key}-{kind}-{value}", source)
            attrs = {
                "type": kind,
                "source": source,
                "document_id": root_id,
                "section": section_title,
                "section_index": section_index,
                "table_index": table_index,
                "row_index": row_index,
                "extraction_method": "static",
            }
            _add_node(nodes, seen_nodes, node_id, _label(value), value, attrs)
            _add_edge(edges, seen_edges, section_id, node_id, "contains")
            if option_id:
                label = "has_consequence" if kind == "consequence" else kind
                _add_edge(edges, seen_edges, option_id, node_id, label)

        decision_value = row.get("decision", "").strip()
        if decision_value and _is_positive_decision_cell(decision_value):
            decision_id = _source_node_id("decision", f"table-{row_key}-decision-{decision_value}", source)
            attrs = {
                "type": "decision",
                "source": source,
                "document_id": root_id,
                "section": section_title,
                "section_index": section_index,
                "table_index": table_index,
                "row_index": row_index,
                "extraction_method": "static",
            }
            status = _decision_status("", decision_value)
            if status:
                attrs["decision_status"] = status
            _add_node(nodes, seen_nodes, decision_id, _label(decision_value), decision_value, attrs)
            _add_edge(edges, seen_edges, section_id, decision_id, "contains")
            if last_problem_id:
                _add_edge(edges, seen_edges, last_problem_id, decision_id, "resolved_by")
            if option_id:
                _add_edge(edges, seen_edges, decision_id, option_id, "selects")
                selected_option_id = option_id
            current_decision_id = decision_id

        confidence_value = row.get("confidence", "").strip()
        if confidence_value:
            confidence_id = _source_node_id("confidence", f"table-{row_key}-confidence-{confidence_value}", source)
            _add_node(nodes, seen_nodes, confidence_id, _label(confidence_value), confidence_value, {
                "type": "confidence",
                "source": source,
                "document_id": root_id,
                "section": section_title,
                "section_index": section_index,
                "table_index": table_index,
                "row_index": row_index,
                "confidence_value": confidence_value,
                "extraction_method": "static",
            })
            _add_edge(edges, seen_edges, section_id, confidence_id, "contains")
            if current_decision_id:
                _add_edge(edges, seen_edges, current_decision_id, confidence_id, "has_confidence")
            elif option_id:
                _add_edge(edges, seen_edges, option_id, confidence_id, "confidence")

    return {
        "current_option_id": current_option_id,
        "selected_option_id": selected_option_id,
        "current_decision_id": current_decision_id,
    }


def _first_present_column(rows: List[Dict[str, str]], candidates: tuple[str, ...]) -> Optional[str]:
    for candidate in candidates:
        if any(row.get(candidate) for row in rows):
            return candidate
    return None


def _row_content(row: Dict[str, str]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in row.items() if value)


def _is_positive_decision_cell(text: str) -> bool:
    lower = text.lower().strip()
    if not lower or lower in {"no", "n", "rejected", "reject", "not selected", "deferred"}:
        return False
    return lower in {"yes", "y", "accepted", "selected", "chosen", "recommended"} or any(
        cue in lower for cue in ("choose", "chosen", "select", "accept", "recommend")
    )


def _select_option_id(text: str, option_records: List[Dict[str, str]]) -> Optional[str]:
    decision_text = _normalize_option_text(text)
    best_id = None
    best_score = 0
    for option in option_records:
        option_text = _normalize_option_text(f"{option['title']} {option['content']}")
        if not option_text:
            continue
        option_terms = [term for term in option_text.split() if len(term) > 2]
        if not option_terms:
            continue
        score = sum(1 for term in set(option_terms) if term in decision_text)
        if option_text in decision_text:
            score += len(option_terms)
        if score > best_score:
            best_id = option["id"]
            best_score = score
    return best_id if best_score > 0 else None


def _normalize_option_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\b(?:option|alternative|approach|solution)\s+[a-z0-9]+[:.)-]?", " ", lowered)
    lowered = re.sub(r"\b(?:option|alternative|approach|solution|selected|chosen|choose|decision)\b[:.)-]?", " ", lowered)
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


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
