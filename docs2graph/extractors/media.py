"""Media graph extraction for images, charts, and visual documents."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from ..types import GraphDict, make_edge, make_node

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}
_CHART_CUES = {
    "axis": ("axis", "x-axis", "y-axis"),
    "legend": ("legend",),
    "bar_chart": ("bar", "histogram"),
    "line_chart": ("line", "trend"),
    "pie_chart": ("pie", "slice"),
    "table": ("row", "column", "total"),
}


def extract_media_graph(path: str, text: str = "") -> GraphDict:
    """
    Extract deterministic visual evidence from an image-like source.

    This does not try to understand chart semantics with an LLM. It records
    file metadata, OCR text, and explicit chart cues so a later enrichment
    layer can reason over the visual evidence with provenance.
    """
    p = Path(path)
    root_id = _node_id("media", path)
    nodes: List[Dict[str, Any]] = [
        make_node(root_id, p.name or path, text[:2000] or None, {
            "type": "media_document",
            "source": path,
            "format": p.suffix.lower().lstrip("."),
            "extraction_method": "static",
        })
    ]
    edges: List[Dict[str, Any]] = []

    metadata = _image_metadata(path)
    if metadata:
        metadata_id = f"{root_id}:metadata"
        nodes.append(make_node(metadata_id, "Image metadata", attributes={
            "type": "image_metadata",
            "source": path,
            "extraction_method": "static",
            **metadata,
        }))
        edges.append(make_edge(root_id, metadata_id, "has_metadata"))

    if text.strip():
        ocr_id = f"{root_id}:ocr"
        nodes.append(make_node(ocr_id, "OCR text", text.strip(), {
            "type": "ocr_text",
            "source": path,
            "extraction_method": "ocr",
        }))
        edges.append(make_edge(root_id, ocr_id, "has_ocr_text"))

        for cue_type in _detect_chart_cues(text):
            cue_id = f"{root_id}:chart:{cue_type}"
            nodes.append(make_node(cue_id, cue_type.replace("_", " ").title(), attributes={
                "type": "chart_signal",
                "chart_signal": cue_type,
                "source": path,
                "extraction_method": "static",
            }))
            edges.append(make_edge(root_id, cue_id, "has_visual_signal"))

    return {"nodes": nodes, "edges": edges, "current_node_id": root_id}


def is_media_path(path: str) -> bool:
    return Path(path).suffix.lower() in _IMAGE_SUFFIXES


def _image_metadata(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists() or p.suffix.lower() not in _IMAGE_SUFFIXES:
        return {}
    try:
        from PIL import Image
    except ImportError:
        return {"size_bytes": p.stat().st_size}

    with Image.open(str(p)) as img:
        return {
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "image_format": img.format,
            "frames": getattr(img, "n_frames", 1),
            "size_bytes": p.stat().st_size,
        }


def _detect_chart_cues(text: str) -> List[str]:
    lower = text.lower()
    found: List[str] = []
    for cue_type, cues in _CHART_CUES.items():
        if any(re.search(rf"\b{re.escape(cue)}\b", lower) for cue in cues):
            found.append(cue_type)
    if re.search(r"\b(q[1-4]|20\d{2}|19\d{2})\b", lower) and re.search(r"\b(%|\$|revenue|growth|rate)\b", lower):
        found.append("time_series_or_metric_chart")
    return sorted(set(found))


def _node_id(prefix: str, value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")[:100]
    return f"{prefix}:{slug or 'item'}"
