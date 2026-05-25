"""
DocumentGraph — the main public interface for doc2graph.

Build a graph from documents (via loaders + extractors) or directly,
then call .rank() to extract a ranked subgraph for LLM context.
"""

import re
from typing import Any, Dict, List, Optional

from .ranking import personalized_page_rank
from .types import GraphDict, make_edge, make_node


class DocumentGraph:
    """
    A knowledge graph built from documents, files, or any text corpus.

    Nodes represent sections, headings, entities, or documents.
    Edges represent relationships (contains, references, links_to, ...).

    The ``label`` of each node is matched against natural language queries,
    so use descriptive names (e.g. headings, titles, concept names).

    Quick start
    -----------
    >>> g = DocumentGraph()
    >>> g.add_node("intro", "Introduction", content="This paper describes...")
    >>> g.add_node("method", "Methodology", content="We use PPR to rank...")
    >>> g.add_edge("intro", "method", "references")
    >>> context = g.rank("how does the ranking work?")
    >>> print(context["nodes"])

    Load from a Markdown file
    -------------------------
    >>> from doc2graph.loaders.markdown import load_markdown
    >>> g = DocumentGraph.from_markdown("paper.md")
    >>> context = g.rank("what is the main contribution?", k=5)
    """

    def __init__(self) -> None:
        self._nodes: List[Dict[str, Any]] = []
        self._edges: List[Dict[str, Any]] = []
        self._node_ids: set = set()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, graph: GraphDict) -> "DocumentGraph":
        """Build a DocumentGraph from a raw ``{"nodes": [...], "edges": [...]}`` dict."""
        instance = cls()
        instance._nodes = list(graph.get("nodes", []))
        instance._edges = list(graph.get("edges", []))
        instance._node_ids = {n["id"] for n in instance._nodes}
        return instance

    @classmethod
    def from_markdown(cls, path: str) -> "DocumentGraph":
        """
        Build a DocumentGraph from a Markdown file.

        Each heading becomes a node; cross-references detected as edges.
        """
        from .loaders.markdown import load_markdown
        from .extractors.section import extract_section_graph
        text = load_markdown(path)
        graph = extract_section_graph(text, source=path)
        return cls.from_dict(graph)

    @classmethod
    def from_text(cls, path: str, title: Optional[str] = None) -> "DocumentGraph":
        """
        Build a DocumentGraph from a plain text file.

        The file is split into paragraphs; each becomes a node.
        """
        from .loaders.text import load_text
        from .extractors.section import extract_paragraph_graph
        text = load_text(path)
        graph = extract_paragraph_graph(text, source=title or path)
        return cls.from_dict(graph)

    @classmethod
    def from_document(cls, path: str, graph_type: str = "knowledge") -> "DocumentGraph":
        """
        Build a DocumentGraph from a documentation file, URL, or directory.

        ``graph_type`` can be ``"knowledge"``, ``"decision"``, ``"schema"``,
        ``"media"``, or ``"all"``. This is the document-native path for
        papers, manuals, ADRs, RFCs, design docs, visual sources, and mixed
        document corpora.
        """
        from .cli import build_graph

        graph = build_graph(path, graph_type=graph_type)
        return cls.from_dict(graph)

    @classmethod
    def from_directory(
        cls,
        path: str,
        graph_type: str = "knowledge",
        **corpus_options: Any,
    ) -> "DocumentGraph":
        """Build one DocumentGraph from a directory of mixed document files."""
        from .corpus import build_corpus_graph

        graph = build_corpus_graph(path, graph_type=graph_type, **corpus_options)
        return cls.from_dict(graph)

    @classmethod
    def from_texts(cls, texts: List[Dict[str, str]]) -> "DocumentGraph":
        """
        Build a DocumentGraph from a list of ``{"title": ..., "content": ...}`` dicts.

        Useful for HotpotQA-style multi-document inputs where you already
        have the text and just need to build the graph.

        Example
        -------
        >>> g = DocumentGraph.from_texts([
        ...     {"title": "Python", "content": "Python is a programming language..."},
        ...     {"title": "Guido van Rossum", "content": "Creator of Python..."},
        ... ])
        """
        from .extractors.links import detect_cross_doc_links
        instance = cls()
        for item in texts:
            title = item.get("title", "")
            content = item.get("content", "")
            node_id = re.sub(r"\W+", "_", title.lower()).strip("_")
            instance.add_node(node_id, title, content=content)

        # Detect cross-document mentions and add edges
        # e.g. "Python" article mentions "Guido van Rossum" → edge Python→Guido
        cross_links = detect_cross_doc_links(instance._nodes)
        for edge in cross_links:
            instance._edges.append(edge)

        return instance

    def add_node(
        self,
        id: str,
        label: str,
        content: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "DocumentGraph":
        """Add a node. Returns self for chaining."""
        if id in self._node_ids:
            raise ValueError(f"Node with id '{id}' already exists.")
        self._nodes.append(make_node(id, label, content, attributes))
        self._node_ids.add(id)
        return self

    def add_edge(self, from_id: str, to_id: str, label: str) -> "DocumentGraph":
        """Add a directed edge. Returns self for chaining."""
        self._edges.append(make_edge(from_id, to_id, label))
        return self

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def rank(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.85,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Rank document nodes by relevance to a natural language query.

        Uses Personalized PageRank. Returns the top-k nodes plus their
        1-hop neighbours as structured context for an LLM.

        Parameters
        ----------
        query : str
            Natural language question.
        k : int
            Number of top-ranked seed nodes (default 5).
        alpha : float
            PPR damping factor (default 0.85).

        Returns
        -------
        dict
            ``{"nodes": [...], "edges": [...]}``
            Top-k nodes carry a ``"score"`` field.
        """
        return personalized_page_rank(
            query=query,
            graph=self.to_dict(),
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
            k=k,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> GraphDict:
        """Return raw ``{"nodes": [...], "edges": [...]}``."""
        return {"nodes": self._nodes, "edges": self._edges}

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return f"DocumentGraph(nodes={len(self._nodes)}, edges={len(self._edges)})"
