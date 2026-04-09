"""
doc2graph — extract knowledge graphs from documents.

Query what's in your files without dumping the whole text into a prompt.
"""

from .graph import DocumentGraph
from .ranking import personalized_page_rank
from .types import make_node, make_edge

__version__ = "0.1.0"
__all__ = ["DocumentGraph", "personalized_page_rank", "make_node", "make_edge"]
