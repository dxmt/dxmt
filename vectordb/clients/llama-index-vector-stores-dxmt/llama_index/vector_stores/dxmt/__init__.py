"""LlamaIndex DXMT Vector Store Integration.

This package provides a vector store implementation for DXMT VectorDB,
enabling seamless integration with LlamaIndex's data framework for LLM applications.
"""

from .base import DXMTVectorStore
from .utils import (
    convert_llamaindex_filters_to_dxmt,
    node_to_dxmt_record,
    dxmt_hit_to_node,
)

__all__ = [
    "DXMTVectorStore",
    "convert_llamaindex_filters_to_dxmt",
    "node_to_dxmt_record",
    "dxmt_hit_to_node",
]

__version__ = "1.4.0"
