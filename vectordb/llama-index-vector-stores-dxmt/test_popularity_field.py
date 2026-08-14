#!/usr/bin/env python3
"""Retest popularity field to confirm it works"""

import sys
import uuid
sys.path.insert(0, '/localdisk/code/dxmt/vectordb/dxmt-python-sdk')

from llama_index.vector_stores.dxmt import DXMTVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)

# Create unique table
table_name = f"pop_test_{uuid.uuid4().hex[:8]}"

store = DXMTVectorStore(
    table_name=table_name,
    dimension=128,
    host="localhost",
    port=9999,
)

# Add documents
print("Adding 3 documents...")
docs = [
    TextNode(
        text="Rust",
        id_="doc_rust",
        embedding=[0.1 + i*0.01 for i in range(128)],
        metadata={"language": "Rust", "popularity": 85}
    ),
    TextNode(
        text="Go",
        id_="doc_go",
        embedding=[0.2 + i*0.01 for i in range(128)],
        metadata={"language": "Go", "popularity": 80}
    ),
    TextNode(
        text="Python",
        id_="doc_python",
        embedding=[0.15 + i*0.01 for i in range(128)],
        metadata={"language": "Python", "popularity": 70}
    ),
]

ids = store.add(docs)
print(f"✓ Added: {ids}\n")

# Test: popularity >= 80
print("Test: popularity >= 80 (should return Rust and Go)")
filters = MetadataFilters(
    filters=[MetadataFilter(key="popularity", value=80, operator=FilterOperator.GTE)]
)
query = VectorStoreQuery(
    query_embedding=[0.15 + i*0.01 for i in range(128)],
    similarity_top_k=5,
    filters=filters,
)
result = store.query(query)
print(f"  Results: {len(result.nodes)}")
for node in result.nodes:
    print(f"    - {node.metadata['language']}: popularity={node.metadata['popularity']}")

# Cleanup
print("\nCleaning up...")
store.clear()
print("Done.")
