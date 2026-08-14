#!/usr/bin/env python3
"""Debug year field indexing"""

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
table_name = f"year_debug_{uuid.uuid4().hex[:8]}"

store = DXMTVectorStore(
    table_name=table_name,
    dimension=128,
    host="localhost",
    port=9998,
)

# Add ONE document with year field
print("Adding document with year=2015...")
doc = TextNode(
    text="Rust",
    id_="doc_rust",
    embedding=[0.1 + i*0.01 for i in range(128)],
    metadata={"language": "Rust", "year": 2015}
)

ids = store.add([doc])
print(f"✓ Added: {ids}\n")

# Retrieve it back
print("Retrieving document...")
nodes = store.get_nodes(node_ids=["doc_rust"])
if len(nodes) > 0:
    node = nodes[0]
    print(f"  Text: {node.text}")
    print(f"  Metadata: {node.metadata}")
    print(f"  year type: {type(node.metadata.get('year'))}")
    print(f"  year value: {node.metadata.get('year')}")
print()

# Test EQ first
print("Test 1: year == 2015 (exact match)")
filters = MetadataFilters(
    filters=[MetadataFilter(key="year", value=2015, operator=FilterOperator.EQ)]
)
query = VectorStoreQuery(
    query_embedding=[0.1 + i*0.01 for i in range(128)],
    similarity_top_k=5,
    filters=filters,
)
result = store.query(query)
print(f"  Results: {len(result.nodes)}")
if len(result.nodes) > 0:
    print(f"  ✓ EQ works!")
else:
    print(f"  ✗ EQ doesn't work - payload indexing problem?")
print()

# Test GTE
print("Test 2: year >= 2000 (should match)")
filters = MetadataFilters(
    filters=[MetadataFilter(key="year", value=2000, operator=FilterOperator.GTE)]
)
query = VectorStoreQuery(
    query_embedding=[0.1 + i*0.01 for i in range(128)],
    similarity_top_k=5,
    filters=filters,
)
result = store.query(query)
print(f"  Results: {len(result.nodes)}")
if len(result.nodes) > 0:
    print(f"  ✓ GTE works!")
else:
    print(f"  ✗ GTE doesn't work")
print()

# Test with higher threshold
print("Test 3: year >= 2020 (should NOT match)")
filters = MetadataFilters(
    filters=[MetadataFilter(key="year", value=2020, operator=FilterOperator.GTE)]
)
query = VectorStoreQuery(
    query_embedding=[0.1 + i*0.01 for i in range(128)],
    similarity_top_k=5,
    filters=filters,
)
result = store.query(query)
print(f"  Results: {len(result.nodes)}")
if len(result.nodes) == 0:
    print(f"  ✓ Correctly filtered out")
else:
    print(f"  ✗ Should have been filtered out")
print()

# Test LT
print("Test 4: year < 2020 (should match)")
filters = MetadataFilters(
    filters=[MetadataFilter(key="year", value=2020, operator=FilterOperator.LT)]
)
query = VectorStoreQuery(
    query_embedding=[0.1 + i*0.01 for i in range(128)],
    similarity_top_k=5,
    filters=filters,
)
result = store.query(query)
print(f"  Results: {len(result.nodes)}")
if len(result.nodes) > 0:
    print(f"  ✓ LT works!")
else:
    print(f"  ✗ LT doesn't work")

# Cleanup
print("\nCleaning up...")
store.clear()
print("Done.")
