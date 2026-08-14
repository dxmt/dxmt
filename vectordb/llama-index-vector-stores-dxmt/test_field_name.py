#!/usr/bin/env python3
"""Test if 'year' field name has special handling"""

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
table_name = f"fieldname_test_{uuid.uuid4().hex[:8]}"

store = DXMTVectorStore(
    table_name=table_name,
    dimension=128,
    host="localhost",
    port=9999,
)

# Add documents with both 'year' and 'year_value' fields
print("Adding 2 documents...")
docs = [
    TextNode(
        text="Rust",
        id_="doc_rust",
        embedding=[0.1 + i*0.01 for i in range(128)],
        metadata={"language": "Rust", "year": 2015, "year_value": 2015, "score": 85}
    ),
    TextNode(
        text="Python",
        id_="doc_python",
        embedding=[0.2 + i*0.01 for i in range(128)],
        metadata={"language": "Python", "year": 1991, "year_value": 1991, "score": 95}
    ),
]

ids = store.add(docs)
print(f"✓ Added: {ids}\n")

# Test 1: year >= 2000
print("Test 1: year >= 2000")
filters = MetadataFilters(
    filters=[MetadataFilter(key="year", value=2000, operator=FilterOperator.GTE)]
)
query = VectorStoreQuery(
    query_embedding=[0.15 + i*0.01 for i in range(128)],
    similarity_top_k=5,
    filters=filters,
)
result = store.query(query)
print(f"  Results: {len(result.nodes)}")
for node in result.nodes:
    print(f"    - {node.metadata['language']}: year={node.metadata['year']}")
print()

# Test 2: year_value >= 2000
print("Test 2: year_value >= 2000")
filters = MetadataFilters(
    filters=[MetadataFilter(key="year_value", value=2000, operator=FilterOperator.GTE)]
)
query = VectorStoreQuery(
    query_embedding=[0.15 + i*0.01 for i in range(128)],
    similarity_top_k=5,
    filters=filters,
)
result = store.query(query)
print(f"  Results: {len(result.nodes)}")
for node in result.nodes:
    print(f"    - {node.metadata['language']}: year_value={node.metadata['year_value']}")
print()

# Test 3: score >= 90
print("Test 3: score >= 90")
filters = MetadataFilters(
    filters=[MetadataFilter(key="score", value=90, operator=FilterOperator.GTE)]
)
query = VectorStoreQuery(
    query_embedding=[0.15 + i*0.01 for i in range(128)],
    similarity_top_k=5,
    filters=filters,
)
result = store.query(query)
print(f"  Results: {len(result.nodes)}")
for node in result.nodes:
    print(f"    - {node.metadata['language']}: score={node.metadata['score']}")
print()

# Test 4: Direct DXMT SDK test
print("Test 4: Direct SDK test - year >= 2000")
from dxmt.filters import convert_simple_filter_to_dxmt

filter_dict = {
    "must": [
        {
            "key": "year",
            "operation": {"type": "GreaterThanOrEqual", "value": 2000}
        }
    ]
}
raw_results = store.client.search(
    table_name,
    vector=[0.15 + i*0.01 for i in range(128)],
    k=5,
    filter=filter_dict,
)
print(f"  SDK Results: {len(raw_results)}")
for hit in raw_results:
    payload = hit.get('payload', {})
    print(f"    - {payload.get('language')}: year={payload.get('year')}")

# Cleanup
print("\nCleaning up...")
store.clear()
print("Done.")
