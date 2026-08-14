#!/usr/bin/env python3
"""Test filters step by step"""

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
table_name = f"simple_test_{uuid.uuid4().hex[:8]}"

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
        metadata={"language": "Rust", "year": 2015, "popularity": 85}
    ),
    TextNode(
        text="Go",
        id_="doc_go",
        embedding=[0.2 + i*0.01 for i in range(128)],
        metadata={"language": "Go", "year": 2009, "popularity": 80}
    ),
    TextNode(
        text="Python",
        id_="doc_python",
        embedding=[0.15 + i*0.01 for i in range(128)],
        metadata={"language": "Python", "year": 1991, "popularity": 95}
    ),
]

ids = store.add(docs)
print(f"✓ Added: {ids}\n")

# Test 1: No filter
print("Test 1: No filter (should return all 3)")
query = VectorStoreQuery(
    query_embedding=[0.15 + i*0.01 for i in range(128)],
    similarity_top_k=5,
)
result = store.query(query)
print(f"  Results: {len(result.nodes)}")
for node in result.nodes:
    print(f"    - {node.metadata['language']}: year={node.metadata['year']}, pop={node.metadata['popularity']}")
print()

# Test 2: Single equality filter
print("Test 2: language == 'Rust' (should return 1)")
filters = MetadataFilters(
    filters=[MetadataFilter(key="language", value="Rust", operator=FilterOperator.EQ)]
)
query = VectorStoreQuery(
    query_embedding=[0.15 + i*0.01 for i in range(128)],
    similarity_top_k=5,
    filters=filters,
)
result = store.query(query)
print(f"  Results: {len(result.nodes)}")
for node in result.nodes:
    print(f"    - {node.metadata['language']}")
print()

# Test 3: Single GTE filter
print("Test 3: year >= 2000 (should return 2: Rust, Go)")
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

# Test 4: Single GTE filter on popularity
print("Test 4: popularity >= 80 (should return all 3)")
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
print()

# Test 5: Two separate single-condition queries
print("Test 5a: year >= 2000")
filters = MetadataFilters(
    filters=[MetadataFilter(key="year", value=2000, operator=FilterOperator.GTE)]
)
query = VectorStoreQuery(
    query_embedding=[0.15 + i*0.01 for i in range(128)],
    similarity_top_k=5,
    filters=filters,
)
result = store.query(query)
year_matches = set(node.node_id for node in result.nodes)
print(f"  Year matches: {year_matches}")

print("Test 5b: popularity >= 80")
filters = MetadataFilters(
    filters=[MetadataFilter(key="popularity", value=80, operator=FilterOperator.GTE)]
)
query = VectorStoreQuery(
    query_embedding=[0.15 + i*0.01 for i in range(128)],
    similarity_top_k=5,
    filters=filters,
)
result = store.query(query)
pop_matches = set(node.node_id for node in result.nodes)
print(f"  Popularity matches: {pop_matches}")

print(f"  Intersection (expected for AND): {year_matches & pop_matches}")
print()

# Test 6: Double condition AND
print("Test 6: year >= 2000 AND popularity >= 80 (should return 2: Rust, Go)")
from llama_index.core.vector_stores.types import FilterCondition
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="year", value=2000, operator=FilterOperator.GTE),
        MetadataFilter(key="popularity", value=80, operator=FilterOperator.GTE),
    ],
    condition=FilterCondition.AND,
)
query = VectorStoreQuery(
    query_embedding=[0.15 + i*0.01 for i in range(128)],
    similarity_top_k=5,
    filters=filters,
)
result = store.query(query)
print(f"  Results: {len(result.nodes)}")
for node in result.nodes:
    print(f"    - {node.metadata['language']}: year={node.metadata['year']}, pop={node.metadata['popularity']}")

# Cleanup
print("\nCleaning up...")
store.clear()
print("Done.")
