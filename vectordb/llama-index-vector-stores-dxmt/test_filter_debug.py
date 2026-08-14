#!/usr/bin/env python3
"""Debug what's being sent to the server"""

import sys
import uuid
import json
sys.path.insert(0, '/localdisk/code/dxmt/vectordb/dxmt-python-sdk')

from llama_index.vector_stores.dxmt import DXMTVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    FilterCondition,
)
from llama_index.vector_stores.dxmt.utils import convert_llamaindex_filters_to_dxmt

# Create test filter
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="year", value=2000, operator=FilterOperator.GTE),
        MetadataFilter(key="popularity", value=80, operator=FilterOperator.GTE),
    ],
    condition=FilterCondition.AND,
)

print("LlamaIndex MetadataFilters:")
print(f"  Filters: {filters.filters}")
print(f"  Condition: {filters.condition}")

print("\nConverted to DXMT format:")
dxmt_filter = convert_llamaindex_filters_to_dxmt(filters)
print(json.dumps(dxmt_filter, indent=2))

print("\nNow let's test with actual data...")

# Create unique table
table_name = f"debug_test_{uuid.uuid4().hex[:8]}"

store = DXMTVectorStore(
    table_name=table_name,
    dimension=128,
    host="localhost",
    port=9999,
)

# Add documents
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

print(f"\nAdding {len(docs)} documents...")
ids = store.add(docs)
print(f"Added: {ids}")

print("\nExpected matches:")
print("  Rust: year=2015 (>=2000 ✓), popularity=85 (>=80 ✓) → MATCH")
print("  Go: year=2009 (>=2000 ✓), popularity=80 (>=80 ✓) → MATCH")
print("  Python: year=1991 (>=2000 ✗) → NO MATCH")

print("\nQuerying with filter...")
query = VectorStoreQuery(
    query_embedding=[0.15 + i*0.01 for i in range(128)],
    similarity_top_k=5,
    filters=filters,
)

result = store.query(query)
print(f"\nResults: {len(result.nodes)} documents")
if len(result.nodes) > 0:
    for i, node in enumerate(result.nodes, 1):
        print(f"  {i}. {node.metadata['language']}: year={node.metadata['year']}, popularity={node.metadata['popularity']}")
else:
    print("  NO RESULTS!")
    print("\n  This suggests the filter is not working correctly on the server side.")
    print("  Let me check what the DXMT SDK actually sends...")

    # Try to see what the SDK sends
    from dxmt.filters import convert_simple_filter_to_dxmt

    print("\n  DXMT SDK's convert_simple_filter_to_dxmt on our filter:")
    sdk_converted = convert_simple_filter_to_dxmt(dxmt_filter)
    print(f"  {json.dumps(sdk_converted, indent=2)}")

# Cleanup
store.clear()
print("\nCleaned up.")
