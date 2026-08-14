#!/usr/bin/env python3
"""Check what metadata is actually stored"""

import sys
import uuid
import json
sys.path.insert(0, '/localdisk/code/dxmt/vectordb/dxmt-python-sdk')

from llama_index.vector_stores.dxmt import DXMTVectorStore
from llama_index.core.schema import TextNode

# Create unique table
table_name = f"metadata_test_{uuid.uuid4().hex[:8]}"

store = DXMTVectorStore(
    table_name=table_name,
    dimension=128,
    host="localhost",
    port=9999,
)

# Add one document
print("Adding document with year=2015 (integer)")
doc = TextNode(
    text="Rust",
    id_="doc_rust",
    embedding=[0.1 + i*0.01 for i in range(128)],
    metadata={"language": "Rust", "year": 2015, "popularity": 85}
)

print(f"  Input metadata type: year={type(doc.metadata['year'])}, value={doc.metadata['year']}")

ids = store.add([doc])
print(f"  ✓ Added: {ids}\n")

# Retrieve the document
print("Retrieving document...")
nodes = store.get_nodes(node_ids=["doc_rust"])

if len(nodes) > 0:
    node = nodes[0]
    print(f"  Retrieved metadata:")
    print(f"    {json.dumps(node.metadata, indent=4)}")
    print(f"\n  Field types:")
    for key, value in node.metadata.items():
        print(f"    {key}: {type(value)} = {value}")
else:
    print("  ERROR: Could not retrieve document!")

# Now let's check what the DXMT SDK returns directly
print("\nChecking raw DXMT response...")
from llama_index.core.vector_stores.types import VectorStoreQuery

query = VectorStoreQuery(
    query_embedding=[0.1 + i*0.01 for i in range(128)],
    similarity_top_k=1,
)
result = store.query(query)

if len(result.nodes) > 0:
    node = result.nodes[0]
    print(f"  Metadata from query result:")
    print(f"    {json.dumps(node.metadata, indent=4)}")

    # Check the raw client response
    print("\n  Checking raw SDK response...")
    raw_results = store.client.search(
        table_name,
        vector=[0.1 + i*0.01 for i in range(128)],
        k=1,
    )
    print(f"  Raw results: {len(raw_results)} hits")
    if len(raw_results) > 0:
        hit = raw_results[0]
        print(f"  Raw hit payload:")
        print(f"    {json.dumps(hit.get('payload', {}), indent=4)}")

# Cleanup
print("\nCleaning up...")
store.clear()
print("Done.")
