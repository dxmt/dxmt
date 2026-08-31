#!/usr/bin/env python3
"""Check how node_to_metadata_dict converts metadata"""

import sys
sys.path.insert(0, '/localdisk/code/dxmt/vectordb/dxmt-python-sdk')

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.dxmt.utils import node_to_dxmt_record
import json

# Create a node
node = TextNode(
    text="Rust programming language",
    id_="doc_rust",
    embedding=[0.1] * 128,
    metadata={
        "language": "Rust",
        "year": 2015,
        "popularity": 85,
        "tags": ["systems", "performance"]
    }
)

print("Original node metadata:")
print(f"  {json.dumps(node.metadata, indent=2)}")
print(f"\n  Types:")
for key, value in node.metadata.items():
    print(f"    {key}: {type(value)} = {value}")

print("\n" + "=" * 60)
print("node_to_metadata_dict output:")
metadata_dict = node_to_metadata_dict(node, remove_text=False, flat_metadata=False)
print(f"  {json.dumps(metadata_dict, indent=2, default=str)}")

print("\n" + "=" * 60)
print("node_to_dxmt_record output:")
dxmt_record = node_to_dxmt_record(node, include_text=True, flat_metadata=False)
print(f"  ID: {dxmt_record['id']}")
print(f"  Vector length: {len(dxmt_record['vector'])}")
print(f"  Payload:")
print(f"    {json.dumps(dxmt_record['payload'], indent=4, default=str)}")

print("\n" + "=" * 60)
print("Checking payload field types:")
for key, value in dxmt_record['payload'].items():
    print(f"  {key}: {type(value)} = {value}")
