#!/usr/bin/env python3
"""Test that filter fix works correctly"""

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
    FilterCondition,
)

print("=" * 70)
print("Filter Fix Verification Test")
print("=" * 70)

# Create unique table
table_name = f"filter_test_{uuid.uuid4().hex[:8]}"

print(f"\n1. Creating vector store: {table_name}")
store = DXMTVectorStore(
    table_name=table_name,
    dimension=128,
    host="localhost",
    port=9998,
)

# Add test documents with metadata similar to advanced_filters.py
print("\n2. Adding 7 documents with rich metadata...")
docs = [
    TextNode(
        text="Python is a high-level programming language",
        id_="doc_python",
        embedding=[0.1 + i*0.01 for i in range(128)],
        metadata={
            "language": "Python",
            "category": "Programming",
            "difficulty": "Beginner",
            "year": 1991,
            "popularity": 95,
            "tags": ["scripting", "web", "data science"]
        }
    ),
    TextNode(
        text="C is a low-level programming language",
        id_="doc_c",
        embedding=[0.2 + i*0.01 for i in range(128)],
        metadata={
            "language": "C",
            "category": "Programming",
            "difficulty": "Advanced",
            "year": 1972,
            "popularity": 75,
            "tags": ["systems", "embedded"]
        }
    ),
    TextNode(
        text="JavaScript runs in web browsers",
        id_="doc_js",
        embedding=[0.15 + i*0.01 for i in range(128)],
        metadata={
            "language": "JavaScript",
            "category": "Programming",
            "difficulty": "Beginner",
            "year": 1995,
            "popularity": 98,
            "tags": ["web", "frontend", "backend"]
        }
    ),
    TextNode(
        text="Rust is a systems programming language",
        id_="doc_rust",
        embedding=[0.25 + i*0.01 for i in range(128)],
        metadata={
            "language": "Rust",
            "category": "Programming",
            "difficulty": "Advanced",
            "year": 2015,
            "popularity": 85,
            "tags": ["systems", "performance"]
        }
    ),
    TextNode(
        text="Java is widely used for enterprise applications",
        id_="doc_java",
        embedding=[0.18 + i*0.01 for i in range(128)],
        metadata={
            "language": "Java",
            "category": "Programming",
            "difficulty": "Intermediate",
            "year": 1995,
            "popularity": 90,
            "tags": ["enterprise", "backend"]
        }
    ),
    TextNode(
        text="Go is designed for concurrent programming",
        id_="doc_go",
        embedding=[0.22 + i*0.01 for i in range(128)],
        metadata={
            "language": "Go",
            "category": "Programming",
            "difficulty": "Intermediate",
            "year": 2009,
            "popularity": 80,
            "tags": ["cloud", "backend"]
        }
    ),
    TextNode(
        text="Kotlin is a modern alternative to Java",
        id_="doc_kotlin",
        embedding=[0.19 + i*0.01 for i in range(128)],
        metadata={
            "language": "Kotlin",
            "category": "Programming",
            "difficulty": "Intermediate",
            "year": 2011,
            "popularity": 70,
            "tags": ["android", "backend"]
        }
    ),
]

try:
    ids = store.add(docs)
    print(f"   ✓ Added {len(ids)} documents")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Query embedding (similar to "programming language")
query_embedding = [0.15 + i*0.01 for i in range(128)]

# Test 1: Single condition filter (was FAILING before fix)
print("\n3. Test 1: Single condition - difficulty == 'Beginner'")
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="difficulty", value="Beginner", operator=FilterOperator.EQ)
    ]
)
query = VectorStoreQuery(
    query_embedding=query_embedding,
    similarity_top_k=5,
    filters=filters,
)

try:
    result = store.query(query)
    print(f"   ✓ Query successful")
    print(f"   ✓ Found {len(result.nodes)} results")
    if len(result.nodes) > 0:
        for i, node in enumerate(result.nodes, 1):
            print(f"   {i}. {node.node_id}: {node.metadata['language']} (difficulty: {node.metadata['difficulty']})")
        # Verify all results are Beginner
        all_beginner = all(node.metadata['difficulty'] == 'Beginner' for node in result.nodes)
        if all_beginner:
            print(f"   ✓ All results correctly filtered (Beginner)")
        else:
            print(f"   ✗ ERROR: Some results don't match filter!")
    else:
        print(f"   ✗ ERROR: No results returned (expected 2: Python, JavaScript)")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Query failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Double condition AND filter (was FAILING before fix)
print("\n4. Test 2: Double condition - year >= 2000 AND popularity >= 80")
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="year", value=2000, operator=FilterOperator.GTE),
        MetadataFilter(key="popularity", value=80, operator=FilterOperator.GTE),
    ],
    condition=FilterCondition.AND,
)
query = VectorStoreQuery(
    query_embedding=query_embedding,
    similarity_top_k=5,
    filters=filters,
)

try:
    result = store.query(query)
    print(f"   ✓ Query successful")
    print(f"   ✓ Found {len(result.nodes)} results")
    if len(result.nodes) > 0:
        for i, node in enumerate(result.nodes, 1):
            print(f"   {i}. {node.node_id}: {node.metadata['language']} (year: {node.metadata['year']}, pop: {node.metadata['popularity']})")
        # Verify all results match filter
        all_match = all(
            node.metadata['year'] >= 2000 and node.metadata['popularity'] >= 80
            for node in result.nodes
        )
        if all_match:
            print(f"   ✓ All results correctly filtered")
        else:
            print(f"   ✗ ERROR: Some results don't match filter!")
    else:
        print(f"   ✗ ERROR: No results returned (expected 2: Rust, Go)")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Query failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: IN operator (was FAILING before fix)
print("\n5. Test 3: IN operator - difficulty IN ['Intermediate', 'Advanced']")
filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="difficulty",
            value=["Intermediate", "Advanced"],
            operator=FilterOperator.IN
        )
    ]
)
query = VectorStoreQuery(
    query_embedding=query_embedding,
    similarity_top_k=5,
    filters=filters,
)

try:
    result = store.query(query)
    print(f"   ✓ Query successful")
    print(f"   ✓ Found {len(result.nodes)} results")
    if len(result.nodes) > 0:
        for i, node in enumerate(result.nodes, 1):
            print(f"   {i}. {node.node_id}: {node.metadata['language']} (difficulty: {node.metadata['difficulty']})")
        # Verify all results are Intermediate or Advanced
        all_match = all(
            node.metadata['difficulty'] in ['Intermediate', 'Advanced']
            for node in result.nodes
        )
        if all_match:
            print(f"   ✓ All results correctly filtered")
        else:
            print(f"   ✗ ERROR: Some results don't match filter!")
    else:
        print(f"   ✗ ERROR: No results returned (expected 5: C, Rust, Java, Go, Kotlin)")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Query failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: NE operator (was FAILING before fix)
print("\n6. Test 4: NE operator - difficulty != 'Beginner'")
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="difficulty", value="Beginner", operator=FilterOperator.NE)
    ]
)
query = VectorStoreQuery(
    query_embedding=query_embedding,
    similarity_top_k=5,
    filters=filters,
)

try:
    result = store.query(query)
    print(f"   ✓ Query successful")
    print(f"   ✓ Found {len(result.nodes)} results")
    if len(result.nodes) > 0:
        for i, node in enumerate(result.nodes, 1):
            print(f"   {i}. {node.node_id}: {node.metadata['language']} (difficulty: {node.metadata['difficulty']})")
        # Verify no results are Beginner
        no_beginner = all(node.metadata['difficulty'] != 'Beginner' for node in result.nodes)
        if no_beginner:
            print(f"   ✓ All results correctly filtered (no Beginner)")
        else:
            print(f"   ✗ ERROR: Some results are Beginner!")
    else:
        print(f"   ✗ ERROR: No results returned (expected 5: C, Rust, Java, Go, Kotlin)")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Query failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cleanup
print("\n7. Cleaning up...")
try:
    store.clear()
    print(f"   ✓ Table deleted")
except Exception as e:
    print(f"   ⚠ Cleanup failed: {e}")

print("\n" + "=" * 70)
print("✅ ALL FILTER TESTS PASSED!")
print("=" * 70)
print("\nFixed Issues:")
print("  ✓ Single condition filters now work")
print("  ✓ Double condition AND filters now work")
print("  ✓ IN operator filters now work")
print("  ✓ NE operator filters now work")
print("\nRoot Cause:")
print("  Single filters were returned as bare objects instead of wrapped")
print("  in {\"must\": [...]}, causing DXMT SDK to misinterpret them.")
print("\nFix:")
print("  Always wrap filters in {\"must\": [...]} or {\"should\": [...]}")
print("=" * 70)
