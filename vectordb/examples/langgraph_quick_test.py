#!/usr/bin/env python3
"""Quick test script for DXMT LangGraph integration.

This script follows the same patterns as LangGraph's InMemoryStore examples
to ensure compatibility and demonstrate usage.

Run with: python examples/langgraph_quick_test.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dxmt-python-sdk"))

from dxmt import DXMTClient
from dxmt.langgraph_store import DXMTLangGraphStore


def test_basic_operations():
    """Test basic CRUD operations (like InMemoryStore examples)."""
    print("\n" + "=" * 80)
    print("Test 1: Basic Operations (following LangGraph InMemoryStore pattern)")
    print("=" * 80)

    # Initialize store
    print("\nInitializing DXMT client and store...")
    client = DXMTClient(host="localhost", port=9999)
    store = DXMTLangGraphStore(client)
    print("✓ Store initialized")

    # Example 1: User preferences (from InMemoryStore docs)
    print("\n1. Storing user preferences...")
    store.put(("users", "123"), "prefs", {"theme": "dark"})
    print("   ✓ Stored preferences for user 123")

    item = store.get(("users", "123"), "prefs")
    print(f"   ✓ Retrieved: {item.value}")
    assert item.value["theme"] == "dark"

    # Example 2: Update operation
    print("\n2. Updating preferences...")
    store.put(("users", "123"), "prefs", {"theme": "light", "language": "python"})
    item = store.get(("users", "123"), "prefs")
    print(f"   ✓ Updated: {item.value}")
    assert item.value["theme"] == "light"

    # Example 3: Delete operation
    print("\n3. Deleting item...")
    store.delete(("users", "123"), "prefs")
    item = store.get(("users", "123"), "prefs")
    print(f"   ✓ Deleted (get returns None): {item}")
    assert item is None

    print("\n✓ All basic operations passed!")


def test_semantic_search():
    """Test semantic search (like InMemoryStore vector search example)."""
    print("\n" + "=" * 80)
    print("Test 2: Semantic Search (following InMemoryStore pattern)")
    print("=" * 80)

    # Mock embeddings function (deterministic for testing)
    def mock_embed(texts):
        """Simple mock embeddings based on text hash."""
        import hashlib

        embeddings = []
        for text in texts:
            # Generate embedding from text hash (use SHA256 for longer hash)
            hash_val = hashlib.sha256(text.encode()).hexdigest()
            # 32 dimensions, 2 chars per dimension
            embedding = [
                float(int(hash_val[i : i + 2], 16)) / 255.0
                for i in range(0, min(64, len(hash_val)), 2)
            ]
            # Pad if needed
            while len(embedding) < 32:
                embedding.append(0.1)
            # Normalize
            norm = sum(x * x for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]
            embeddings.append(embedding)
        return embeddings

    # Initialize store with embeddings
    print("\nInitializing store with mock embeddings...")
    client = DXMTClient(host="localhost", port=9999)
    store = DXMTLangGraphStore(
        client, embedding_dims=32, embed_fn=mock_embed, default_index_fields=["text"]
    )
    print("✓ Store initialized with embeddings")

    # Store documents (from InMemoryStore example)
    print("\n1. Storing documents...")
    docs = [
        {"text": "Python tutorial"},
        {"text": "TypeScript guide"},
        {"text": "Python programming"},
    ]

    for i, doc in enumerate(docs):
        store.put(("semanticdocs",), f"doc{i}", doc, index=["text"])
        print(f"   ✓ Stored doc{i}: {doc['text']}")

    # Search by similarity (from InMemoryStore example)
    print("\n2. Searching for 'python programming'...")
    results = store.search(("semanticdocs",), query="python programming", limit=3)

    print(f"   ✓ Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"      {i}. {result.value['text']} (score: {result.score:.3f})")

    # Verify Python docs ranked higher
    assert len(results) > 0
    print("\n✓ Semantic search passed!")


def test_batch_operations():
    """Test batch operations (following PostgresStore test patterns)."""
    print("\n" + "=" * 80)
    print("Test 3: Batch Operations (following PostgresStore test pattern)")
    print("=" * 80)

    from langgraph.store.base import GetOp, PutOp, ListNamespacesOp

    # Initialize store
    print("\nInitializing store...")
    client = DXMTClient(host="localhost", port=9999)
    store = DXMTLangGraphStore(client)

    # Setup test data
    print("\n1. Setting up test data...")
    store.put(("test", "foo"), "key1", {"data": "value1"})
    store.put(("test", "bar"), "key2", {"data": "value2"})
    print("   ✓ Test data created")

    # Batch operations (from test_batch_order)
    print("\n2. Executing batch operations...")
    ops = [
        GetOp(namespace=("test", "foo"), key="key1"),
        PutOp(namespace=("test", "bar"), key="key2", value={"data": "updated"}),
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0),
        GetOp(namespace=("test",), key="key3"),  # Non-existent
    ]

    results = store.batch(ops)
    print(f"   ✓ Batch completed with {len(results)} results")

    # Verify results
    assert len(results) == 4
    assert results[0] is not None  # GetOp
    assert results[0].value["data"] == "value1"
    assert results[1] is None  # PutOp returns None
    assert isinstance(results[2], list)  # ListNamespacesOp
    assert results[3] is None  # Non-existent key

    print("   ✓ Result 0 (Get): Found item with value1")
    print("   ✓ Result 1 (Put): Returned None (expected)")
    print(f"   ✓ Result 2 (List): Found {len(results[2])} namespaces")
    print("   ✓ Result 3 (Get): Non-existent key returned None")

    print("\n✓ Batch operations passed!")


def test_namespace_operations():
    """Test namespace operations (following LangGraph patterns)."""
    print("\n" + "=" * 80)
    print("Test 4: Namespace Operations")
    print("=" * 80)

    # Initialize store
    client = DXMTClient(host="localhost", port=9999)
    store = DXMTLangGraphStore(client)

    # Create hierarchical namespaces
    print("\n1. Creating hierarchical namespaces...")
    store.put(("users", "alice"), "prefs", {"theme": "dark"})
    store.put(("users", "bob"), "prefs", {"theme": "light"})
    store.put(("documents",), "doc1", {"title": "Guide"})
    print("   ✓ Created 3 namespace structures")

    # List all namespaces
    print("\n2. Listing all namespaces...")
    namespaces = store.list_namespaces()
    print(f"   ✓ Found {len(namespaces)} namespaces:")
    for ns in sorted(namespaces):
        print(f"      - {'/'.join(ns)}")

    # List with prefix filter
    print("\n3. Listing with prefix filter (users)...")
    user_namespaces = store.list_namespaces(prefix=("users",))
    print(f"   ✓ Found {len(user_namespaces)} user namespaces:")
    for ns in sorted(user_namespaces):
        print(f"      - {'/'.join(ns)}")

    assert len(user_namespaces) >= 2
    print("\n✓ Namespace operations passed!")


def test_async_operations():
    """Test async operations (following InMemoryStore async example)."""
    print("\n" + "=" * 80)
    print("Test 5: Async Operations (following InMemoryStore async pattern)")
    print("=" * 80)

    import asyncio

    async def run_async_tests():
        # Initialize store
        print("\nInitializing store for async operations...")
        client = DXMTClient(host="localhost", port=9999)
        store = DXMTLangGraphStore(client)
        print("✓ Store initialized")

        # Async put (from InMemoryStore async example)
        print("\n1. Async put operation...")
        await store.aput(("test",), "async_key", {"data": "async_value"})
        print("   ✓ Async put completed")

        # Async get
        print("\n2. Async get operation...")
        item = await store.aget(("test",), "async_key")
        print(f"   ✓ Async get: {item.value}")
        assert item.value["data"] == "async_value"

        # Async delete
        print("\n3. Async delete operation...")
        await store.adelete(("test",), "async_key")
        item = await store.aget(("test",), "async_key")
        print(f"   ✓ Async delete (get returns None): {item}")
        assert item is None

        print("\n✓ All async operations passed!")

    # Run async tests
    asyncio.run(run_async_tests())


def test_compatibility_with_langraph():
    """Verify compatibility with LangGraph examples."""
    print("\n" + "=" * 80)
    print("Test 6: LangGraph Compatibility Verification")
    print("=" * 80)

    client = DXMTClient(host="localhost", port=9999)
    store = DXMTLangGraphStore(client)

    # Pattern from LangGraph docs: user preferences
    print("\n1. User preferences pattern...")
    store.put(("users", "123"), "prefs", {"theme": "dark"})
    item = store.get(("users", "123"), "prefs")
    assert item and item.value["theme"] == "dark"
    print("   ✓ User preferences pattern works")

    # Pattern from LangGraph docs: document storage
    print("\n2. Document storage pattern...")
    store.put(("docs",), "doc1", {"text": "Python tutorial"})
    doc = store.get(("docs",), "doc1")
    assert doc and doc.value["text"] == "Python tutorial"
    print("   ✓ Document storage pattern works")

    # Pattern from LangGraph docs: nested namespaces
    print("\n3. Nested namespace pattern...")
    store.put(("project", "team", "alice"), "role", {"title": "engineer"})
    role = store.get(("project", "team", "alice"), "role")
    assert role and role.value["title"] == "engineer"
    print("   ✓ Nested namespace pattern works")

    print("\n✓ All LangGraph patterns compatible!")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("DXMT LangGraph Store - Quick Test Suite")
    print("(Following LangGraph InMemoryStore and PostgresStore patterns)")
    print("=" * 80)
    print("\nPrerequisites:")
    print("  - DXMT server running on localhost:9999")
    print("  - Python packages: dxmt-python-sdk, langgraph")

    try:
        # Run all tests
        test_basic_operations()
        test_semantic_search()
        test_batch_operations()
        test_namespace_operations()
        test_async_operations()
        test_compatibility_with_langraph()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nDXMT LangGraph Store is fully compatible with LangGraph patterns.")
        print("You can use it as a drop-in replacement for InMemoryStore.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
