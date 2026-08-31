#!/usr/bin/env python3
"""Simplified LangGraph integration test that works within DXMT constraints.

This test demonstrates basic functionality that works with DXMT's current
limitations (no true upsert support).
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dxmt-python-sdk"))

from dxmt import DXMTClient
from dxmt.langgraph_store import DXMTLangGraphStore


def main():
    print("\n" + "=" * 80)
    print("DXMT LangGraph Store - Simple Validation Test")
    print("=" * 80)
    
    client = DXMTClient(host="localhost", port=9999)
    store = DXMTLangGraphStore(client)
    
    # Test 1: Basic put and get
    print("\nTest 1: Basic put and get operations")
    store.put(("users", "alice"), "profile", {"name": "Alice", "age": 30})
    item = store.get(("users", "alice"), "profile")
    assert item is not None
    assert item.value["name"] == "Alice"
    print("   ✓ Put and get work correctly")
    
    # Test 2: Multiple namespaces
    print("\nTest 2: Multiple namespaces")
    store.put(("users", "bob"), "profile", {"name": "Bob", "age": 25})
    store.put(("docs",), "readme", {"title": "README", "content": "Hello"})
    
    namespaces = store.list_namespaces()
    print(f"   ✓ Created {len(namespaces)} namespaces")
    
    # Test 3: Delete
    print("\nTest 3: Delete operations")
    store.delete(("users", "alice"), "profile")
    item = store.get(("users", "alice"), "profile")
    assert item is None
    print("   ✓ Delete works correctly")
    
    # Test 4: Semantic search (with mock embeddings)
    print("\nTest 4: Semantic search")
    import hashlib
    def mock_embed(texts):
        embeddings = []
        for text in texts:
            hash_val = hashlib.sha256(text.encode()).hexdigest()
            embedding = [float(int(hash_val[i:i+2], 16)) / 255.0 
                        for i in range(0, min(64, len(hash_val)), 2)]
            while len(embedding) < 32:
                embedding.append(0.1)
            norm = sum(x * x for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]
            embeddings.append(embedding)
        return embeddings
    
    search_store = DXMTLangGraphStore(
        client, embedding_dims=32, embed_fn=mock_embed
    )
    
    search_store.put(("articles",), "python1", {"text": "Python programming"}, index=["text"])
    search_store.put(("articles",), "python2", {"text": "Python tutorial"}, index=["text"])
    search_store.put(("articles",), "java1", {"text": "Java programming"}, index=["text"])
    
    results = search_store.search(("articles",), query="python coding", limit=3)
    print(f"   ✓ Found {len(results)} results for 'python coding'")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nNote: DXMT currently has a known limitation with record updates.")
    print("Use unique keys for each record or implement custom update logic.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
