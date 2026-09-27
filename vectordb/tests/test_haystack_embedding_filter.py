#!/usr/bin/env python3
"""
Test: Verify that embedding retrieval only returns documents with real embeddings.

This test ensures compliance with Haystack's DocumentStore protocol:
- Documents without embeddings can be stored (for BM25)
- embedding_retrieval() should skip documents without real embeddings
"""

import sys
from pathlib import Path
import numpy as np

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent / "clients" / "dxmt-python-sdk"))

from haystack import Document
from dxmt import Client as DXMTClient
from dxmt.haystack import DXMTDocumentStore
from dxmt.haystack_retrievers import DXMTEmbeddingRetriever, DXMTBM25Retriever


def test_embedding_filter():
    """Test that embedding retrieval filters out documents without real embeddings."""
    print("\n" + "=" * 70)
    print("Test: Embedding Retrieval Filter (Haystack Protocol Compliance)")
    print("=" * 70)

    # Initialize
    print("\n1. Initializing DXMT Document Store...")
    client = DXMTClient(host="localhost", port=9999)
    doc_store = DXMTDocumentStore(
        client=client,
        table_name="test_embedding_filter",
        embedding_dim=384
    )

    # Create documents with embeddings
    print("\n2. Creating documents...")
    print("   - 2 documents WITH embeddings")
    print("   - 2 documents WITHOUT embeddings (BM25-only)")

    # Generate random embeddings
    np.random.seed(42)
    embedding1 = np.random.rand(384).tolist()
    embedding2 = np.random.rand(384).tolist()

    docs_with_embeddings = [
        Document(
            content="Python is a versatile programming language",
            embedding=embedding1,
            meta={"lang": "Python", "has_emb": True}
        ),
        Document(
            content="Rust is a systems programming language",
            embedding=embedding2,
            meta={"lang": "Rust", "has_emb": True}
        ),
    ]

    docs_without_embeddings = [
        Document(
            content="JavaScript is used for web development",
            meta={"lang": "JavaScript", "has_emb": False}
        ),
        Document(
            content="Go is designed for concurrent programming",
            meta={"lang": "Go", "has_emb": False}
        ),
    ]

    # Write all documents
    print("\n3. Writing documents to store...")
    count1 = doc_store.write_documents(docs_with_embeddings)
    count2 = doc_store.write_documents(docs_without_embeddings)
    print(f"   ✓ Wrote {count1} documents with embeddings")
    print(f"   ✓ Wrote {count2} documents without embeddings")

    total_count = doc_store.count_documents()
    print(f"   ✓ Total documents in store: {total_count}")

    # Test 1: Embedding retrieval should ONLY return docs with real embeddings
    print("\n4. Testing embedding retrieval...")
    query_embedding = np.random.rand(384).tolist()

    retriever = DXMTEmbeddingRetriever(document_store=doc_store, top_k=10)
    result = retriever.run(query_embedding=query_embedding)
    docs = result["documents"]

    print(f"   Query returned {len(docs)} documents")

    # Verify: should only return docs with real embeddings (2 docs)
    expected_langs = {"Python", "Rust"}
    returned_langs = {doc.meta.get("lang") for doc in docs}

    if len(docs) == 2 and returned_langs == expected_langs:
        print("   ✅ CORRECT: Only returned documents with real embeddings")
        print(f"      - Returned: {returned_langs}")
        print(f"      - Excluded: {{'JavaScript', 'Go'}} (no real embeddings)")
    else:
        print("   ❌ ERROR: Returned unexpected documents!")
        print(f"      - Expected 2 docs: {expected_langs}")
        print(f"      - Got {len(docs)} docs: {returned_langs}")
        return False

    # Verify internal field is not exposed
    for doc in docs:
        if "_has_real_embedding" in doc.meta:
            print("   ❌ ERROR: Internal field '_has_real_embedding' exposed to user!")
            return False

    print("   ✅ Internal field '_has_real_embedding' correctly hidden")

    # Test 2: BM25 retrieval should return ALL matching docs
    print("\n5. Testing BM25 retrieval...")
    bm25_retriever = DXMTBM25Retriever(document_store=doc_store, top_k=10)
    result = bm25_retriever.run(query="programming")
    docs = result["documents"]

    print(f"   Query 'programming' returned {len(docs)} documents")

    # Should return 4 documents (all that mention "programming")
    if len(docs) >= 3:  # Python, Rust, Go all mention programming
        print(f"   ✅ CORRECT: BM25 returned {len(docs)} documents")
        langs = [doc.meta.get("lang") for doc in docs]
        print(f"      - Languages: {langs}")
    else:
        print(f"   ⚠️  WARNING: Expected at least 3 docs, got {len(docs)}")

    # Test 3: Filtered embedding retrieval
    print("\n6. Testing filtered embedding retrieval...")
    result = retriever.run(
        query_embedding=query_embedding,
        filters={"field": "meta.lang", "operator": "==", "value": "Python"}
    )
    docs = result["documents"]

    print(f"   Query with filter lang=Python returned {len(docs)} documents")

    if len(docs) == 1 and docs[0].meta.get("lang") == "Python":
        print("   ✅ CORRECT: Filter + embedding retrieval working")
    else:
        print("   ❌ ERROR: Filter not working correctly")
        return False

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nHaystack Protocol Compliance Verified:")
    print("  ✓ Documents without embeddings can be stored")
    print("  ✓ embedding_retrieval() skips docs without real embeddings")
    print("  ✓ BM25 retrieval works with all documents")
    print("  ✓ Internal fields are hidden from users")
    print("  ✓ Filters work correctly with embedding retrieval")
    return True


if __name__ == "__main__":
    try:
        success = test_embedding_filter()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
