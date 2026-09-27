#!/usr/bin/env python3
"""
Simple test for DXMT + Haystack BM25 retrieval (no OpenAI required)
"""

import sys
from pathlib import Path

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent / "clients" / "dxmt-python-sdk"))

from haystack import Document
from dxmt import Client as DXMTClient
from dxmt.haystack import DXMTDocumentStore
from dxmt.haystack_retrievers import DXMTBM25Retriever


def test_bm25_retrieval():
    """Test BM25 retrieval without OpenAI."""
    print("\n" + "=" * 70)
    print("Testing DXMT BM25 Retrieval (No OpenAI Required)")
    print("=" * 70)

    # Initialize
    print("\n1. Initializing DXMT Document Store...")
    client = DXMTClient(host="localhost", port=9999)
    document_store = DXMTDocumentStore(
        client=client,
        table_name="test_bm25",
        embedding_dim=384,
    )

    # Create documents (no embeddings needed for BM25)
    print("\n2. Creating documents...")
    documents = [
        Document(
            content="Python is a high-level programming language known for its simplicity.",
            meta={"language": "Python", "topic": "programming"}
        ),
        Document(
            content="Rust is a systems programming language that focuses on safety and performance.",
            meta={"language": "Rust", "topic": "programming"}
        ),
        Document(
            content="JavaScript is the language of the web, used for frontend development.",
            meta={"language": "JavaScript", "topic": "web"}
        ),
        Document(
            content="Machine learning is a subset of artificial intelligence.",
            meta={"topic": "AI"}
        ),
    ]

    # Write documents directly (without embeddings)
    print("\n3. Writing documents to store...")
    document_store.write_documents(documents)
    count = document_store.count_documents()
    print(f"   ✓ Stored {count} documents")

    # Test BM25 retrieval
    print("\n4. Testing BM25 retrieval...")
    retriever = DXMTBM25Retriever(document_store=document_store, top_k=3)

    queries = [
        "programming language",
        "web development",
        "artificial intelligence",
    ]

    for query in queries:
        print(f"\n   Query: '{query}'")
        result = retriever.run(query=query)
        documents = result["documents"]
        print(f"   Found {len(documents)} documents:")
        for i, doc in enumerate(documents, 1):
            score = doc.score if hasattr(doc, 'score') else 'N/A'
            print(f"      {i}. [{score:.4f}] {doc.content[:60]}...")

    # Test with filters
    print("\n5. Testing BM25 with filters...")
    query = "programming"
    filter_dict = {"field": "meta.topic", "operator": "==", "value": "programming"}
    print(f"   Query: '{query}' with filter topic='programming'")
    result = retriever.run(query=query, filters=filter_dict)
    documents = result["documents"]
    print(f"   Found {len(documents)} documents:")
    for i, doc in enumerate(documents, 1):
        score = doc.score if hasattr(doc, 'score') else 'N/A'
        print(f"      {i}. [{score:.4f}] {doc.meta.get('language', 'N/A')}: {doc.content[:50]}...")

    print("\n" + "=" * 70)
    print("✓ BM25 retrieval test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_bm25_retrieval()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
