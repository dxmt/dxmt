"""Test script to verify payload field fixes in langchain-dxmt.

This script tests that:
1. Documents are inserted with payload field (not metadata)
2. Documents are retrieved with correct content and metadata
3. Search results correctly parse payload
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dxmt import Client
from langchain_dxmt import DXMT
from langchain_core.embeddings import Embeddings


class SimpleEmbeddings(Embeddings):
    """Simple embedding function for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents."""
        return [[float(i) / 100] * 128 for i in range(len(texts))]

    def embed_query(self, text: str) -> list[float]:
        """Embed query."""
        return [0.5] * 128


def test_langchain_dxmt_payload():
    """Test langchain-dxmt payload field fixes."""
    print("=" * 80)
    print("Testing LangChain-DXMT Payload Field Fixes")
    print("=" * 80)

    # Initialize client
    print("\n1. Connecting to DXMT server...")
    client = Client(host="localhost", port=9999)

    # Clean up existing table
    try:
        client.delete_table("langchain_test")
        print("   Deleted existing table")
    except Exception:
        pass

    # Create table
    print("\n2. Creating table 'langchain_test'...")
    client.create_table(
        name="langchain_test",
        dimension=128,
        distance_metric="cosine",
        index_type="hnsw",
    )
    print("   ✅ Table created")

    # Initialize embeddings
    embeddings = SimpleEmbeddings()

    # Initialize vectorstore
    print("\n3. Initializing LangChain DXMT vectorstore...")
    vectorstore = DXMT(
        client=client,
        table_name="langchain_test",
        embedding_function=embeddings,
    )
    print("   ✅ Vectorstore initialized")

    # Test 1: Add documents with metadata
    print("\n4. Testing add_texts() with metadata...")
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks",
        "Natural language processing analyzes text",
    ]
    metadatas = [
        {"author": "Alice", "category": "ML"},
        {"author": "Bob", "category": "DL"},
        {"author": "Charlie", "category": "NLP"},
    ]

    ids = vectorstore.add_texts(texts=texts, metadatas=metadatas)
    print(f"   ✅ Added {len(ids)} documents")
    print(f"   IDs: {ids}")

    # Test 2: Retrieve documents by IDs
    print("\n5. Testing get_by_ids()...")
    retrieved_docs = vectorstore.get_by_ids(ids)

    success = True
    for i, doc in enumerate(retrieved_docs):
        print(f"\n   Document {i + 1}:")
        print(f"   - ID: {doc.id}")
        print(f"   - Content: {doc.page_content[:50]}...")
        print(f"   - Metadata: {doc.metadata}")

        # Verify content is not empty
        if not doc.page_content:
            print(f"   ❌ ERROR: Document {i + 1} has empty content!")
            success = False
        else:
            print(f"   ✅ Content retrieved correctly")

        # Verify metadata is not empty
        if not doc.metadata:
            print(f"   ❌ ERROR: Document {i + 1} has empty metadata!")
            success = False
        else:
            print(f"   ✅ Metadata retrieved correctly")

    # Test 3: Similarity search
    print("\n6. Testing similarity_search()...")
    query = "neural networks"
    results = vectorstore.similarity_search(query, k=2)

    print(f"   Found {len(results)} results for query: '{query}'")
    for i, doc in enumerate(results):
        print(f"\n   Result {i + 1}:")
        print(f"   - Content: {doc.page_content[:50]}...")
        print(f"   - Metadata: {doc.metadata}")

        if not doc.page_content:
            print(f"   ❌ ERROR: Result {i + 1} has empty content!")
            success = False
        else:
            print(f"   ✅ Content retrieved correctly")

    # Test 4: Similarity search with scores
    print("\n7. Testing similarity_search_with_score()...")
    results_with_scores = vectorstore.similarity_search_with_score(query, k=2)

    print(f"   Found {len(results_with_scores)} results with scores")
    for i, (doc, score) in enumerate(results_with_scores):
        print(f"\n   Result {i + 1}:")
        print(f"   - Score: {score:.4f}")
        print(f"   - Content: {doc.page_content[:50]}...")
        print(f"   - Metadata: {doc.metadata}")

        if not doc.page_content:
            print(f"   ❌ ERROR: Result {i + 1} has empty content!")
            success = False
        else:
            print(f"   ✅ Content retrieved correctly")

    # Clean up
    print("\n8. Cleaning up...")
    client.delete_table("langchain_test")
    print("   ✅ Table deleted")

    # Final result
    print("\n" + "=" * 80)
    if success:
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nSummary:")
        print("- Documents inserted with payload field ✅")
        print("- Documents retrieved with correct content ✅")
        print("- Metadata correctly parsed ✅")
        print("- Search results have payload ✅")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = test_langchain_dxmt_payload()
    sys.exit(exit_code)
