"""Test script to verify search payload bug fix.

This script uses fake embeddings (no OpenAI API key needed) to test
whether the search endpoint properly returns document content and metadata.
"""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_dxmt import DXMTVectorDB


class FakeEmbeddings(Embeddings):
    """Fake embeddings for testing."""

    def embed_documents(self, texts):
        """Generate fake embeddings for documents."""
        import random
        return [[random.random() for _ in range(128)] for _ in texts]

    def embed_query(self, text):
        """Generate fake embedding for query."""
        import random
        return [random.random() for _ in range(128)]


def main():
    """Run the test."""
    print("=== Testing DXMT Search Payload Fix ===\n")

    # 1. Initialize fake embeddings
    print("1. Initializing fake embeddings...")
    embeddings = FakeEmbeddings()

    # 2. Create vector store
    print("2. Creating DXMT vector store...")
    vector_store = DXMTVectorDB(
        table_name="test_search_fix",
        embedding=embeddings,
        dimension=128,
        host="localhost",
        port=9999,
        index_type="hnsw",
        distance_metric="cosine",
    )
    print(f"   ✓ Vector store created: {vector_store._table_name}\n")

    # 3. Add test documents
    print("3. Adding test documents...")
    documents = [
        Document(
            page_content="The quick brown fox jumps over the lazy dog.",
            metadata={"source": "test1", "category": "animals"},
        ),
        Document(
            page_content="Python is a great programming language.",
            metadata={"source": "test2", "category": "programming"},
        ),
        Document(
            page_content="Machine learning is transforming AI.",
            metadata={"source": "test3", "category": "technology"},
        ),
    ]

    ids = vector_store.add_documents(documents)
    print(f"   ✓ Added {len(ids)} documents with IDs: {ids[:3]}\n")

    # 4. Test similarity search (the bug was here - returned empty content)
    print("4. Testing similarity search...")
    query = "programming"
    results = vector_store.similarity_search(query, k=3)

    print(f"   Query: '{query}'")
    print(f"   Found {len(results)} results:\n")

    all_content_present = True
    all_metadata_present = True

    for i, doc in enumerate(results, 1):
        print(f"   Result {i}:")
        print(f"     Content: {doc.page_content}")
        print(f"     Metadata: {doc.metadata}")

        if not doc.page_content:
            all_content_present = False
            print("     ❌ ERROR: Content is empty!")
        if not doc.metadata:
            all_metadata_present = False
            print("     ❌ ERROR: Metadata is empty!")
        print()

    # 5. Test similarity search with scores
    print("5. Testing similarity search with scores...")
    query = "technology"
    results_with_scores = vector_store.similarity_search_with_score(query, k=2)

    print(f"   Query: '{query}'")
    print(f"   Found {len(results_with_scores)} results:\n")

    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"   Result {i} (score: {score:.4f}):")
        print(f"     Content: {doc.page_content}")
        print(f"     Metadata: {doc.metadata}")

        if not doc.page_content:
            all_content_present = False
            print("     ❌ ERROR: Content is empty!")
        if not doc.metadata:
            all_metadata_present = False
            print("     ❌ ERROR: Metadata is empty!")
        print()

    # 6. Cleanup
    print("6. Cleaning up...")
    vector_store.delete(ids=ids)
    print("   ✓ Deleted test documents\n")

    # Final result
    print("=== Test Results ===")
    if all_content_present and all_metadata_present:
        print("✅ SUCCESS: All search results have content and metadata!")
        print("   The bug fix is working correctly.")
        return 0
    else:
        print("❌ FAILURE: Some search results are missing content or metadata!")
        if not all_content_present:
            print("   - Content is missing from some results")
        if not all_metadata_present:
            print("   - Metadata is missing from some results")
        return 1


if __name__ == "__main__":
    exit(main())
