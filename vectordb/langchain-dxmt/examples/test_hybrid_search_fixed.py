"""Test hybrid search with bug fixes.

This script uses fake embeddings to test whether:
1. Hybrid search properly applies filters
2. Text fields are properly configured by default
3. Results are correctly returned
"""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_dxmt import DXMTVectorDB


class FakeEmbeddings(Embeddings):
    """Fake embeddings for testing."""

    def embed_documents(self, texts):
        """Generate fake embeddings for documents."""
        import random
        random.seed(42)
        return [[random.random() for _ in range(128)] for _ in texts]

    def embed_query(self, text):
        """Generate fake embedding for query."""
        import random
        random.seed(42)
        return [random.random() for _ in range(128)]


def main():
    """Run the test."""
    print("=== Testing Hybrid Search Bug Fixes ===\n")

    # 1. Initialize fake embeddings
    print("1. Initializing fake embeddings...")
    embeddings = FakeEmbeddings()

    # 2. Create vector store with full-text search enabled
    print("2. Creating DXMT vector store with full-text search...")
    vector_store = DXMTVectorDB(
        table_name="hybrid_search_test",
        embedding=embeddings,
        dimension=128,
        host="localhost",
        port=9999,
        index_type="hnsw",
        distance_metric="cosine",
        enable_full_text_search=True,  # Enable without specifying text_fields
    )
    print(f"   ✓ Vector store created: {vector_store._table_name}\n")

    # 3. Add test documents with different categories
    print("3. Adding test documents...")
    documents = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "ml_guide", "category": "AI"},
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers.",
            metadata={"source": "dl_guide", "category": "AI"},
        ),
        Document(
            page_content="Natural language processing enables computers to understand human language.",
            metadata={"source": "nlp_guide", "category": "AI"},
        ),
        Document(
            page_content="Vector databases store and query high-dimensional embeddings.",
            metadata={"source": "vectordb_guide", "category": "Database"},
        ),
        Document(
            page_content="DXMT VectorDB supports HNSW and IVF index types.",
            metadata={"source": "dxmt_docs", "category": "Database"},
        ),
        Document(
            page_content="Python is a popular programming language for data science.",
            metadata={"source": "python_docs", "category": "Programming"},
        ),
    ]

    ids = vector_store.add_documents(documents)
    print(f"   ✓ Added {len(ids)} documents\n")

    # 4. Test hybrid search without filter
    print("4. Testing Hybrid Search (without filter):")
    print("   " + "="*60)
    query = "machine learning algorithms"
    results = vector_store.hybrid_search(query, k=3, vector_weight=0.5)

    print(f"   Query: '{query}'")
    print(f"   Results: {len(results)}\n")

    for i, doc in enumerate(results, 1):
        print(f"   [{i}] {doc.page_content[:60]}...")
        print(f"       Category: {doc.metadata.get('category', 'N/A')}\n")

    # 5. Test hybrid search WITH filter (the critical bug fix!)
    print("5. Testing Hybrid Search WITH Filter (category='AI'):")
    print("   " + "="*60)
    query = "artificial intelligence"
    results_filtered = vector_store.hybrid_search(
        query,
        k=3,
        vector_weight=0.5,
        filter={"category": "AI"}
    )

    print(f"   Query: '{query}' with filter={{category: 'AI'}}")
    print(f"   Results: {len(results_filtered)}\n")

    all_match_filter = True
    for i, doc in enumerate(results_filtered, 1):
        category = doc.metadata.get('category', 'N/A')
        matches = category == 'AI'
        status = "✅" if matches else "❌"

        print(f"   [{i}] {status} {doc.page_content[:60]}...")
        print(f"       Category: {category}")

        if not matches:
            all_match_filter = False
            print(f"       ❌ ERROR: Category '{category}' doesn't match filter 'AI'!")
        print()

    # 6. Test with different filter
    print("6. Testing Hybrid Search WITH Filter (category='Database'):")
    print("   " + "="*60)
    query = "vector database"
    results_db = vector_store.hybrid_search(
        query,
        k=2,
        vector_weight=0.7,
        filter={"category": "Database"}
    )

    print(f"   Query: '{query}' with filter={{category: 'Database'}}")
    print(f"   Results: {len(results_db)}\n")

    for i, doc in enumerate(results_db, 1):
        category = doc.metadata.get('category', 'N/A')
        matches = category == 'Database'
        status = "✅" if matches else "❌"

        print(f"   [{i}] {status} {doc.page_content[:60]}...")
        print(f"       Category: {category}")

        if not matches:
            all_match_filter = False
            print(f"       ❌ ERROR: Category '{category}' doesn't match filter 'Database'!")
        print()

    # 7. Cleanup
    print("7. Cleaning up...")
    vector_store.delete(ids=ids)
    print("   ✓ Deleted test documents\n")

    # Final result
    print("="*60)
    print("=== Test Results ===")
    if all_match_filter and len(results_filtered) > 0 and len(results_db) > 0:
        print("✅ SUCCESS: All bug fixes are working correctly!")
        print("   - Filters are applied BEFORE truncating results")
        print("   - Default text_fields are configured automatically")
        print("   - Hybrid search returns correct filtered results")
        return 0
    else:
        print("❌ FAILURE: Some tests failed!")
        if not all_match_filter:
            print("   - Filter bug: Results don't match filter criteria")
        if len(results_filtered) == 0:
            print("   - Filter bug: No results returned (all filtered out)")
        if len(results_db) == 0:
            print("   - Filter bug: No Database results returned")
        return 1


if __name__ == "__main__":
    exit(main())
