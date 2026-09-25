"""Hybrid search example for langchain-dxmt.

This example demonstrates DXMT's unique hybrid search capability,
combining vector similarity search with BM25 full-text search.

Prerequisites:
- DXMT VectorDB server running on localhost:9999
- OpenAI API key set in environment

Run:
    python examples/hybrid_search.py
"""

from langchain_core.documents import Document
from embedding import default_embeddings, embed_dimension

from langchain_dxmt import DXMTVectorDB


def main() -> None:
    """Run hybrid search example."""
    print("=== LangChain DXMT Hybrid Search Example ===\n")

    # 1. Initialize embeddings
    print("1. Initializing OpenAI embeddings...")
    embeddings = default_embeddings()

    # 2. Create vector store with full-text search enabled
    print("2. Creating DXMT vector store with full-text search...")
    vector_store = DXMTVectorDB(
        table_name="hybrid_search_demo",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
        host="localhost",
        port=9999,
        index_type="hnsw",
        distance_metric="cosine",
        enable_full_text_search=True,  # Enable BM25 full-text search
    )
    print(f"   ✓ Vector store created with hybrid search capability\n")

    # 3. Add documents with rich text content
    print("3. Adding documents...")
    documents = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
            metadata={"source": "ml_guide", "category": "AI"},
        ),
        Document(
            page_content="Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
            metadata={"source": "dl_guide", "category": "AI"},
        ),
        Document(
            page_content="Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
            metadata={"source": "nlp_guide", "category": "AI"},
        ),
        Document(
            page_content="Vector databases are specialized databases designed to store and query high-dimensional vector embeddings efficiently.",
            metadata={"source": "vectordb_guide", "category": "Database"},
        ),
        Document(
            page_content="DXMT VectorDB supports multiple index types including HNSW, IVF, and IVF-PQ for efficient similarity search at scale.",
            metadata={"source": "dxmt_docs", "category": "Database"},
        ),
        Document(
            page_content="Python is a high-level, interpreted programming language with dynamic semantics and strong support for integration with other languages.",
            metadata={"source": "python_docs", "category": "Programming"},
        ),
    ]

    ids = vector_store.add_documents(documents)
    print(f"   ✓ Added {len(ids)} documents\n")

    # 4. Pure vector search (baseline)
    print("4. Pure Vector Similarity Search:")
    print("   " + "="*60)
    query = "machine learning algorithms"
    results = vector_store.similarity_search(query, k=3)

    print(f"   Query: '{query}'")
    print(f"   Results:\n")
    for i, doc in enumerate(results, 1):
        print(f"   [{i}] {doc.page_content[:80]}...")
        print(f"       Category: {doc.metadata.get('category', 'N/A')}\n")

    # 5. Hybrid search - Balanced (50/50)
    print("5. Hybrid Search (50% vector, 50% text):")
    print("   " + "="*60)
    query = "machine learning algorithms"
    results = vector_store.hybrid_search(
        query,
        k=3,
        vector_weight=0.5,  # 50% vector, 50% text
    )

    print(f"   Query: '{query}'")
    print(f"   Results:\n")
    for i, doc in enumerate(results, 1):
        print(f"   [{i}] {doc.page_content[:80]}...")
        print(f"       Category: {doc.metadata.get('category', 'N/A')}\n")

    # 6. Hybrid search - Vector-heavy (80/20)
    print("6. Hybrid Search (80% vector, 20% text):")
    print("   " + "="*60)
    query = "neural networks deep learning"
    results = vector_store.hybrid_search(
        query,
        k=3,
        vector_weight=0.8,  # 80% vector, 20% text
    )

    print(f"   Query: '{query}'")
    print(f"   Results:\n")
    for i, doc in enumerate(results, 1):
        print(f"   [{i}] {doc.page_content[:80]}...")
        print(f"       Category: {doc.metadata.get('category', 'N/A')}\n")

    # 7. Hybrid search - Text-heavy (20/80)
    print("7. Hybrid Search (20% vector, 80% text):")
    print("   " + "="*60)
    query = "vector database HNSW IVF"
    results = vector_store.hybrid_search(
        query,
        k=3,
        vector_weight=0.2,  # 20% vector, 80% text
    )

    print(f"   Query: '{query}'")
    print(f"   Results:\n")
    for i, doc in enumerate(results, 1):
        print(f"   [{i}] {doc.page_content[:80]}...")
        print(f"       Category: {doc.metadata.get('category', 'N/A')}\n")

    # 8. Hybrid search with filter
    print("8. Hybrid Search with Metadata Filter:")
    print("   " + "="*60)
    query = "artificial intelligence"
    results = vector_store.hybrid_search(
        query,
        k=5,
        vector_weight=0.6,
        filter={"category": "AI"},
    )

    print(f"   Query: '{query}' (filtered by category='AI')")
    print(f"   Results:\n")
    for i, doc in enumerate(results, 1):
        print(f"   [{i}] {doc.page_content[:80]}...")
        print(f"       Source: {doc.metadata.get('source', 'N/A')}\n")

    # 9. Cleanup
    print("9. Cleaning up...")
    if ids:
        vector_store.delete(ids=ids)
        print("   ✓ Deleted documents\n")

    print("=== Example completed successfully! ===")
    print("\n💡 Key Insights:")
    print("   - Hybrid search combines semantic (vector) + lexical (text) matching")
    print("   - Adjust vector_weight to control the balance (0.0-1.0)")
    print("   - Higher vector_weight: more semantic, less keyword-focused")
    print("   - Lower vector_weight: more keyword-focused, less semantic")
    print("   - Use hybrid search for queries with specific terminology\n")


if __name__ == "__main__":
    main()
