"""Semantic search example with real embeddings for DXMT Python SDK.

This example demonstrates:
1. Using sentence-transformers for text embeddings
2. Semantic search (finding similar meaning, not just keywords)
3. Comparing semantic search with keyword matching
4. Real-world use case: technical documentation search

Prerequisites:
    pip install sentence-transformers

Use case:
    Search technical documentation using natural language queries
    to find semantically similar content, even with different wording.
"""

import json
from typing import List, Dict, Any

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from dxmt import Client


def create_documentation_corpus() -> List[Dict[str, Any]]:
    """Create a corpus of technical documentation snippets."""
    documents = [
        {
            "id": "doc1",
            "text": "Vector databases store high-dimensional embeddings for semantic similarity search.",
            "metadata": {
                "title": "Introduction to Vector Databases",
                "category": "fundamentals",
                "difficulty": "beginner",
            },
        },
        {
            "id": "doc2",
            "text": "HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for approximate nearest neighbor search.",
            "metadata": {
                "title": "HNSW Algorithm Overview",
                "category": "algorithms",
                "difficulty": "advanced",
            },
        },
        {
            "id": "doc3",
            "text": "Embeddings are numerical representations of text that capture semantic meaning.",
            "metadata": {
                "title": "Understanding Embeddings",
                "category": "fundamentals",
                "difficulty": "beginner",
            },
        },
        {
            "id": "doc4",
            "text": "Cosine similarity measures the angle between two vectors, commonly used for text similarity.",
            "metadata": {
                "title": "Distance Metrics Explained",
                "category": "algorithms",
                "difficulty": "intermediate",
            },
        },
        {
            "id": "doc5",
            "text": "Product Quantization (PQ) compresses vectors to reduce memory usage while maintaining search quality.",
            "metadata": {
                "title": "Vector Compression Techniques",
                "category": "optimization",
                "difficulty": "advanced",
            },
        },
        {
            "id": "doc6",
            "text": "Full-text search uses inverted indices to find documents containing specific keywords.",
            "metadata": {
                "title": "Full-Text Search Basics",
                "category": "fundamentals",
                "difficulty": "beginner",
            },
        },
        {
            "id": "doc7",
            "text": "Hybrid search combines vector similarity and keyword matching for improved retrieval accuracy.",
            "metadata": {
                "title": "Hybrid Search Strategies",
                "category": "fundamentals",
                "difficulty": "intermediate",
            },
        },
        {
            "id": "doc8",
            "text": "DiskANN enables billion-scale vector search by using SSD storage efficiently.",
            "metadata": {
                "title": "Scaling Vector Search",
                "category": "optimization",
                "difficulty": "advanced",
            },
        },
        {
            "id": "doc9",
            "text": "Batch processing can improve throughput by grouping multiple operations together.",
            "metadata": {
                "title": "Performance Optimization",
                "category": "optimization",
                "difficulty": "intermediate",
            },
        },
        {
            "id": "doc10",
            "text": "Metadata filtering allows you to combine semantic search with structured queries.",
            "metadata": {
                "title": "Filtered Vector Search",
                "category": "fundamentals",
                "difficulty": "intermediate",
            },
        },
    ]
    return documents


def print_search_results(query: str, results: List[Dict[str, Any]], records_map: Dict[str, Any]) -> None:
    """Pretty print search results."""
    print(f"\n{'─' * 80}")
    print(f"Query: \"{query}\"")
    print(f"{'─' * 80}")
    print(f"Found {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        doc_id = result["id"]
        score = result["score"]

        # Get full document
        doc = records_map.get(doc_id)
        if not doc:
            continue

        # Parse payload
        payload = {}
        if "payload_json" in doc and doc["payload_json"] and doc["payload_json"] != "null":
            payload = json.loads(doc["payload_json"])

        print(f"{i}. {payload.get('title', doc_id)} (Score: {score:.4f})")
        print(f"   Content: {payload.get('content', 'N/A')}")
        print(f"   Category: {payload.get('category', 'N/A')} | "
              f"Difficulty: {payload.get('difficulty', 'N/A')}")


def main() -> None:
    """Run semantic search examples."""
    print("=" * 80)
    print("DXMT Semantic Search with Real Embeddings")
    print("=" * 80)

    # Check dependencies
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\n❌ Error: sentence-transformers not installed")
        print("   Install with: pip install sentence-transformers")
        return

    # Load embedding model
    print("\nLoading embedding model (all-MiniLM-L6-v2)...")
    print("   This may take a moment on first run...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"✅ Model loaded (dimension: {embedding_dim})")

    # Create client
    print("\nConnecting to DXMT server...")
    client = Client(host="localhost", port=9999)

    try:
        # Create table
        print("\nCreating documentation table...")
        try:
            client.delete_table("docs")
        except Exception:
            pass

        client.create_table(
            name="docs",
            dimension=embedding_dim,
            distance_metric="cosine",
            index_type="hnsw",
        )
        print("✅ Table created")

        # Load documents and generate embeddings
        print("\nGenerating embeddings for documents...")
        documents = create_documentation_corpus()

        # Generate embeddings
        texts = [doc["text"] for doc in documents]
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        print(f"✅ Generated {len(embeddings)} embeddings")

        # Prepare records for insertion
        records = []
        for doc, embedding in zip(documents, embeddings):
            records.append({
                "id": doc["id"],
                "vector": embedding.tolist(),
                "payload": {
                    "content": doc["text"],
                    "title": doc["metadata"]["title"],
                    "category": doc["metadata"]["category"],
                    "difficulty": doc["metadata"]["difficulty"],
                },
            })

        # Insert records
        print("\nInserting documents into DXMT...")
        client.insert_records(table_name="docs", records=records, mode="batch")
        print(f"✅ Inserted {len(records)} documents")

        # Get all records for display (create a map)
        all_records = client.get_records(table_name="docs", ids=[doc["id"] for doc in documents])
        records_map = {record["id"]: record for record in all_records}

        # ================================================================
        # Example Searches
        # ================================================================
        print("\n" + "=" * 80)
        print("Semantic Search Examples")
        print("=" * 80)

        queries = [
            "How do I find similar items using vectors?",
            "What are graph-based nearest neighbor methods?",
            "How can I reduce memory consumption of vectors?",
            "Combining keyword search with semantic search",
            "Improving search performance with batching",
        ]

        for query in queries:
            # Generate query embedding
            query_embedding = model.encode(query, convert_to_numpy=True)

            # Search
            results = client.search(
                table_name="docs",
                vector=query_embedding.tolist(),
                k=3,
            )

            # Display results
            print_search_results(query, results, records_map)

        # ================================================================
        # Comparison: Semantic vs Keyword
        # ================================================================
        print("\n" + "=" * 80)
        print("Semantic Search vs Keyword Matching Comparison")
        print("=" * 80)

        comparison_query = "How do I search for similar documents?"

        print(f"\nQuery: \"{comparison_query}\"")
        print("\nSemantic search finds documents with SIMILAR MEANING:")
        print("  Expected: Documents about vector search, similarity, embeddings")
        print("\nKeyword search would find documents CONTAINING THESE WORDS:")
        print("  Expected: Documents with 'search', 'similar', 'documents'")

        # Generate query embedding
        query_embedding = model.encode(comparison_query, convert_to_numpy=True)

        # Semantic search
        results = client.search(
            table_name="docs",
            vector=query_embedding.tolist(),
            k=3,
        )

        print(f"\n{'─' * 80}")
        print("Semantic Search Results:")
        print(f"{'─' * 80}")
        for i, result in enumerate(results, 1):
            doc = records_map.get(result["id"])
            if doc and "payload_json" in doc:
                payload = json.loads(doc["payload_json"])
                print(f"{i}. {payload.get('title')} (Score: {result['score']:.4f})")
                print(f"   {payload.get('content')}")

        # ================================================================
        # Filtered Semantic Search
        # ================================================================
        print("\n" + "=" * 80)
        print("Filtered Semantic Search (Beginner-Level Only)")
        print("=" * 80)

        query = "What is semantic search?"
        query_embedding = model.encode(query, convert_to_numpy=True)

        results = client.search(
            table_name="docs",
            vector=query_embedding.tolist(),
            k=5,
            filter={"difficulty": "beginner"},
        )

        print_search_results(query, results, records_map)

        # ================================================================
        # Summary
        # ================================================================
        print("\n" + "=" * 80)
        print("Key Insights")
        print("=" * 80)
        print("""
Semantic Search Advantages:
  ✓ Finds content with similar MEANING, not just matching keywords
  ✓ Works with synonyms and paraphrases
  ✓ Better user experience for natural language queries
  ✓ Handles typos and variations in wording

When to use Semantic Search:
  - FAQ and documentation search
  - Customer support ticket matching
  - Content recommendation systems
  - Research paper similarity
  - Code snippet search

Embedding Models:
  - all-MiniLM-L6-v2: Fast, good for general text (384 dims)
  - all-mpnet-base-v2: Higher quality, slower (768 dims)
  - multilingual models: Support multiple languages
  - domain-specific models: Better for specialized content

Best Practices:
  1. Choose embedding model based on your use case
  2. Use consistent model for all documents and queries
  3. Combine with metadata filters for better results
  4. Consider hybrid search (semantic + keyword) for critical applications
  5. Cache embeddings to avoid recomputation
        """)

    finally:
        # Cleanup
        print("\nCleaning up...")
        try:
            client.delete_table("docs")
            print("✅ Deleted table 'docs'")
        except Exception as e:
            print(f"⚠️  Cleanup failed: {e}")

        client.close()
        print("\n✅ Done!")


if __name__ == "__main__":
    main()
