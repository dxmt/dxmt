"""Hybrid search example for DXMT Python SDK.

This example demonstrates combining vector similarity search with full-text search.
"""

from dxmt import Client


def main() -> None:
    """Run hybrid search example."""
    # Create client
    print("Connecting to DXMT server...")
    client = Client(host="localhost", port=9999)

    # Create table with full-text search enabled
    print("\nCreating table with full-text search...")
    client.create_table(
        name="articles",
        dimension=128,
        distance_metric="cosine",
        index_type="hnsw",
        enable_full_text_search=True,  # Enable full-text search
    )

    # Insert articles with text content
    print("\nInserting articles...")
    articles = [
        {
            "id": "art1",
            "vector": [0.1] * 128,
            "payload": {  # Note: use 'payload' not 'metadata'
                "title": "Introduction to Machine Learning",
                "content": "Machine learning is a subset of artificial intelligence...",
                "category": "AI",
            },
        },
        {
            "id": "art2",
            "vector": [0.2] * 128,
            "payload": {
                "title": "Deep Learning Fundamentals",
                "content": "Deep learning uses neural networks with multiple layers...",
                "category": "AI",
            },
        },
        {
            "id": "art3",
            "vector": [0.3] * 128,
            "payload": {
                "title": "Natural Language Processing",
                "content": "NLP enables computers to understand human language...",
                "category": "AI",
            },
        },
        {
            "id": "art4",
            "vector": [0.4] * 128,
            "payload": {
                "title": "Computer Vision Applications",
                "content": "Computer vision allows machines to interpret visual data...",
                "category": "CV",
            },
        },
    ]

    client.insert_records(table_name="articles", records=articles)
    print(f"Inserted {len(articles)} articles")

    # Perform hybrid search (vector + text)
    print("\n" + "=" * 60)
    print("HYBRID SEARCH: Combining vector similarity + text matching")
    print("=" * 60)

    query_vector = [0.15] * 128
    text_query = "machine learning"

    # Try different alpha values
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]

    for alpha in alphas:
        print(f"\nAlpha = {alpha} ({'text only' if alpha == 0 else 'vector only' if alpha == 1 else 'balanced'}):")
        print("-" * 60)

        # Note: SDK now supports both k/limit, text/text_query, vector_weight/alpha as aliases
        results = client.hybrid_search(
            table_name="articles",
            vector=query_vector,
            text_query=text_query,  # Can also use: text=text_query
            alpha=alpha,  # Can also use: vector_weight=alpha (0 = text only, 1 = vector only)
            limit=3,  # Can also use: k=3
        )

        for i, result in enumerate(results, 1):
            # Note: Search results may not include payload, fetch full record if needed
            rid = result['id']
            score = result['score']
            print(f"{i}. ID: {rid}")
            print(f"   Score: {score:.4f}")

    # Pure text search
    print("\n" + "=" * 60)
    print("FULL-TEXT SEARCH: Text matching only")
    print("=" * 60)

    results = client.full_text_search(
        table_name="articles", text_query="neural networks", limit=3
    )

    print(f"\nFound {len(results)} results for 'neural networks':")
    for i, result in enumerate(results, 1):
        print(f"{i}. ID: {result['id']}")
        print(f"   Score: {result.get('score', 0):.4f}")

    # Clean up
    print("\nCleaning up...")
    client.delete_table("articles")
    client.close()
    print("Done!")


if __name__ == "__main__":
    main()
