"""Test to verify why hybrid search text component returns 0 score."""

from dxmt import Client


def main() -> None:
    """Test text search in hybrid mode."""
    client = Client(host="localhost", port=9999)

    try:
        # Create table
        print("Creating test table...")
        try:
            client.delete_table("test_hybrid")
        except Exception:
            pass

        client.create_table(
            name="test_hybrid",
            dimension=128,
            distance_metric="cosine",
            index_type="hnsw",
            enable_full_text_search=True,
        )

        # Insert test articles
        print("\nInserting test articles...")
        articles = [
            {
                "id": "art1",
                "vector": [0.1] * 128,
                "payload": {
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
        ]

        client.insert_records(table_name="test_hybrid", records=articles)
        print("Inserted articles")

        # Test 1: Direct full-text search for "machine learning"
        print("\n" + "=" * 60)
        print("TEST 1: Full-text search for 'machine learning'")
        print("=" * 60)

        try:
            results = client.full_text_search(
                table_name="test_hybrid",
                text_query="machine learning",
                limit=10,
            )

            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"{i}. ID: {result['id']}, Score: {result.get('score', 0):.4f}")

            if len(results) == 0:
                print("⚠️  WARNING: Full-text search returned 0 results!")
                print("   This means the query doesn't match any documents.")
        except Exception as e:
            print(f"❌ Full-text search failed: {e}")

        # Test 2: Hybrid search with alpha=0.0 (text only)
        print("\n" + "=" * 60)
        print("TEST 2: Hybrid search with alpha=0.0 (text only)")
        print("=" * 60)

        query_vector = [0.15] * 128

        try:
            results = client.hybrid_search(
                table_name="test_hybrid",
                vector=query_vector,
                text_query="machine learning",
                alpha=0.0,  # Text only
                limit=10,
            )

            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"{i}. ID: {result['id']}, Score: {result.get('score', 0):.4f}")

            if all(r.get('score', 0) == 0 for r in results):
                print("⚠️  WARNING: All scores are 0!")
                print("   This indicates text search component is not working in hybrid mode.")
        except Exception as e:
            print(f"❌ Hybrid search failed: {e}")

        # Test 3: Search for exact match
        print("\n" + "=" * 60)
        print("TEST 3: Full-text search for exact phrase in content")
        print("=" * 60)

        try:
            results = client.full_text_search(
                table_name="test_hybrid",
                text_query="neural networks",
                limit=10,
            )

            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"{i}. ID: {result['id']}, Score: {result.get('score', 0):.4f}")
        except Exception as e:
            print(f"❌ Full-text search failed: {e}")

        # Test 4: Search for single word
        print("\n" + "=" * 60)
        print("TEST 4: Full-text search for single word 'learning'")
        print("=" * 60)

        try:
            results = client.full_text_search(
                table_name="test_hybrid",
                text_query="learning",
                limit=10,
            )

            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"{i}. ID: {result['id']}, Score: {result.get('score', 0):.4f}")
        except Exception as e:
            print(f"❌ Full-text search failed: {e}")

    finally:
        # Cleanup
        print("\nCleaning up...")
        try:
            client.delete_table("test_hybrid")
        except Exception:
            pass
        client.close()
        print("Done!")


if __name__ == "__main__":
    main()
