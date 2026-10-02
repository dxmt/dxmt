"""Duplicate detection example for DXMT Python SDK.

This example demonstrates:
1. Using vector similarity to detect duplicate or near-duplicate content
2. Finding similar documents based on similarity threshold
3. Deduplication strategies for data cleaning
4. Identifying potential plagiarism or copied content

Prerequisites:
    pip install sentence-transformers

Use case:
    - Remove duplicate articles or posts
    - Detect plagiarism in submissions
    - Find similar support tickets
    - Clean up redundant database entries
"""

import json
from typing import List, Dict, Any, Tuple

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from dxmt import Client


def create_articles_with_duplicates() -> List[Dict[str, Any]]:
    """Create a corpus with some duplicate and near-duplicate articles."""
    articles = [
        # Original articles
        {
            "id": "art1",
            "text": "Vector databases are optimized for storing and searching high-dimensional embeddings.",
            "metadata": {"author": "Alice", "date": "2024-01-01", "type": "original"},
        },
        {
            "id": "art2",
            "text": "Machine learning models can be used to generate embeddings from text data.",
            "metadata": {"author": "Bob", "date": "2024-01-02", "type": "original"},
        },
        {
            "id": "art3",
            "text": "Semantic search helps find relevant information based on meaning rather than keywords.",
            "metadata": {"author": "Carol", "date": "2024-01-03", "type": "original"},
        },

        # Near-duplicates (paraphrased)
        {
            "id": "art4",
            "text": "Databases designed for vectors excel at storing and querying high-dimensional embeddings.",
            "metadata": {"author": "Dave", "date": "2024-01-04", "type": "near-duplicate"},
        },
        {
            "id": "art5",
            "text": "Embeddings from text can be generated using machine learning algorithms.",
            "metadata": {"author": "Eve", "date": "2024-01-05", "type": "near-duplicate"},
        },

        # Exact duplicates (same text, different metadata)
        {
            "id": "art6",
            "text": "Vector databases are optimized for storing and searching high-dimensional embeddings.",
            "metadata": {"author": "Frank", "date": "2024-01-06", "type": "exact-duplicate"},
        },

        # Similar but different
        {
            "id": "art7",
            "text": "Full-text search engines use inverted indices for fast keyword matching.",
            "metadata": {"author": "Grace", "date": "2024-01-07", "type": "original"},
        },
        {
            "id": "art8",
            "text": "Graph databases are great for storing relationships between entities.",
            "metadata": {"author": "Henry", "date": "2024-01-08", "type": "original"},
        },

        # Another near-duplicate
        {
            "id": "art9",
            "text": "Semantic search finds relevant content by analyzing meaning instead of just keywords.",
            "metadata": {"author": "Ivy", "date": "2024-01-09", "type": "near-duplicate"},
        },

        # Unique article
        {
            "id": "art10",
            "text": "Time-series databases are optimized for handling sequential data with timestamps.",
            "metadata": {"author": "Jack", "date": "2024-01-10", "type": "original"},
        },
    ]
    return articles


def find_duplicates(
    client: Client,
    table_name: str,
    records_map: Dict[str, Any],
    similarity_threshold: float = 0.9
) -> List[Tuple[str, str, float]]:
    """Find duplicate pairs based on similarity threshold.

    Returns:
        List of (id1, id2, similarity) tuples
    """
    duplicates = []
    checked_pairs = set()

    # For each document, search for similar ones
    for doc_id, record in records_map.items():
        # Get vector
        vector = record["vector"]

        # Search for similar documents
        results = client.search(
            table_name=table_name,
            vector=vector,
            k=5,  # Top 5 similar
        )

        # Check each result
        for result in results:
            other_id = result["id"]
            similarity = result["score"]

            # Skip self-match
            if other_id == doc_id:
                continue

            # Check threshold
            if similarity >= similarity_threshold:
                # Create sorted pair to avoid duplicates
                pair = tuple(sorted([doc_id, other_id]))
                if pair not in checked_pairs:
                    checked_pairs.add(pair)
                    duplicates.append((doc_id, other_id, similarity))

    return duplicates


def main() -> None:
    """Run duplicate detection examples."""
    print("=" * 80)
    print("DXMT Duplicate Detection Example")
    print("=" * 80)

    # Check dependencies
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\n❌ Error: sentence-transformers not installed")
        print("   Install with: pip install sentence-transformers")
        return

    # Load embedding model
    print("\nLoading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"✅ Model loaded (dimension: {embedding_dim})")

    # Create client
    print("\nConnecting to DXMT server...")
    client = Client(host="localhost", port=9999)

    try:
        # Create table
        print("\nCreating articles table...")
        try:
            client.delete_table("articles")
        except Exception:
            pass

        client.create_table(
            name="articles",
            dimension=embedding_dim,
            distance_metric="cosine",
            index_type="hnsw",
        )
        print("✅ Table created")

        # Load articles with duplicates
        print("\nLoading articles (including some duplicates)...")
        articles = create_articles_with_duplicates()

        # Generate embeddings
        texts = [article["text"] for article in articles]
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Prepare records
        records = []
        for article, embedding in zip(articles, embeddings):
            records.append({
                "id": article["id"],
                "vector": embedding.tolist(),
                "payload": {
                    "content": article["text"],
                    "author": article["metadata"]["author"],
                    "date": article["metadata"]["date"],
                    "type": article["metadata"]["type"],
                },
            })

        # Insert records
        print("\nInserting articles into DXMT...")
        client.insert_records(table_name="articles", records=records)
        print(f"✅ Inserted {len(records)} articles")

        # Get all records
        all_records = client.get_records(table_name="articles", ids=[a["id"] for a in articles])
        records_map = {record["id"]: record for record in all_records}

        # ================================================================
        # Example 1: Find Exact and Near Duplicates
        # ================================================================
        print("\n" + "=" * 80)
        print("Example 1: Detecting Duplicates (Threshold: 0.9)")
        print("=" * 80)

        duplicates = find_duplicates(client, "articles", records_map, similarity_threshold=0.9)

        print(f"\nFound {len(duplicates)} duplicate pairs:\n")

        for id1, id2, similarity in duplicates:
            record1 = records_map[id1]
            record2 = records_map[id2]

            payload1 = json.loads(record1["payload_json"])
            payload2 = json.loads(record2["payload_json"])

            print(f"{'─' * 80}")
            print(f"Similarity: {similarity:.4f}")
            print(f"\n1. {id1} (by {payload1['author']}, {payload1['type']})")
            print(f"   \"{payload1['content']}\"")
            print(f"\n2. {id2} (by {payload2['author']}, {payload2['type']})")
            print(f"   \"{payload2['content']}\"")
            print()

        # ================================================================
        # Example 2: Lower Threshold for Similar Content
        # ================================================================
        print("\n" + "=" * 80)
        print("Example 2: Finding Similar Content (Threshold: 0.7)")
        print("=" * 80)

        similar_pairs = find_duplicates(client, "articles", records_map, similarity_threshold=0.7)

        print(f"\nFound {len(similar_pairs)} similar pairs:\n")

        # Show just the count by similarity range
        ranges = {
            "Exact (>0.95)": 0,
            "Near-duplicate (0.9-0.95)": 0,
            "Similar (0.7-0.9)": 0,
        }

        for _, _, similarity in similar_pairs:
            if similarity > 0.95:
                ranges["Exact (>0.95)"] += 1
            elif similarity >= 0.9:
                ranges["Near-duplicate (0.9-0.95)"] += 1
            else:
                ranges["Similar (0.7-0.9)"] += 1

        for range_name, count in ranges.items():
            print(f"  {range_name}: {count} pairs")

        # ================================================================
        # Example 3: Check a New Document for Duplicates
        # ================================================================
        print("\n" + "=" * 80)
        print("Example 3: Check New Document Before Insertion")
        print("=" * 80)

        new_document = "Vector databases excel at storing and searching embeddings."
        print(f"\nNew document to check:")
        print(f'  "{new_document}"')

        # Generate embedding
        new_embedding = model.encode(new_document, convert_to_numpy=True)

        # Search for similar documents
        results = client.search(
            table_name="articles",
            vector=new_embedding.tolist(),
            k=3,
        )

        print(f"\nMost similar existing documents:")
        for i, result in enumerate(results, 1):
            record = records_map[result["id"]]
            payload = json.loads(record["payload_json"])
            similarity = result["score"]

            print(f"\n{i}. {result['id']} (Similarity: {similarity:.4f})")
            print(f'   "{payload["content"]}"')
            print(f"   Author: {payload['author']}, Type: {payload['type']}")

            if similarity > 0.9:
                print("   ⚠️  WARNING: Likely duplicate! Consider rejecting.")
            elif similarity > 0.7:
                print("   ⚠️  WARNING: Very similar content exists.")
            else:
                print("   ✅ OK: Sufficiently different.")

        # ================================================================
        # Example 4: Deduplication Strategy
        # ================================================================
        print("\n" + "=" * 80)
        print("Example 4: Deduplication Strategy")
        print("=" * 80)

        print("\nStrategy: Keep earliest article, remove duplicates")

        # Find all duplicates
        all_duplicates = find_duplicates(client, "articles", records_map, similarity_threshold=0.9)

        # Group duplicates
        to_remove = set()
        for id1, id2, _ in all_duplicates:
            record1 = records_map[id1]
            record2 = records_map[id2]

            payload1 = json.loads(record1["payload_json"])
            payload2 = json.loads(record2["payload_json"])

            # Keep the one with earlier date
            if payload1["date"] < payload2["date"]:
                to_remove.add(id2)
            else:
                to_remove.add(id1)

        print(f"\nDocuments to remove: {len(to_remove)}")
        for doc_id in sorted(to_remove):
            record = records_map[doc_id]
            payload = json.loads(record["payload_json"])
            print(f"  - {doc_id}: {payload['type']} by {payload['author']}")

        # Actually remove them
        if to_remove:
            print(f"\nRemoving {len(to_remove)} duplicate documents...")
            client.delete_records(table_name="articles", ids=list(to_remove))
            print("✅ Duplicates removed")

            # Verify
            remaining = client.get_records(
                table_name="articles",
                ids=[a["id"] for a in articles if a["id"] not in to_remove]
            )
            print(f"\nRemaining documents: {len(remaining)}")
            for record in remaining:
                payload = json.loads(record["payload_json"])
                print(f"  - {record['id']}: {payload['type']}")

        # ================================================================
        # Summary
        # ================================================================
        print("\n" + "=" * 80)
        print("Duplicate Detection Best Practices")
        print("=" * 80)
        print("""
Similarity Thresholds:
  - > 0.95: Exact or nearly exact duplicates
  - 0.90-0.95: Near-duplicates (paraphrased)
  - 0.70-0.90: Similar content
  - < 0.70: Different content

Deduplication Strategies:
  1. Keep earliest (by timestamp)
  2. Keep from trusted source
  3. Keep highest quality (by ratings/votes)
  4. Merge metadata from duplicates

Real-World Applications:
  - Content moderation (remove spam duplicates)
  - Data cleaning (deduplicate scraped data)
  - Plagiarism detection (find copied content)
  - Support ticket routing (find similar issues)
  - Product catalog cleanup (merge duplicate listings)

Performance Tips:
  - Use lower k value (3-5) for faster checks
  - Batch check multiple documents together
  - Cache embeddings to avoid recomputation
  - Use metadata filters to reduce search space

Integration Pattern:
  1. Generate embedding for new document
  2. Search for similar documents
  3. If similarity > threshold, flag or reject
  4. Otherwise, insert the document
        """)

    finally:
        # Cleanup
        print("\nCleaning up...")
        try:
            client.delete_table("articles")
            print("✅ Deleted table 'articles'")
        except Exception as e:
            print(f"⚠️  Cleanup failed: {e}")

        client.close()
        print("\n✅ Done!")


if __name__ == "__main__":
    main()
