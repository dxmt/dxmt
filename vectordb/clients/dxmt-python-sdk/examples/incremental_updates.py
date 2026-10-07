"""Incremental updates example for DXMT Python SDK.

This example demonstrates:
1. Updating vector embeddings when content changes
2. Updating metadata without changing vectors
3. Incremental indexing strategies
4. Handling deletions and re-insertions

Use case:
    - Content management systems with frequent updates
    - Real-time data pipelines (CDC - Change Data Capture)
    - User profile updates
    - Document versioning
"""

import json
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from dxmt import Client
from environment import client_kwargs, skip


def create_initial_documents() -> List[Dict[str, Any]]:
    """Create initial document corpus."""
    documents = [
        {
            "id": "doc1",
            "content": "Introduction to vector databases and their applications in AI.",
            "metadata": {
                "title": "Vector Database Basics",
                "version": 1,
                "status": "published",
                "views": 100,
                "last_updated": "2024-01-01",
            },
        },
        {
            "id": "doc2",
            "content": "HNSW algorithm provides efficient approximate nearest neighbor search.",
            "metadata": {
                "title": "HNSW Algorithm",
                "version": 1,
                "status": "published",
                "views": 85,
                "last_updated": "2024-01-02",
            },
        },
        {
            "id": "doc3",
            "content": "Semantic search uses embeddings to understand query meaning.",
            "metadata": {
                "title": "Semantic Search",
                "version": 1,
                "status": "draft",
                "views": 50,
                "last_updated": "2024-01-03",
            },
        },
    ]
    return documents


def simulate_content_updates() -> Dict[str, str]:
    """Simulate content updates for documents."""
    return {
        "doc1": "Introduction to vector databases and their applications in artificial intelligence and machine learning.",
        "doc3": "Semantic search uses vector embeddings to understand the true meaning and intent behind search queries.",
    }


def simulate_metadata_updates() -> Dict[str, Dict[str, Any]]:
    """Simulate metadata-only updates with progressive dates."""
    # Use dates that progress forward from current time
    base_date = datetime.now()
    return {
        "doc1": {"views": 150, "last_updated": (base_date + timedelta(days=1)).strftime("%Y-%m-%d")},
        "doc2": {"views": 120, "status": "featured", "last_updated": (base_date + timedelta(days=2)).strftime("%Y-%m-%d")},
        "doc3": {"status": "published", "last_updated": (base_date + timedelta(days=3)).strftime("%Y-%m-%d")},
    }


def main() -> None:
    """Run incremental updates examples."""
    print("=" * 80)
    print("DXMT Incremental Updates Example")
    print("=" * 80)

    # Check dependencies
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        skip("sentence-transformers is not installed: pip install sentence-transformers")

    # Load embedding model
    print("\nLoading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"✅ Model loaded (dimension: {embedding_dim})")

    # Create client
    print("\nConnecting to DXMT server...")
    client = Client(**client_kwargs())

    try:
        # ================================================================
        # Phase 1: Initial Load
        # ================================================================
        print("\n" + "=" * 80)
        print("Phase 1: Initial Data Load")
        print("=" * 80)

        # Create table
        print("\nCreating documents table...")
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

        # Load initial documents
        print("\nLoading initial documents...")
        documents = create_initial_documents()

        # Generate embeddings
        contents = [doc["content"] for doc in documents]
        embeddings = model.encode(contents, show_progress_bar=False, convert_to_numpy=True)

        # Prepare records
        records = []
        for doc, embedding in zip(documents, embeddings):
            records.append({
                "id": doc["id"],
                "vector": embedding.tolist(),
                "payload": {
                    "content": doc["content"],
                    "title": doc["metadata"]["title"],
                    "version": doc["metadata"]["version"],
                    "status": doc["metadata"]["status"],
                    "views": doc["metadata"]["views"],
                    "last_updated": doc["metadata"]["last_updated"],
                },
            })

        # Insert initial records
        client.insert_records(table_name="docs", records=records)
        print(f"✅ Inserted {len(records)} documents")

        # Display initial state
        print("\nInitial documents:")
        all_records = client.get_records(table_name="docs", ids=[d["id"] for d in documents])
        for record in all_records:
            payload = json.loads(record["payload_json"])
            print(f"  - {record['id']}: v{payload['version']}, "
                  f"{payload['status']}, {payload['views']} views")

        time.sleep(1)

        # ================================================================
        # Phase 2: Content Updates (Requires Vector Re-embedding)
        # ================================================================
        print("\n" + "=" * 80)
        print("Phase 2: Content Updates (Vector Re-embedding)")
        print("=" * 80)

        print("\nSimulating content changes...")
        content_updates = simulate_content_updates()

        for doc_id, new_content in content_updates.items():
            # Get current document
            current_records = client.get_records(table_name="docs", ids=[doc_id])
            if not current_records:
                continue

            current_record = current_records[0]
            current_payload = json.loads(current_record["payload_json"])

            print(f"\n{doc_id}:")
            print(f"  Old: {current_payload['content'][:60]}...")
            print(f"  New: {new_content[:60]}...")

            # Generate new embedding
            new_embedding = model.encode(new_content, convert_to_numpy=True)

            # Update using the new update API
            print(f"  Updating vector embedding...")

            # Increment version
            current_payload["content"] = new_content
            current_payload["version"] += 1
            current_payload["last_updated"] = time.strftime("%Y-%m-%d")

            # Update record with new vector and payload
            client.update_record(
                table_name="docs",
                record_id=doc_id,
                vector=new_embedding.tolist(),
                payload=current_payload,
            )
            print(f"  ✅ Updated to version {current_payload['version']}")

        time.sleep(1)

        # ================================================================
        # Phase 3: Metadata-Only Updates (No Vector Change)
        # ================================================================
        print("\n" + "=" * 80)
        print("Phase 3: Metadata-Only Updates (No Vector Change)")
        print("=" * 80)

        print("\nSimulating metadata updates...")
        metadata_updates = simulate_metadata_updates()

        for doc_id, updates in metadata_updates.items():
            # Get current document
            current_records = client.get_records(table_name="docs", ids=[doc_id])
            if not current_records:
                continue

            current_record = current_records[0]
            current_payload = json.loads(current_record["payload_json"])

            print(f"\n{doc_id}:")
            for key, new_value in updates.items():
                old_value = current_payload.get(key)
                print(f"  {key}: {old_value} → {new_value}")

            # Update metadata (keep same vector)
            # Update payload
            current_payload.update(updates)

            # Update only metadata, vector stays the same
            client.update_record(
                table_name="docs",
                record_id=doc_id,
                payload=current_payload,
            )
            print(f"  ✅ Metadata updated (vector unchanged)")

        time.sleep(1)

        # ================================================================
        # Phase 4: Verify Updates with Search
        # ================================================================
        print("\n" + "=" * 80)
        print("Phase 4: Verify Updates with Search")
        print("=" * 80)

        # Search for updated content
        query = "artificial intelligence and machine learning"
        query_embedding = model.encode(query, convert_to_numpy=True)

        print(f"\nQuery: \"{query}\"")
        print("Expected: doc1 should rank higher due to updated content")

        results = client.search(
            table_name="docs",
            vector=query_embedding.tolist(),
            k=3,
        )

        print("\nSearch results:")
        for i, result in enumerate(results, 1):
            record_ids = [result["id"]]
            full_records = client.get_records(table_name="docs", ids=record_ids)
            if full_records:
                payload = json.loads(full_records[0]["payload_json"])
                print(f"{i}. {result['id']} (Score: {result['score']:.4f})")
                print(f"   Version: {payload['version']}, Status: {payload['status']}")
                print(f"   Content: {payload['content'][:60]}...")

        # ================================================================
        # Phase 5: Batch Updates
        # ================================================================
        print("\n" + "=" * 80)
        print("Phase 5: Batch Updates (Simulating CDC Pipeline)")
        print("=" * 80)

        print("\nSimulating batch of changes from CDC stream...")

        # Simulate a batch of changes
        batch_changes = [
            {"id": "doc4", "action": "insert", "content": "New document about vector indexing strategies."},
            {"id": "doc2", "action": "update", "content": "HNSW algorithm provides efficient and accurate approximate nearest neighbor search."},
            {"id": "doc1", "action": "delete"},
        ]

        for change in batch_changes:
            doc_id = change["id"]
            action = change["action"]

            print(f"\n{doc_id}: {action}")

            if action == "insert":
                # Insert new document
                content = change["content"]
                embedding = model.encode(content, convert_to_numpy=True)

                client.insert_records(
                    table_name="docs",
                    records=[{
                        "id": doc_id,
                        "vector": embedding.tolist(),
                        "payload": {
                            "content": content,
                            "title": f"Document {doc_id}",
                            "version": 1,
                            "status": "published",
                            "views": 0,
                            "last_updated": time.strftime("%Y-%m-%d"),
                        },
                    }]
                )
                print(f"  ✅ Inserted")

            elif action == "update":
                # Update existing document
                content = change["content"]
                embedding = model.encode(content, convert_to_numpy=True)

                # Get current metadata
                current_records = client.get_records(table_name="docs", ids=[doc_id])
                if current_records:
                    current_payload = json.loads(current_records[0]["payload_json"])

                    current_payload["content"] = content
                    current_payload["version"] += 1
                    current_payload["last_updated"] = time.strftime("%Y-%m-%d")

                    # Use update API
                    client.update_record(
                        table_name="docs",
                        record_id=doc_id,
                        vector=embedding.tolist(),
                        payload=current_payload,
                    )
                    print(f"  ✅ Updated")

            elif action == "delete":
                # Delete document
                client.delete_records(table_name="docs", ids=[doc_id])
                print(f"  ✅ Deleted")

        # Display final state
        print("\n" + "=" * 80)
        print("Final State")
        print("=" * 80)

        remaining_ids = ["doc2", "doc3", "doc4"]
        final_records = client.get_records(table_name="docs", ids=remaining_ids)

        print(f"\nRemaining documents: {len(final_records)}")
        for record in final_records:
            payload = json.loads(record["payload_json"])
            print(f"  - {record['id']}: v{payload['version']}, "
                  f"{payload['status']}, {payload['views']} views")
            print(f"    Content: {payload['content'][:50]}...")

        # ================================================================
        # Summary
        # ================================================================
        print("\n" + "=" * 80)
        print("Incremental Updates Best Practices")
        print("=" * 80)
        print("""
Update Strategies:

1. Content Updates (Vector Changes):
   - Re-generate embedding from new content
   - Use update_record() with new vector and payload
   - Track version number for conflict resolution

2. Metadata-Only Updates:
   - Keep existing vector
   - Use update_record() with only payload parameter
   - Much faster than re-embedding
   - Use for views, status, tags, etc.

3. Batch Updates (CDC Pattern):
   - Process changes in batches for efficiency
   - Group similar operations (inserts, updates, deletes)
   - Use transactions if available
   - Handle errors gracefully

Performance Tips:
  - Cache embeddings to detect content changes
  - Use metadata timestamps for incremental processing
  - Batch multiple updates together
  - Consider update frequency vs. search accuracy tradeoff
  - Use background jobs for large updates

Error Handling:
  - Retry failed updates
  - Log all changes for audit trail
  - Implement rollback mechanism
  - Validate data before updating
  - Monitor update lag

CDC Integration:
  - Listen to database change events
  - Buffer changes for batch processing
  - Handle out-of-order updates
  - Deduplicate rapid updates to same document
  - Use watermarks for progress tracking

Real-World Applications:
  - Content Management: Article updates
  - E-commerce: Product catalog updates
  - Social Media: User profile updates
  - Real-time Analytics: Incremental data ingestion
  - Document Management: Version control

Note: DXMT now has a native update API via client.update_record(),
which is more efficient than the delete + re-insert pattern,
especially for metadata-only updates.
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
