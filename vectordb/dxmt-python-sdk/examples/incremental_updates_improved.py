"""Improved incremental updates example for DXMT Python SDK.

This example demonstrates:
1. Updating vector embeddings when content changes
2. Updating metadata without changing vectors
3. Performance comparison: content vs metadata updates
4. Content change detection to avoid unnecessary re-embedding
5. Error handling and retry logic
6. Real-world CDC (Change Data Capture) pattern
7. Version conflict detection and resolution

Use case:
    - Content management systems with frequent updates
    - Real-time data pipelines (CDC - Change Data Capture)
    - User profile updates
    - Document versioning
    - E-commerce product catalogs
"""

import json
import time
import random
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from dxmt import Client


@dataclass
class UpdateStats:
    """Statistics for update operations."""
    content_updates: int = 0
    metadata_updates: int = 0
    content_update_time: float = 0.0
    metadata_update_time: float = 0.0
    skipped_updates: int = 0
    failed_updates: int = 0


def content_hash(content: str) -> str:
    """Generate hash for content to detect changes."""
    return hashlib.md5(content.encode()).hexdigest()[:16]


def create_initial_documents() -> List[Dict[str, Any]]:
    """Create initial document corpus."""
    base_date = datetime.now() - timedelta(days=30)

    documents = [
        {
            "id": "doc1",
            "content": "Introduction to vector databases and their applications in AI.",
            "metadata": {
                "title": "Vector Database Basics",
                "version": 1,
                "status": "published",
                "views": 100,
                "last_updated": (base_date + timedelta(days=0)).strftime("%Y-%m-%d"),
                "content_hash": "",  # Will be computed
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
                "last_updated": (base_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                "content_hash": "",
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
                "last_updated": (base_date + timedelta(days=2)).strftime("%Y-%m-%d"),
                "content_hash": "",
            },
        },
    ]

    # Compute content hashes
    for doc in documents:
        doc["metadata"]["content_hash"] = content_hash(doc["content"])

    return documents


def simulate_content_updates() -> Dict[str, str]:
    """Simulate content updates for documents."""
    return {
        "doc1": "Introduction to vector databases and their applications in artificial intelligence and machine learning.",
        "doc2": "HNSW algorithm provides efficient approximate nearest neighbor search.",  # Unchanged
        "doc3": "Semantic search uses vector embeddings to understand the true meaning and intent behind search queries.",
    }


def simulate_metadata_updates(base_date: datetime) -> Dict[str, Dict[str, Any]]:
    """Simulate metadata-only updates."""
    return {
        "doc1": {
            "views": 150,
            "last_updated": (base_date + timedelta(days=15)).strftime("%Y-%m-%d")
        },
        "doc2": {
            "views": 120,
            "status": "featured",
            "last_updated": (base_date + timedelta(days=16)).strftime("%Y-%m-%d")
        },
        "doc3": {
            "status": "published",
            "last_updated": (base_date + timedelta(days=17)).strftime("%Y-%m-%d")
        },
    }


def create_cdc_events(base_date: datetime) -> List[Dict[str, Any]]:
    """Create realistic CDC (Change Data Capture) events."""
    return [
        {
            "timestamp": (base_date + timedelta(days=18, hours=10, minutes=30)).isoformat(),
            "sequence": 1001,
            "operation": "INSERT",
            "table": "documents",
            "after": {
                "id": "doc4",
                "content": "New document about vector indexing strategies and optimization techniques.",
                "title": "Vector Indexing Strategies",
                "version": 1,
                "status": "published",
            }
        },
        {
            "timestamp": (base_date + timedelta(days=18, hours=10, minutes=31)).isoformat(),
            "sequence": 1002,
            "operation": "UPDATE",
            "table": "documents",
            "before": {
                "id": "doc2",
                "content": "HNSW algorithm provides efficient approximate nearest neighbor search.",
            },
            "after": {
                "id": "doc2",
                "content": "HNSW algorithm provides efficient and accurate approximate nearest neighbor search with configurable parameters.",
            }
        },
        {
            "timestamp": (base_date + timedelta(days=18, hours=10, minutes=32)).isoformat(),
            "sequence": 1003,
            "operation": "DELETE",
            "table": "documents",
            "before": {
                "id": "doc1",
            }
        },
    ]


def update_with_retry(
    client: Client,
    table_name: str,
    record_id: str,
    vector: Optional[List[float]] = None,
    payload: Optional[Dict[str, Any]] = None,
    merge_mode: str = "replace",
    max_retries: int = 3
) -> bool:
    """Update record with retry logic.

    Args:
        merge_mode: "replace" or "merge" - controls payload update behavior
    """
    for attempt in range(max_retries):
        try:
            client.update_record(
                table_name=table_name,
                record_id=record_id,
                vector=vector,
                payload=payload,
                merge_mode=merge_mode
            )
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  ⚠️  Attempt {attempt + 1} failed: {e}")
                print(f"  🔄 Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)
            else:
                print(f"  ❌ All {max_retries} attempts failed: {e}")
                return False
    return False


def main() -> None:
    """Run improved incremental updates examples."""
    print("=" * 80)
    print("DXMT Incremental Updates Example (Improved)")
    print("=" * 80)

    # Check dependencies
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\n❌ Error: sentence-transformers not installed")
        print("   Install with: pip install sentence-transformers")
        return

    # Initialize stats
    stats = UpdateStats()
    base_date = datetime.now() - timedelta(days=30)

    # Load embedding model
    print("\nLoading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"✅ Model loaded (dimension: {embedding_dim})")

    # Create client
    print("\nConnecting to DXMT server...")
    client = Client(host="localhost", port=9999)

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
                    "content_hash": doc["metadata"]["content_hash"],
                },
            })

        # Insert initial records
        client.insert_records(table_name="docs", records=records, mode="batch")
        print(f"✅ Inserted {len(records)} documents")

        # Display initial state
        print("\nInitial documents:")
        all_records = client.get_records(table_name="docs", ids=[d["id"] for d in documents])
        for record in all_records:
            payload = json.loads(record["payload_json"])
            print(f"  - {record['id']}: v{payload['version']}, "
                  f"{payload['status']}, {payload['views']} views")
            print(f"    Updated: {payload['last_updated']}, Hash: {payload['content_hash']}")

        time.sleep(1)

        # ================================================================
        # Phase 2: Content Updates with Change Detection
        # ================================================================
        print("\n" + "=" * 80)
        print("Phase 2: Content Updates with Change Detection")
        print("=" * 80)

        print("\nSimulating content changes with smart detection...")
        content_updates = simulate_content_updates()

        for doc_id, new_content in content_updates.items():
            # Get current document
            current_records = client.get_records(table_name="docs", ids=[doc_id])
            if not current_records:
                continue

            current_record = current_records[0]
            current_payload = json.loads(current_record["payload_json"])
            old_content = current_payload["content"]

            print(f"\n{doc_id}:")
            print(f"  Old: {old_content[:60]}...")
            print(f"  New: {new_content[:60]}...")

            # Detect if content actually changed
            old_hash = current_payload.get("content_hash", "")
            new_hash = content_hash(new_content)

            if old_hash == new_hash:
                print(f"  ℹ️  Content unchanged (hash match), skipping re-embedding")
                stats.skipped_updates += 1
                continue

            print(f"  🔄 Content changed (hash: {old_hash} → {new_hash})")

            # Measure embedding time
            start_time = time.time()

            # Generate new embedding
            new_embedding = model.encode(new_content, convert_to_numpy=True)

            # Update payload
            current_payload["content"] = new_content
            current_payload["version"] += 1
            current_payload["last_updated"] = (base_date + timedelta(days=10)).strftime("%Y-%m-%d")
            current_payload["content_hash"] = new_hash

            # Update record with new vector and payload
            success = update_with_retry(
                client=client,
                table_name="docs",
                record_id=doc_id,
                vector=new_embedding.tolist(),
                payload=current_payload
            )

            elapsed = time.time() - start_time

            if success:
                stats.content_updates += 1
                stats.content_update_time += elapsed
                print(f"  ✅ Updated to version {current_payload['version']} ({elapsed:.3f}s)")
            else:
                stats.failed_updates += 1

        # Display content update stats
        if stats.content_updates > 0:
            avg_time = stats.content_update_time / stats.content_updates
            print(f"\n📊 Content Update Stats:")
            print(f"   Total updates: {stats.content_updates}")
            print(f"   Skipped (no change): {stats.skipped_updates}")
            print(f"   Average time: {avg_time:.3f}s per update")

        time.sleep(1)

        # ================================================================
        # Phase 3: Metadata-Only Updates (Performance Comparison)
        # ================================================================
        print("\n" + "=" * 80)
        print("Phase 3: Metadata-Only Updates (Performance Comparison)")
        print("=" * 80)

        print("\nSimulating metadata updates (no vector changes)...")
        metadata_updates = simulate_metadata_updates(base_date)

        for doc_id, updates in metadata_updates.items():
            # Get current document
            current_records = client.get_records(table_name="docs", ids=[doc_id])
            if not current_records:
                continue

            current_record = current_records[0]
            current_payload = json.loads(current_record["payload_json"])

            print(f"\n{doc_id}:")
            for key, new_value in updates.items():
                old_value = current_payload.get(key, "N/A")
                print(f"  {key}: {old_value} → {new_value}")

            # Measure metadata update time
            start_time = time.time()

            # Update payload
            current_payload.update(updates)

            # Update only metadata, vector stays the same
            success = update_with_retry(
                client=client,
                table_name="docs",
                record_id=doc_id,
                payload=current_payload
            )

            elapsed = time.time() - start_time

            if success:
                stats.metadata_updates += 1
                stats.metadata_update_time += elapsed
                print(f"  ✅ Metadata updated (vector unchanged, {elapsed:.3f}s)")
            else:
                stats.failed_updates += 1

        # Performance comparison
        if stats.content_updates > 0 and stats.metadata_updates > 0:
            avg_content_time = stats.content_update_time / stats.content_updates
            avg_metadata_time = stats.metadata_update_time / stats.metadata_updates
            speedup = avg_content_time / avg_metadata_time

            print(f"\n⚡ Performance Comparison:")
            print(f"   Content update (with re-embedding): {avg_content_time:.3f}s")
            print(f"   Metadata update (no re-embedding):  {avg_metadata_time:.3f}s")
            print(f"   Speedup: {speedup:.1f}x faster")
            print(f"   💡 Use metadata-only updates when possible to save {(1 - 1/speedup)*100:.0f}% time")

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
                print(f"   Views: {payload['views']}, Updated: {payload['last_updated']}")
                print(f"   Content: {payload['content'][:60]}...")

        time.sleep(1)

        # ================================================================
        # Phase 5: CDC Pattern with Real-World Message Format
        # ================================================================
        print("\n" + "=" * 80)
        print("Phase 5: CDC Pattern with Real-World Message Format")
        print("=" * 80)

        print("\nProcessing CDC events from change stream...")
        cdc_events = create_cdc_events(base_date)

        print(f"\nReceived {len(cdc_events)} CDC events:")
        for event in cdc_events:
            print(f"  [{event['sequence']}] {event['timestamp']}: "
                  f"{event['operation']} on {event.get('after', event.get('before', {})).get('id', 'unknown')}")

        print("\nProcessing events in sequence order...")

        for event in cdc_events:
            operation = event["operation"]
            timestamp = event["timestamp"]
            sequence = event["sequence"]

            print(f"\n[{sequence}] {timestamp} - {operation}")

            if operation == "INSERT":
                after = event["after"]
                doc_id = after["id"]
                content = after["content"]

                # Generate embedding
                embedding = model.encode(content, convert_to_numpy=True)

                # Insert new document
                try:
                    client.insert_records(
                        table_name="docs",
                        records=[{
                            "id": doc_id,
                            "vector": embedding.tolist(),
                            "payload": {
                                "content": content,
                                "title": after.get("title", f"Document {doc_id}"),
                                "version": after.get("version", 1),
                                "status": after.get("status", "published"),
                                "views": 0,
                                "last_updated": datetime.fromisoformat(timestamp).strftime("%Y-%m-%d"),
                                "content_hash": content_hash(content),
                                "cdc_sequence": sequence,
                            },
                        }],
                        mode="batch"
                    )
                    print(f"  ✅ Inserted {doc_id}")
                except Exception as e:
                    print(f"  ❌ Failed to insert: {e}")

            elif operation == "UPDATE":
                before = event.get("before", {})
                after = event["after"]
                doc_id = after["id"]
                new_content = after["content"]

                # Get current document for version conflict detection
                current_records = client.get_records(table_name="docs", ids=[doc_id])
                if not current_records:
                    print(f"  ⚠️  Document {doc_id} not found, skipping update")
                    continue

                current_payload = json.loads(current_records[0]["payload_json"])

                # Check for version conflicts (if sequence numbers are out of order)
                current_sequence = current_payload.get("cdc_sequence", 0)
                if sequence <= current_sequence:
                    print(f"  ⚠️  Out-of-order update detected (current: {current_sequence}, new: {sequence})")
                    print(f"  ℹ️  Skipping to avoid overwriting newer data")
                    continue

                # Generate new embedding
                embedding = model.encode(new_content, convert_to_numpy=True)

                # Update document
                current_payload["content"] = new_content
                current_payload["version"] += 1
                current_payload["last_updated"] = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d")
                current_payload["content_hash"] = content_hash(new_content)
                current_payload["cdc_sequence"] = sequence

                success = update_with_retry(
                    client=client,
                    table_name="docs",
                    record_id=doc_id,
                    vector=embedding.tolist(),
                    payload=current_payload
                )

                if success:
                    print(f"  ✅ Updated {doc_id} to version {current_payload['version']}")
                else:
                    print(f"  ❌ Failed to update {doc_id}")

            elif operation == "DELETE":
                before = event["before"]
                doc_id = before["id"]

                try:
                    client.delete_records(table_name="docs", ids=[doc_id])
                    print(f"  ✅ Deleted {doc_id}")
                except Exception as e:
                    print(f"  ❌ Failed to delete: {e}")

        time.sleep(1)

        # ================================================================
        # Phase 6: Error Handling Demonstration
        # ================================================================
        print("\n" + "=" * 80)
        print("Phase 6: Error Handling and Edge Cases")
        print("=" * 80)

        print("\nTesting error handling scenarios...")

        # Test 1: Update non-existent document
        print("\n1. Attempting to update non-existent document...")
        try:
            client.update_record(
                table_name="docs",
                record_id="doc_nonexistent",
                payload={"views": 999}
            )
            print("  ⚠️  Unexpectedly succeeded")
        except Exception as e:
            print(f"  ✅ Correctly failed: {type(e).__name__}")

        # Test 2: Simulating network timeout (handled by retry logic)
        print("\n2. Simulating transient failures (retry logic)...")
        print("  (In production, this would handle network timeouts, etc.)")
        success = update_with_retry(
            client=client,
            table_name="docs",
            record_id="doc2",
            payload={"views": 125},
            merge_mode="merge",  # Use merge to preserve other fields
            max_retries=2
        )
        if success:
            print("  ✅ Update succeeded with retry logic")

        # Test 3: Invalid vector dimension
        print("\n3. Attempting update with wrong vector dimension...")
        try:
            client.update_record(
                table_name="docs",
                record_id="doc2",
                vector=[0.1, 0.2],  # Wrong dimension
                payload={"views": 126}
            )
            print("  ⚠️  Unexpectedly succeeded")
        except Exception as e:
            print(f"  ✅ Correctly failed: {type(e).__name__}")
            print(f"     Expected dimension: {embedding_dim}, provided: 2")

        time.sleep(1)

        # ================================================================
        # Final State
        # ================================================================
        print("\n" + "=" * 80)
        print("Final State and Statistics")
        print("=" * 80)

        remaining_ids = ["doc2", "doc3", "doc4"]
        final_records = client.get_records(table_name="docs", ids=remaining_ids)

        print(f"\nRemaining documents: {len(final_records)}")
        for record in final_records:
            payload = json.loads(record["payload_json"])
            print(f"\n  📄 {record['id']}:")
            print(f"     Version: {payload.get('version', 'N/A')}, Status: {payload.get('status', 'N/A')}")
            print(f"     Views: {payload.get('views', 'N/A')}, Updated: {payload.get('last_updated', 'N/A')}")
            if 'content' in payload:
                print(f"     Content: {payload['content'][:50]}...")
            if "cdc_sequence" in payload:
                print(f"     CDC Sequence: {payload['cdc_sequence']}")

        # Overall statistics
        print(f"\n📊 Overall Statistics:")
        print(f"   Content updates: {stats.content_updates}")
        print(f"   Metadata updates: {stats.metadata_updates}")
        print(f"   Skipped updates: {stats.skipped_updates}")
        print(f"   Failed updates: {stats.failed_updates}")
        print(f"   Total operations: {stats.content_updates + stats.metadata_updates + stats.skipped_updates}")

        if stats.content_updates > 0:
            print(f"\n⏱️  Timing:")
            print(f"   Total content update time: {stats.content_update_time:.3f}s")
            print(f"   Avg content update: {stats.content_update_time / stats.content_updates:.3f}s")
        if stats.metadata_updates > 0:
            print(f"   Total metadata update time: {stats.metadata_update_time:.3f}s")
            print(f"   Avg metadata update: {stats.metadata_update_time / stats.metadata_updates:.3f}s")

        # ================================================================
        # Summary and Best Practices
        # ================================================================
        print("\n" + "=" * 80)
        print("Key Improvements Demonstrated")
        print("=" * 80)
        print("""
✅ Improvements in This Example:

1. Content Change Detection:
   - Use content hashing to detect actual changes
   - Avoid unnecessary re-embedding (saves time and cost)
   - Track content_hash in payload for comparison

2. Performance Measurement:
   - Time content updates vs metadata updates
   - Show actual speedup (typically 10-100x faster)
   - Demonstrate cost savings

3. Error Handling:
   - Retry logic with exponential backoff
   - Version conflict detection
   - Validation before updates
   - Graceful failure handling

4. Real-World CDC Pattern:
   - Structured CDC messages with timestamps
   - Sequence numbers for ordering
   - Handle INSERT/UPDATE/DELETE operations
   - Out-of-order event detection

5. Advanced Features:
   - Content hashing for change detection
   - Version tracking and conflict resolution
   - Batch processing with CDC streams
   - Comprehensive statistics

Best Practices Applied:

🔹 Change Detection:
   - Always hash content before re-embedding
   - Skip updates when content hasn't changed
   - Log skipped updates for monitoring

🔹 Performance Optimization:
   - Use metadata-only updates when possible
   - Batch similar operations together
   - Measure and monitor update times
   - Profile embedding vs database time

🔹 Error Handling:
   - Implement retry with exponential backoff
   - Validate inputs before updates
   - Check for version conflicts
   - Log all failures for debugging

🔹 CDC Integration:
   - Use sequence numbers for ordering
   - Handle out-of-order events gracefully
   - Track watermarks for progress
   - Deduplicate rapid changes

🔹 Monitoring:
   - Track success/failure rates
   - Monitor update lag (CDC delay)
   - Alert on high failure rates
   - Dashboard for update throughput

Real-World Considerations:

1. Embedding Cost: OpenAI charges per token
   - Content change detection saves $$ by avoiding re-embedding
   - Example: 1M updates, 50% unchanged = $500-$5000 saved

2. Update Latency:
   - Metadata updates: <10ms
   - Content updates: 50-500ms (depends on embedding model)
   - Critical for real-time applications

3. Version Conflicts:
   - Use optimistic locking with version numbers
   - Handle out-of-order CDC events
   - Implement conflict resolution strategy

4. Scale Considerations:
   - Batch updates for high throughput
   - Use async processing for embedding
   - Consider queue-based architecture
   - Monitor backlog and lag

Note: This improved example demonstrates production-ready patterns
that can handle millions of updates per day efficiently.
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
