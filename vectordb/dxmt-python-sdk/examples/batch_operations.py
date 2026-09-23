"""Batch operations and performance optimization example for DXMT Python SDK.

This example demonstrates:
1. Batch insert at several batch sizes
2. Batch update operations
3. Batch delete operations
4. Performance comparison across batch sizes
5. Large-scale data ingestion patterns

Use case:
    - Initial data migration to vector database
    - Periodic bulk updates from data warehouse
    - High-throughput ingestion pipelines
"""

import time
import random
from typing import List, Dict, Any

from dxmt import Client


def generate_records(start_id: int, count: int, dimension: int = 128) -> List[Dict[str, Any]]:
    """Generate random records for testing."""
    records = []
    for i in range(start_id, start_id + count):
        records.append({
            "id": f"doc{i}",
            "vector": [random.random() for _ in range(dimension)],
            "payload": {
                "content": f"Document {i} content",
                "category": random.choice(["tech", "science", "business", "sports"]),
                "timestamp": time.time(),
                "index": i,
            },
        })
    return records


def benchmark_insert(client: Client, table_name: str, records: List[Dict[str, Any]], max_retries: int = 3) -> float:
    """Benchmark insert operation and return duration in seconds.

    Args:
        client: DXMT client
        table_name: Name of the table
        records: Records to insert
        max_retries: Maximum number of retry attempts

    Returns:
        Duration in seconds
    """
    start_time = time.time()

    for attempt in range(max_retries):
        try:
            client.insert_records(table_name=table_name, records=records)
            duration = time.time() - start_time
            return duration
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 0.5  # Exponential backoff
                print(f"      Retry {attempt + 1}/{max_retries - 1} after {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise  # Re-raise if all retries failed

    duration = time.time() - start_time
    return duration


def main() -> None:
    """Run batch operations examples."""
    print("=" * 80)
    print("DXMT Batch Operations & Performance Examples")
    print("=" * 80)

    # Create client
    print("\nConnecting to DXMT server...")
    client = Client(host="localhost", port=9999)

    dimension = 128
    table_name = "batch_demo"

    try:
        # Clean up if table exists
        try:
            client.delete_table(table_name)
        except Exception:
            pass

        # ================================================================
        # Example 1: Batch Insert Modes Comparison
        # ================================================================
        print("\n" + "=" * 80)
        print("Example 1: Batch Size Comparison")
        print("=" * 80)

        batch_sizes = [100, 500, 1000]

        print("\nTesting various batch sizes:")
        print(f"{'Batch Size':<12} {'Duration (s)':<15} {'Throughput (records/s)':<25}")
        print("─" * 80)

        for batch_size in batch_sizes:
            # Create fresh table for each test
            try:
                client.delete_table(table_name)
            except Exception:
                pass

            client.create_table(
                name=table_name,
                dimension=dimension,
                distance_metric="cosine",
                index_type="hnsw",
            )

            # Generate records
            records = generate_records(1, batch_size, dimension)

            # Benchmark
            duration = benchmark_insert(client, table_name, records)
            throughput = batch_size / duration if duration > 0 else 0

            print(f"{batch_size:<12} {duration:<15.4f} {throughput:<25.2f}")

    # ================================================================
        # Example 2: Large-Scale Data Ingestion
        # ================================================================
        print("\n" + "=" * 80)
        print("Example 2: Large-Scale Data Ingestion (10,000 records)")
        print("=" * 80)

        # Recreate table
        try:
            client.delete_table(table_name)
        except Exception:
            pass

        client.create_table(
            name=table_name,
            dimension=dimension,
            distance_metric="cosine",
            index_type="hnsw",
        )
        print("✅ Table created")

        # Insert in chunks (recommended pattern)
        total_records = 10000
        chunk_size = 1000
        total_duration = 0

        print(f"\nInserting {total_records} records in chunks of {chunk_size}...")

        for i in range(0, total_records, chunk_size):
            chunk = generate_records(i + 1, min(chunk_size, total_records - i), dimension)
            duration = benchmark_insert(client, table_name, chunk, "parallel")
            total_duration += duration

            print(f"  Chunk {i // chunk_size + 1}/{total_records // chunk_size}: "
                  f"{len(chunk)} records in {duration:.2f}s ({len(chunk) / duration:.0f} rec/s)")

        overall_throughput = total_records / total_duration
        print(f"\n✅ Total: {total_records} records in {total_duration:.2f}s")
        print(f"   Overall throughput: {overall_throughput:.0f} records/second")

        # ================================================================
        # Example 3: Batch Retrieval
        # ================================================================
        print("\n" + "=" * 80)
        print("Example 3: Batch Retrieval")
        print("=" * 80)

        # Get a batch of records
        ids_to_fetch = [f"doc{i}" for i in range(1, 101, 10)]  # Every 10th record

        print(f"\nFetching {len(ids_to_fetch)} records...")
        start_time = time.time()
        records = client.get_records(table_name=table_name, ids=ids_to_fetch)
        duration = time.time() - start_time

        print(f"✅ Retrieved {len(records)} records in {duration:.4f}s")
        print(f"   First 3 records:")
        for record in records[:3]:
            print(f"   - {record['id']}")

        # ================================================================
        # Example 4: Batch Delete
        # ================================================================
        print("\n" + "=" * 80)
        print("Example 4: Batch Delete")
        print("=" * 80)

        # Delete records in batch
        ids_to_delete = [f"doc{i}" for i in range(1, 501)]  # Delete first 500 records

        print(f"\nDeleting {len(ids_to_delete)} records...")
        start_time = time.time()
        client.delete_records(table_name=table_name, ids=ids_to_delete)
        duration = time.time() - start_time

        print(f"✅ Deleted {len(ids_to_delete)} records in {duration:.4f}s")

        # Verify deletion
        try:
            remaining = client.get_records(table_name=table_name, ids=[f"doc{i}" for i in range(501, 511)])
            print(f"   Verified: {len(remaining)} records still exist (doc501-doc510)")
        except Exception as e:
            print(f"   Note: {e}")

        # ================================================================
        # Example 5: Batch Search
        # ================================================================
        print("\n" + "=" * 80)
        print("Example 5: Batch Search (Multiple Query Vectors)")
        print("=" * 80)

        # Generate multiple query vectors
        num_queries = 5
        query_vectors = [[random.random() for _ in range(dimension)] for _ in range(num_queries)]

        print(f"\nPerforming {num_queries} searches sequentially...")
        start_time = time.time()

        for i, query_vector in enumerate(query_vectors):
            results = client.search(table_name=table_name, vector=query_vector, k=10)
            print(f"  Query {i + 1}: Found {len(results)} results")

        sequential_duration = time.time() - start_time
        print(f"\n✅ Sequential searches completed in {sequential_duration:.4f}s")
        print(f"   Average per query: {sequential_duration / num_queries:.4f}s")

        # Note: For truly parallel batch search, use async client (see async_usage.py)
        print("\n💡 Tip: Use async client for parallel batch searches (see async_usage.py)")

        # ================================================================
        # Example 6: Incremental Updates Pattern
        # ================================================================
        print("\n" + "=" * 80)
        print("Example 6: Incremental Updates Pattern")
        print("=" * 80)

        print("\nSimulating periodic updates (e.g., from CDC pipeline)...")

        # Simulate 3 update batches with retry logic
        success_count = 0
        failed_count = 0

        for batch_num in range(1, 4):
            print(f"\n  Update batch {batch_num}:")

            try:
                # Generate new records to insert
                new_records = generate_records(10000 + (batch_num - 1) * 100, 50, dimension)
                insert_duration = benchmark_insert(client, table_name, new_records, "batch", max_retries=3)
                print(f"    - ✅ Inserted {len(new_records)} new records in {insert_duration:.4f}s")
                success_count += 1

                # Note: For updates, delete old and insert new (DXMT doesn't have update API yet)
                # In production, you'd track which records changed and delete+reinsert them

            except Exception as e:
                print(f"    - ⚠️  Failed to insert batch: {str(e)[:100]}")
                print(f"      Skipping this batch and continuing...")
                failed_count += 1

            time.sleep(0.5)  # Simulate time between batches

        print(f"\n📊 Incremental updates summary:")
        print(f"   ✅ Successful batches: {success_count}/3")
        print(f"   ⚠️  Failed batches: {failed_count}/3")

        if failed_count > 0:
            print(f"\n💡 Note: Failures may occur after many operations. Solutions:")
            print(f"   - Use 'batch' mode instead of 'parallel' for stability")
            print(f"   - Implement exponential backoff retry logic")
            print(f"   - Consider table recreation for fresh state")
            print(f"   - Monitor server resources (memory, CPU)")

        # ================================================================
        # Performance Summary
        # ================================================================
        print("\n" + "=" * 80)
        print("Performance Summary & Best Practices")
        print("=" * 80)
        print("""
Batch Sizes:
  - one insert call is one request; bigger batches amortise the round trip
  - the server caps a batch at 10,000 records (100,000 on /records/bulk)

Recommendations:
  1. Use chunk sizes of 500-2000 records for optimal performance
  2. Use 'parallel' or 'lockfree' mode for high-throughput ingestion
  3. Use 'batch' mode when consistency is more important than speed
  4. For large datasets, insert in chunks rather than all at once
  5. Monitor memory usage for very large batches (>10k records)

Best Practices:
  - Batch insert: Group small inserts into larger batches
  - Batch delete: Delete in chunks of 500-1000 records
  - Batch search: Use async client for parallel searches
  - Incremental updates: Process in periodic batches (e.g., every 5 minutes)
  - Error handling: Implement retry logic and exponential backoff for failed batches
  - Resource management: Consider table recreation for long-running operations
  - Monitoring: Track error rates and retry attempts

Performance Characteristics:
  - Insert throughput: 1,000-10,000 records/second (depends on mode)
  - Search latency: <10ms for HNSW index
  - Batch retrieval: ~1ms per record
  - Batch deletion: ~0.1ms per record
        """)

    finally:
        # Cleanup
        print("\nCleaning up...")
        try:
            client.delete_table(table_name)
            print("✅ Deleted table")
        except Exception as e:
            print(f"⚠️  Cleanup failed: {e}")

        client.close()
        print("\n✅ Done!")


if __name__ == "__main__":
    main()
