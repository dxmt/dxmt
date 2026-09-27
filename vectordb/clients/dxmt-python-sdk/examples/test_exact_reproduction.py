"""Exact reproduction of batch_operations.py sequence.

This test replicates EVERY operation from batch_operations.py
to identify the exact trigger for the insert failure.
"""

import random
import time
from typing import List, Dict, Any

from dxmt import Client


def generate_records(start_id: int, count: int, dimension: int = 128) -> List[Dict[str, Any]]:
    """Generate random records."""
    return [{
        "id": f"doc{i}",
        "vector": [random.random() for _ in range(dimension)],
        "payload": {
            "content": f"Document {i} content",
            "category": random.choice(["tech", "science", "business", "sports"]),
            "timestamp": time.time(),
            "index": i,
        },
    } for i in range(start_id, start_id + count)]


def main():
    """Exact reproduction of batch_operations.py."""
    print("=" * 80)
    print("Exact Reproduction Test")
    print("=" * 80)
    print("\nReplicating EVERY operation from batch_operations.py\n")

    client = Client(host="localhost", port=9999)
    dimension = 128
    table_name = "batch_demo"

    try:
        # Initial cleanup
        try:
            client.delete_table(table_name)
        except Exception:
            pass

        # ================================================================
        # Example 1: Batch Insert Modes Comparison (EXACT REPLICA)
        # ================================================================
        print("=" * 80)
        print("Example 1: Batch Size Comparison")
        print("=" * 80)

        batch_sizes = [100, 500, 1000]

        for repeat in range(3):
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

                # Insert
                client.insert_records(table_name=table_name, records=records)
                print(f"  {batch_size}: ✅")

        print("✅ Example 1 completed")

        # ================================================================
        # Example 2: Large-Scale Data Ingestion (EXACT REPLICA)
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

        # Insert in chunks
        total_records = 10000
        chunk_size = 1000

        for i in range(0, total_records, chunk_size):
            chunk = generate_records(i + 1, min(chunk_size, total_records - i), dimension)
            client.insert_records(table_name=table_name, records=chunk)
            print(f"  Chunk {i // chunk_size + 1}/10: ✅")

        print("✅ Example 2 completed")

        # ================================================================
        # Example 3: Batch Retrieval (EXACT REPLICA)
        # ================================================================
        print("\n" + "=" * 80)
        print("Example 3: Batch Retrieval")
        print("=" * 80)

        ids_to_fetch = [f"doc{i}" for i in range(1, 101, 10)]
        records = client.get_records(table_name=table_name, ids=ids_to_fetch)
        print(f"  Retrieved {len(records)} records: ✅")

        # ================================================================
        # Example 4: Batch Delete (EXACT REPLICA)
        # ================================================================
        print("\n" + "=" * 80)
        print("Example 4: Batch Delete")
        print("=" * 80)

        ids_to_delete = [f"doc{i}" for i in range(1, 501)]
        client.delete_records(table_name=table_name, ids=ids_to_delete)
        print(f"  Deleted {len(ids_to_delete)} records: ✅")

        # Verify
        remaining = client.get_records(table_name=table_name, ids=[f"doc{i}" for i in range(501, 511)])
        print(f"  Verified {len(remaining)} remaining records: ✅")

        # ================================================================
        # Example 5: Batch Search (EXACT REPLICA)
        # ================================================================
        print("\n" + "=" * 80)
        print("Example 5: Batch Search (Multiple Query Vectors)")
        print("=" * 80)

        num_queries = 5
        query_vectors = [[random.random() for _ in range(dimension)] for _ in range(num_queries)]

        for i, query_vector in enumerate(query_vectors):
            results = client.search(table_name=table_name, vector=query_vector, k=10)
            print(f"  Query {i + 1}: {len(results)} results ✅")

        print("✅ Example 5 completed")

        # ================================================================
        # Example 6: Incremental Updates Pattern (EXACT REPLICA)
        # ================================================================
        print("\n" + "=" * 80)
        print("Example 6: Incremental Updates Pattern")
        print("=" * 80)

        print("\nSimulating periodic updates...")

        # Simulate 3 update batches
        for batch_num in range(1, 4):
            print(f"\n  Update batch {batch_num}:")

            # Generate new records to insert
            new_records = generate_records(10000 + (batch_num - 1) * 100, 50, dimension)

            for attempt in range(3):
                try:
                    start = time.time()
                    client.insert_records(table_name=table_name, records=new_records)
                    duration = time.time() - start
                    print(f"    ✅ Inserted {len(new_records)} records (attempt {attempt + 1}, {duration:.4f}s)")
                    break
                except Exception as e:
                    print(f"    ❌ Failed (attempt {attempt + 1}): {str(e)[:60]}")
                    if attempt < 2:
                        wait = (attempt + 1) * 0.5
                        print(f"       Retrying after {wait}s...")
                        time.sleep(wait)
            else:
                print(f"    ⚠️  All retries failed for batch {batch_num}")

            time.sleep(0.5)

        print("\n✅ Example 6 completed")

        # ================================================================
        # Analysis
        # ================================================================
        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)
        print("""
If this test successfully replicates the failure, we can conclude:
  - The issue is deterministic and related to the specific sequence
  - The trigger is in the combination of all operations, not just one

If this test PASSES (no failure), it suggests:
  - The issue is non-deterministic (timing, race condition)
  - Server state varies between runs
  - External factors (system load, memory pressure)

Next steps depend on whether we can reproduce the failure.
        """)

    finally:
        # Cleanup
        print("\nCleaning up...")
        try:
            client.delete_table(table_name)
            print("✅ Deleted test table")
        except Exception as e:
            print(f"⚠️  Cleanup failed: {e}")

        client.close()
        print("\n✅ Test complete!")


if __name__ == "__main__":
    main()
