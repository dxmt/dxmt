"""Test table churn (frequent create/delete) impact.

This test specifically targets the issue found in batch_operations.py
where frequent table creation/deletion leads to insert failures.
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
        "payload": {"index": i},
    } for i in range(start_id, start_id + count)]


def main():
    """Test table churn impact."""
    print("=" * 80)
    print("Table Churn Test")
    print("=" * 80)
    print("\nThis test replicates the exact sequence from batch_operations.py")
    print("to identify if frequent table create/delete causes issues.\n")

    client = Client(host="localhost", port=9999)
    dimension = 128
    table_name = "churn_test"

    try:
        # Clean up
        try:
            client.delete_table(table_name)
        except Exception:
            pass

        # ================================================================
        # Phase 1: Replicate Example 1 (9 table create/delete cycles)
        # ================================================================
        print("=" * 80)
        print("Phase 1: 9 Table Create/Delete Cycles (like Example 1)")
        print("=" * 80)

        batch_sizes = [100, 500, 1000]
        cycle_count = 0

        for repeat in range(3):
            for batch_size in batch_sizes:
                cycle_count += 1
                print(f"\nCycle {cycle_count}/9: size={batch_size}")

                # Delete and create table
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

                # Insert records
                records = generate_records(1, batch_size, dimension)
                try:
                    client.insert_records(table_name=table_name, records=records)
                    print(f"  ✅ Inserted {batch_size} records")
                except Exception as e:
                    print(f"  ❌ Failed to insert: {str(e)[:80]}")

        print("\n✅ Completed 9 table create/delete cycles")

        # ================================================================
        # Phase 2: Replicate Example 2 (10th cycle + 10K records)
        # ================================================================
        print("\n" + "=" * 80)
        print("Phase 2: 10th Cycle + 10,000 Records (like Example 2)")
        print("=" * 80)

        # 10th cycle
        print("\nCreating table for 10th time...")
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

        # Insert 10,000 records
        print("\nInserting 10,000 records in chunks...")
        chunk_size = 1000
        for i in range(0, 10000, chunk_size):
            records = generate_records(i + 1, chunk_size, dimension)
            try:
                client.insert_records(table_name=table_name, records=records)
                print(f"  Chunk {i // chunk_size + 1}/10: ✅")
            except Exception as e:
                print(f"  Chunk {i // chunk_size + 1}/10: ❌ {str(e)[:60]}")

        print("\n✅ Completed phase 2")

        # ================================================================
        # Phase 3: Attempt insert (like Example 6)
        # ================================================================
        print("\n" + "=" * 80)
        print("Phase 3: Attempt Insert After All Operations (like Example 6)")
        print("=" * 80)

        print("\nAttempting to insert 50 records...")
        test_records = generate_records(10001, 50, dimension)

        for attempt in range(3):
            try:
                start = time.time()
                client.insert_records(table_name=table_name, records=test_records)
                duration = time.time() - start
                print(f"  ✅ Insert succeeded on attempt {attempt + 1} ({duration:.4f}s)")
                break
            except Exception as e:
                print(f"  ❌ Insert failed on attempt {attempt + 1}: {str(e)[:80]}")
                if attempt < 2:
                    wait = (attempt + 1) * 0.5
                    print(f"     Waiting {wait}s before retry...")
                    time.sleep(wait)
        else:
            print("  ⚠️  All retry attempts exhausted")

        # ================================================================
        # Control Test: Fresh Start
        # ================================================================
        print("\n" + "=" * 80)
        print("Control Test: Fresh Start (No Table Churn)")
        print("=" * 80)

        # Delete and recreate
        print("\nRecreating table from scratch...")
        try:
            client.delete_table(table_name)
        except Exception:
            pass

        time.sleep(1.0)  # Give server time to cleanup

        client.create_table(
            name=table_name,
            dimension=dimension,
            distance_metric="cosine",
            index_type="hnsw",
        )

        # Insert same amount of data
        print("\nInserting 10,000 records...")
        for i in range(0, 10000, 1000):
            records = generate_records(i + 1, 1000, dimension)
            client.insert_records(table_name=table_name, records=records)

        # Immediate insert
        print("\nImmediate insert (50 records)...")
        test_records = generate_records(10001, 50, dimension)
        try:
            client.insert_records(table_name=table_name, records=test_records)
            print("  ✅ Insert succeeded (no table churn)")
        except Exception as e:
            print(f"  ❌ Insert failed even without churn: {str(e)[:80]}")

        # ================================================================
        # Summary
        # ================================================================
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("""
This test replicates the exact sequence from batch_operations.py:
1. 9 table create/delete cycles (Example 1)
2. 10th table creation + 10K record insertion (Example 2)
3. Attempt additional insert (Example 6)

If this test shows the same failure pattern as batch_operations.py,
then the issue is clearly related to:
  - Frequent table creation/deletion (table churn)
  - Cumulative resource pressure
  - Possible resource leak in table cleanup

Recommendation:
  - Add delay after table deletion (e.g., 100-500ms)
  - Implement table creation throttling
  - Server-side: Improve resource cleanup after table deletion
  - Server-side: Add connection/table limits
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
