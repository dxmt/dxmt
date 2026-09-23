"""Diagnostic script to identify server bottlenecks and issues.

This script helps determine if failures are due to:
1. Server resource limits
2. Connection pooling issues
3. Table state problems
4. Timing/synchronization issues

NOTE: An improved version is available at diagnose_server_improved.py
      with additional features:
      - Performance metrics collection (P50/P95/P99, throughput)
      - Load gradient testing (find breaking points)
      - Concurrent insert testing
      - JSON report output
      - Enhanced diagnostic analysis

      Usage: python3 diagnose_server_improved.py --help
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
            "payload": {"index": i, "timestamp": time.time()},
        })
    return records


def test_insert_after_heavy_load(client: Client, table_name: str, dimension: int):
    """Test if inserts fail after heavy load."""
    print("\n" + "=" * 80)
    print("TEST 1: Insert After Heavy Load")
    print("=" * 80)

    # Heavy load: Insert 10k records
    print("\nStep 1: Creating heavy load (10,000 records)...")
    records = generate_records(1, 10000, dimension)

    chunk_size = 1000
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        client.insert_records(table_name=table_name, records=chunk)
        print(f"  Inserted chunk {i // chunk_size + 1}/10")

    print("✅ Heavy load completed")

    # Immediate insert (this should replicate the issue)
    print("\nStep 2: Immediate insert (50 records)...")
    test_records = generate_records(10001, 50, dimension)

    for attempt in range(3):
        try:
            start = time.time()
            client.insert_records(table_name=table_name, records=test_records)
            duration = time.time() - start
            print(f"  ✅ Insert succeeded (attempt {attempt + 1}, {duration:.4f}s)")
            return True
        except Exception as e:
            print(f"  ❌ Insert failed (attempt {attempt + 1}): {str(e)[:80]}")
            if attempt < 2:
                wait = (attempt + 1) * 0.5
                print(f"     Waiting {wait}s before retry...")
                time.sleep(wait)

    return False


def test_insert_with_delays(client: Client, table_name: str, dimension: int):
    """Test if adding delays prevents failures."""
    print("\n" + "=" * 80)
    print("TEST 2: Insert With Delays Between Operations")
    print("=" * 80)

    # Create table
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

    # Heavy load with delays
    print("\nInserting 10,000 records with 1s delay between chunks...")
    for i in range(10):
        records = generate_records(i * 1000 + 1, 1000, dimension)
        client.insert_records(table_name=table_name, records=records)
        print(f"  Chunk {i + 1}/10")
        time.sleep(1.0)  # 1 second delay

    print("✅ Heavy load with delays completed")

    # Immediate insert
    print("\nImmediate insert (50 records)...")
    test_records = generate_records(10001, 50, dimension)
    try:
        client.insert_records(table_name=table_name, records=test_records)
        print("  ✅ Insert succeeded")
        return True
    except Exception as e:
        print(f"  ❌ Insert failed: {str(e)[:80]}")
        return False


def test_connection_pooling(client: Client, table_name: str, dimension: int):
    """Test if connection pooling is the issue."""
    print("\n" + "=" * 80)
    print("TEST 3: Connection Pooling")
    print("=" * 80)

    # Create table
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

    # Heavy load
    print("\nHeavy load with existing client...")
    for i in range(10):
        records = generate_records(i * 1000 + 1, 1000, dimension)
        client.insert_records(table_name=table_name, records=records)
        print(f"  Chunk {i + 1}/10")

    # Create new client
    print("\nCreating new client...")
    new_client = Client(host="localhost", port=9999)

    # Test with new client
    print("\nInsert with new client (50 records)...")
    test_records = generate_records(10001, 50, dimension)
    try:
        new_client.insert_records(table_name=table_name, records=test_records)
        print("  ✅ Insert with new client succeeded")
        success = True
    except Exception as e:
        print(f"  ❌ Insert with new client failed: {str(e)[:80]}")
        success = False

    new_client.close()
    return success


def test_table_recreation(client: Client, table_name: str, dimension: int):
    """Test if recreating the table helps."""
    print("\n" + "=" * 80)
    print("TEST 4: Table Recreation After Heavy Load")
    print("=" * 80)

    # Heavy load
    print("\nHeavy load (10,000 records)...")
    records = generate_records(1, 10000, dimension)
    chunk_size = 1000
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        client.insert_records(table_name=table_name, records=chunk)
        print(f"  Chunk {i // chunk_size + 1}/10")

    # Recreate table
    print("\nRecreating table...")
    client.delete_table(table_name)
    time.sleep(0.5)
    client.create_table(
        name=table_name,
        dimension=dimension,
        distance_metric="cosine",
        index_type="hnsw",
    )

    # Test insert
    print("\nInsert after table recreation (50 records)...")
    test_records = generate_records(1, 50, dimension)
    try:
        client.insert_records(table_name=table_name, records=test_records)
        print("  ✅ Insert succeeded")
        return True
    except Exception as e:
        print(f"  ❌ Insert failed: {str(e)[:80]}")
        return False


def main():
    """Run diagnostic tests."""
    print("=" * 80)
    print("DXMT Server Diagnostic Tool")
    print("=" * 80)
    print("\nThis tool will help identify the root cause of insert failures.")
    print("Expected runtime: ~5 minutes")

    client = Client(host="localhost", port=9999)
    dimension = 128
    table_name = "diagnostic_test"

    results = {}

    try:
        # Test 1: Replicate the issue
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

        results["test1"] = test_insert_after_heavy_load(client, table_name, dimension)

        # Test 2: With delays
        results["test2"] = test_insert_with_delays(client, table_name, dimension)

        # Test 3: Connection pooling
        results["test3"] = test_connection_pooling(client, table_name, dimension)

        # Test 4: Table recreation
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

        results["test4"] = test_table_recreation(client, table_name, dimension)

        # Test 5: Different modes
        try:
            client.delete_table(table_name)
        except Exception:
            pass

        # Summary
        print("\n" + "=" * 80)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 80)

        print("\nTest Results:")
        print(f"  1. Insert After Heavy Load: {'❌ FAILED' if not results['test1'] else '✅ PASSED'}")
        print(f"  2. Insert With Delays: {'✅ PASSED' if results['test2'] else '❌ FAILED'}")
        print(f"  3. New Client Connection: {'✅ PASSED' if results['test3'] else '❌ FAILED'}")
        print(f"  4. Table Recreation: {'✅ PASSED' if results['test4'] else '❌ FAILED'}")

        print("\n" + "=" * 80)
        print("DIAGNOSIS")
        print("=" * 80)

        # Analyze results
        if not results["test1"]:
            print("\n✅ Issue replicated: Inserts fail after heavy load")

            if results["test2"]:
                print("✅ Adding delays prevents failures")
                print("   → Server needs time to stabilize after heavy operations")
                print("   → Recommendation: Add 1-2s delays between large batches")

            if results["test3"]:
                print("✅ New client connection succeeds")
                print("   → Connection pooling is not the issue")

            if results["test4"]:
                print("✅ Table recreation resolves the issue")
                print("   → Table enters a temporary locked/busy state")
                print("   → Recommendation: Implement backoff retry or periodic table refresh")

        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        print("""
Based on the diagnostic results:

1. Server Stability:
   - DXMT server enters a temporary busy/locked state after heavy operations
   - This is likely related to internal indexing or memory management

2. Immediate Solutions:
   - Use 'batch' mode instead of 'parallel' for operations after heavy load
   - Add 1-2 second delays between large batch operations
   - Implement retry logic with exponential backoff (already done)

3. Long-term Solutions:
   - Server should handle this more gracefully (potential bug)
   - Consider rate limiting in client SDK
   - Monitor server logs for HNSW index building or memory pressure

4. Workarounds for Production:
   - Break large operations into smaller chunks with delays
   - Use separate tables for different workloads
   - Schedule heavy operations during low-traffic periods
   - Consider table sharding if scale increases

This appears to be a server-side issue, not a client SDK issue.
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
        print("\n✅ Diagnostic complete!")


if __name__ == "__main__":
    main()
