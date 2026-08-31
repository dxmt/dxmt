"""Test script to verify server API fixes.

Tests the three critical fixes:
1. Vector dimension validation
2. Payload merge mode
3. Upsert parameter for record existence check
"""

import json
import numpy as np
from dxmt import Client
import httpx


def test_dimension_validation(client: Client):
    """Test 1: Vector dimension validation."""
    print("\n" + "=" * 80)
    print("Test 1: Vector Dimension Validation")
    print("=" * 80)

    # Clean up and create fresh table
    try:
        client.delete_table("test_dim")
    except:
        pass

    client.create_table(
        name="test_dim",
        dimension=384,
        distance_metric="cosine"
    )
    print("✅ Created table with dimension=384")

    # Insert a record
    correct_vector = np.random.rand(384).tolist()
    client.insert_records(
        table_name="test_dim",
        records=[{
            "id": "doc1",
            "vector": correct_vector,
            "payload": {"content": "Test document", "version": 1}
        }],
        mode="batch"
    )
    print("✅ Inserted record with correct dimension")

    # Test: Try to update with wrong dimension
    print("\nAttempting to update with wrong dimension (2)...")
    try:
        wrong_vector = [0.1, 0.2]  # Only 2 dimensions
        response = client._client.put(
            f"/tables/test_dim/records/doc1",
            json={
                "vector": wrong_vector,
                "payload": {"views": 100}
            }
        )

        if response.status_code == 400:
            error_data = response.json()
            print(f"✅ Correctly rejected (400 Bad Request)")
            print(f"   Error code: {error_data.get('error', {}).get('code', 'N/A')}")
            print(f"   Message: {error_data.get('error', {}).get('message', 'N/A')}")
            if 'details' in error_data.get('error', {}):
                print(f"   Expected dimension: {error_data['error']['details'].get('expected_dimension')}")
                print(f"   Actual dimension: {error_data['error']['details'].get('actual_dimension')}")
            test_ok = True
        else:
            print(f"❌ Unexpectedly succeeded (status: {response.status_code})")
            test_ok = False
    except Exception as e:
        print(f"❌ Exception: {e}")
        test_ok = False
    finally:
        # Clean up
        try:
            client.delete_table("test_dim")
        except:
            pass

    return test_ok


def test_payload_merge_mode(client: Client):
    """Test 2: Payload merge vs replace mode."""
    print("\n" + "=" * 80)
    print("Test 2: Payload Merge Mode")
    print("=" * 80)

    # Clean up and create fresh table
    try:
        client.delete_table("test_merge")
    except:
        pass

    client.create_table(
        name="test_merge",
        dimension=384,
        distance_metric="cosine"
    )

    # Setup: Insert a record with multiple fields
    vector = np.random.rand(384).tolist()
    client.insert_records(
        table_name="test_merge",
        records=[{
            "id": "doc2",
            "vector": vector,
            "payload": {
                "content": "Original content",
                "title": "Original title",
                "version": 1,
                "status": "published",
                "views": 100
            }
        }],
        mode="batch"
    )
    print("✅ Inserted record with multiple payload fields")

    initial_record = client.get_records(table_name="test_merge", ids=["doc2"])[0]
    initial_payload = json.loads(initial_record["payload_json"])
    print(f"   Initial payload: {list(initial_payload.keys())}")

    # Test 2a: Replace mode (default)
    print("\n2a. Testing REPLACE mode (default)...")
    response = client._client.put(
        f"/tables/test_merge/records/doc2",
        json={
            "payload": {"views": 150},  # Only update views
            "merge_mode": "replace"
        }
    )

    if response.status_code == 200:
        print("✅ Update succeeded")
        record = client.get_records(table_name="test_merge", ids=["doc2"])[0]
        payload = json.loads(record["payload_json"])
        print(f"   Payload after replace: {list(payload.keys())}")

        # In replace mode, only the new field should exist
        if "views" in payload and payload["views"] == 150:
            if "content" not in payload or "title" not in payload:
                print("✅ Replace mode working: other fields removed")
                replace_ok = True
            else:
                print("❌ Replace mode failed: other fields still present")
                replace_ok = False
        else:
            print("❌ Views not updated")
            replace_ok = False
    else:
        print(f"❌ Update failed (status: {response.status_code})")
        replace_ok = False

    # Reset record for merge test using update with replace mode
    response = client._client.put(
        f"/tables/test_merge/records/doc2",
        json={
            "vector": vector,
            "payload": {
                "content": "Original content",
                "title": "Original title",
                "version": 1,
                "status": "published",
                "views": 100
            },
            "merge_mode": "replace",
            "upsert": True
        }
    )

    # Test 2b: Merge mode
    print("\n2b. Testing MERGE mode...")
    response = client._client.put(
        f"/tables/test_merge/records/doc2",
        json={
            "payload": {"views": 200, "last_updated": "2026-01-22"},
            "merge_mode": "merge"
        }
    )

    if response.status_code == 200:
        print("✅ Update succeeded")
        record = client.get_records(table_name="test_merge", ids=["doc2"])[0]
        payload = json.loads(record["payload_json"])
        print(f"   Payload after merge: {list(payload.keys())}")

        # In merge mode, old fields should be preserved
        if ("views" in payload and payload["views"] == 200 and
            "content" in payload and payload["content"] == "Original content" and
            "title" in payload and
            "last_updated" in payload):
            print("✅ Merge mode working: old fields preserved, new fields added")
            merge_ok = True
        else:
            print("❌ Merge mode failed:")
            print(f"   views={payload.get('views')} (expected 200)")
            print(f"   content present: {'content' in payload} (expected True)")
            print(f"   title present: {'title' in payload} (expected True)")
            print(f"   last_updated present: {'last_updated' in payload} (expected True)")
            merge_ok = False
    else:
        print(f"❌ Update failed (status: {response.status_code})")
        merge_ok = False

    # Clean up
    try:
        client.delete_table("test_merge")
    except:
        pass

    return replace_ok and merge_ok


def test_upsert_parameter(client: Client):
    """Test 3: Upsert parameter for record existence check."""
    print("\n" + "=" * 80)
    print("Test 3: Upsert Parameter")
    print("=" * 80)

    # Clean up and create fresh table
    try:
        client.delete_table("test_upsert")
    except:
        pass

    client.create_table(
        name="test_upsert",
        dimension=384,
        distance_metric="cosine"
    )

    # Test 3a: Update non-existent record with upsert=false (default)
    print("\n3a. Testing upsert=false (default) on non-existent record...")
    response = client._client.put(
        f"/tables/test_upsert/records/doc_nonexistent",
        json={
            "payload": {"views": 999},
            "upsert": False  # Explicit false
        }
    )

    if response.status_code == 404:
        error_data = response.json()
        print("✅ Correctly rejected (404 Not Found)")
        print(f"   Error code: {error_data.get('error', {}).get('code', 'N/A')}")
        print(f"   Message: {error_data.get('error', {}).get('message', 'N/A')}")
        upsert_false_ok = True
    else:
        print(f"❌ Unexpectedly succeeded or wrong status (status: {response.status_code})")
        upsert_false_ok = False

    # Test 3b: Update non-existent record with upsert=true
    print("\n3b. Testing upsert=true on non-existent record...")
    vector = np.random.rand(384).tolist()
    response = client._client.put(
        f"/tables/test_upsert/records/doc3",
        json={
            "vector": vector,
            "payload": {"content": "Created via upsert", "version": 1},
            "upsert": True  # Explicit true
        }
    )

    if response.status_code == 200:
        print("✅ Successfully created record via upsert")

        # Verify record exists
        try:
            record = client.get_records(table_name="test_upsert", ids=["doc3"])[0]
            payload = json.loads(record["payload_json"])
            if payload.get("content") == "Created via upsert":
                print("✅ Record created with correct payload")
                upsert_true_ok = True
            else:
                print("❌ Record payload incorrect")
                upsert_true_ok = False
        except:
            print("❌ Record not found after upsert")
            upsert_true_ok = False
    else:
        print(f"❌ Upsert failed (status: {response.status_code})")
        upsert_true_ok = False

    # Clean up
    try:
        client.delete_table("test_upsert")
    except:
        pass

    return upsert_false_ok and upsert_true_ok


def main():
    """Run all API fix tests."""
    print("=" * 80)
    print("DXMT Server API Fixes - Verification Tests")
    print("=" * 80)
    print("\nTesting the following fixes:")
    print("1. 🔴 CRITICAL: Vector dimension validation")
    print("2. 🟡 HIGH: Payload merge vs replace mode")
    print("3. 🟢 LOW: Upsert parameter for record existence")

    # Connect to server
    print("\nConnecting to DXMT server at localhost:9999...")
    client = Client(host="localhost", port=9999)
    print("✅ Connected")

    # Run tests
    results = {}

    try:
        results['dimension_validation'] = test_dimension_validation(client)
        results['payload_merge'] = test_payload_merge_mode(client)
        results['upsert_parameter'] = test_upsert_parameter(client)

        # Summary
        print("\n" + "=" * 80)
        print("Test Results Summary")
        print("=" * 80)

        all_passed = all(results.values())

        print(f"\n1. Vector Dimension Validation: {'✅ PASS' if results['dimension_validation'] else '❌ FAIL'}")
        print(f"2. Payload Merge Mode:          {'✅ PASS' if results['payload_merge'] else '❌ FAIL'}")
        print(f"3. Upsert Parameter:            {'✅ PASS' if results['upsert_parameter'] else '❌ FAIL'}")

        print("\n" + "=" * 80)
        if all_passed:
            print("✅ ALL TESTS PASSED! Server API fixes are working correctly.")
        else:
            print("❌ SOME TESTS FAILED. Please check the output above.")
        print("=" * 80)

    finally:
        client.close()
        print("\n✅ Connection closed")


if __name__ == "__main__":
    main()
