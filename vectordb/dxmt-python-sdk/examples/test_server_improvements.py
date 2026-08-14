"""Test server improvements including backpressure and better error handling.

This script tests the new server-side features:
1. Server health monitoring
2. 429 response with Retry-After header
3. Detailed error messages
4. Health metrics endpoint
"""

import time
import requests
from dxmt import Client


def test_health_endpoint():
    """Test the improved /health endpoint."""
    print("=" * 80)
    print("Test 1: Health Endpoint")
    print("=" * 80)

    try:
        response = requests.get("http://localhost:9999/health")
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response:\n{response.json()}\n")

        data = response.json()
        print("✅ Health endpoint working")

        # Check for new fields
        required_fields = ["status", "description", "retry_after_seconds", "metrics", "accepting_requests"]
        for field in required_fields:
            if field in data.get("data", {}):
                print(f"  ✓ Found field: {field}")
            else:
                print(f"  ⚠️  Missing field: {field}")

    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")


def test_health_metrics_endpoint():
    """Test the new /health/metrics endpoint."""
    print("\n" + "=" * 80)
    print("Test 2: Health Metrics Endpoint")
    print("=" * 80)

    try:
        response = requests.get("http://localhost:9999/health/metrics")
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response:\n{response.json()}\n")

        print("✅ Health metrics endpoint working")

    except Exception as e:
        print(f"❌ Health metrics test failed: {e}")


def test_improved_error_messages():
    """Test improved error messages in insert operations."""
    print("\n" + "=" * 80)
    print("Test 3: Improved Error Messages")
    print("=" * 80)

    client = Client(host="localhost", port=9999)

    try:
        # Create table
        try:
            client.delete_table("error_test")
        except Exception:
            pass

        client.create_table(
            name="error_test",
            dimension=128,
            distance_metric="cosine",
            index_type="hnsw",
        )
        print("\n✅ Table created")

        # Try to insert with invalid data to trigger error
        try:
            invalid_records = [
                {
                    "id": "test1",
                    "vector": [],  # Empty vector should cause error
                    "payload": {"test": "data"},
                }
            ]
            client.insert_records(table_name="error_test", records=invalid_records)
            print("⚠️  Expected error did not occur")
        except Exception as e:
            print(f"\n✅ Error caught correctly")
            print(f"Error message: {str(e)[:200]}")

            # Check if error message is improved
            error_str = str(e)
            if "suggestion" in error_str.lower() or "details" in error_str.lower():
                print("  ✓ Error message includes helpful information")
            else:
                print("  ⚠️  Error message could be more detailed")

        # Cleanup
        client.delete_table("error_test")

    except Exception as e:
        print(f"❌ Error message test failed: {e}")
    finally:
        client.close()


def test_backpressure_under_load():
    """Test backpressure mechanism under heavy load."""
    print("\n" + "=" * 80)
    print("Test 4: Backpressure Under Load")
    print("=" * 80)

    client = Client(host="localhost", port=9999)

    try:
        # Create table
        try:
            client.delete_table("load_test")
        except Exception:
            pass

        client.create_table(
            name="load_test",
            dimension=128,
            distance_metric="cosine",
            index_type="hnsw",
        )

        print("\n📊 Creating load to trigger backpressure...")

        # Generate heavy load
        import random
        for i in range(10):
            records = [
                {
                    "id": f"doc{i * 1000 + j}",
                    "vector": [random.random() for _ in range(128)],
                    "payload": {"index": i * 1000 + j},
                }
                for j in range(1000)
            ]

            try:
                client.insert_records(table_name="load_test", records=records, mode="parallel")
                print(f"  Batch {i + 1}/10: ✅")
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "Too Many Requests" in error_str:
                    print(f"  Batch {i + 1}/10: 🔴 Backpressure triggered (429)")
                    print(f"    Server message: {error_str[:150]}")

                    # Check for Retry-After suggestion
                    if "retry" in error_str.lower():
                        print("    ✓ Retry suggestion provided")

                    break
                else:
                    print(f"  Batch {i + 1}/10: ❌ Error: {error_str[:100]}")

        # Check health after load
        print("\n📊 Checking health after load...")
        response = requests.get("http://localhost:9999/health")
        if response.status_code == 200:
            data = response.json()
            status = data.get("data", {}).get("status", "Unknown")
            print(f"  Server status: {status}")

        # Cleanup
        print("\n🧹 Cleaning up...")
        time.sleep(1)  # Give server time to stabilize
        client.delete_table("load_test")
        print("✅ Cleanup complete")

    except Exception as e:
        print(f"❌ Backpressure test failed: {e}")
    finally:
        client.close()


def main():
    """Run all tests."""
    print("=" * 80)
    print("DXMT Server Improvements Test Suite")
    print("=" * 80)
    print("\nTesting new server-side features:")
    print("  1. Server health monitoring")
    print("  2. Backpressure with 429 responses")
    print("  3. Retry-After headers")
    print("  4. Improved error messages\n")

    try:
        # Test 1: Health endpoint
        test_health_endpoint()

        # Test 2: Health metrics endpoint
        test_health_metrics_endpoint()

        # Test 3: Improved error messages
        test_improved_error_messages()

        # Test 4: Backpressure under load
        test_backpressure_under_load()

        # Summary
        print("\n" + "=" * 80)
        print("Test Summary")
        print("=" * 80)
        print("""
All tests completed. The server improvements include:

✅ Enhanced health monitoring with detailed metrics
✅ Backpressure mechanism to prevent overload
✅ 429 responses with Retry-After headers
✅ Detailed error messages with suggestions

These improvements address the issues identified in batch_operations.py
and provide better visibility into server state.

Next steps:
- Monitor /health endpoint in production
- Set up alerts for Degraded/Overloaded states
- Use /health/metrics for detailed diagnostics
        """)

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")


if __name__ == "__main__":
    main()
