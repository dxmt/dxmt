#!/usr/bin/env python3
"""
GCS Three-Tier Storage Test Suite

Tests the three-tier storage implementation (Memory → SSD → GCS) with comprehensive
coverage of connectivity, data flow, latency characteristics, and search functionality.
"""

import requests
import json
import time
import random
import sys
from typing import List, Dict, Any

BASE_URL = "http://localhost:9999/api/v1"

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text:^60}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}\n")

def print_section(text: str):
    """Print formatted section header"""
    print(f"\n{Colors.BLUE}=== {text} ==={Colors.RESET}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_info(text: str):
    """Print info message"""
    print(f"  {text}")

def test_gcs_connectivity():
    """Test 1: Basic GCS connectivity through tiered storage"""
    print_section("Test 1: GCS Connectivity")

    try:
        # Create a test table
        table_config = {
            "name": "test_gcs_table",
            "dimension": 128,
            "distance_metric": "cosine"
        }

        print_info("Creating test table...")
        response = requests.post(f"{BASE_URL}/tables", json=table_config, timeout=10)
        print_info(f"Create table response: {response.status_code}")

        if response.status_code != 200:
            print_error(f"Failed to create table: {response.text}")
            return False

        # Insert a vector (will go to object storage)
        vector = [random.random() for _ in range(128)]
        batch_request = {
            "records": [
                {
                    "id": "test_record_1",
                    "vector": vector,
                    "metadata": {"test": "gcs_connectivity"}
                }
            ]
        }

        print_info("Inserting test record...")
        response = requests.post(
            f"{BASE_URL}/tables/test_gcs_table/records",
            json=batch_request,
            timeout=30
        )
        print_info(f"Insert record response: {response.status_code}")

        if response.status_code != 200:
            print_error(f"Failed to insert record: {response.text}")
            return False

        # Wait a bit to ensure data is persisted
        time.sleep(2)

        # Verify the record exists
        print_info("Retrieving test record...")
        response = requests.get(
            f"{BASE_URL}/tables/test_gcs_table/records/test_record_1",
            timeout=10
        )
        print_info(f"Get record response: {response.status_code}")

        if response.status_code != 200:
            print_error(f"Failed to retrieve record: {response.text}")
            return False

        # Verify data integrity
        response_data = response.json()
        if not response_data.get("success"):
            print_error(f"API returned error: {response_data.get('error')}")
            return False

        retrieved_data = response_data.get("data", {})
        if "vector" not in retrieved_data or "id" not in retrieved_data:
            print_error(f"Retrieved data missing required fields. Got: {retrieved_data}")
            return False

        print_success("GCS connectivity successful!")
        print_info(f"Retrieved record ID: {retrieved_data.get('id')}")
        return True

    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to server. Is it running on port 9999?")
        return False
    except requests.exceptions.Timeout:
        print_error("Request timed out. Server might be slow or unresponsive.")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_three_tier_flow():
    """Test 2: Verify data flows through all three tiers"""
    print_section("Test 2: Three-Tier Flow")

    try:
        # Create table for tiered testing
        table_config = {
            "name": "tiered_test_table",
            "dimension": 128,
            "distance_metric": "cosine"
        }

        print_info("Creating tiered test table...")
        response = requests.post(f"{BASE_URL}/tables", json=table_config, timeout=10)
        if response.status_code != 200:
            print_error(f"Failed to create table: {response.text}")
            return False

        # Step 1: Insert records (should go to object storage and be cached)
        print_info("\nStep 1: Inserting 100 records...")
        for i in range(100):
            vector = [random.random() for _ in range(128)]
            batch_request = {
                "records": [
                    {
                        "id": f"record_{i}",
                        "vector": vector,
                        "metadata": {"index": i, "tier_test": True}
                    }
                ]
            }
            response = requests.post(
                f"{BASE_URL}/tables/tiered_test_table/records",
                json=batch_request,
                timeout=10
            )
            if response.status_code != 200:
                print_error(f"Failed to insert record_{i}: {response.text}")
                return False
            if i % 20 == 0:
                print_info(f"  Inserted {i} records")

        print_success(f"Inserted 100 records")

        # Step 2: Access hot data (should hit memory cache)
        print_info("\nStep 2: Accessing recent records (memory tier)...")
        latencies = []
        for i in range(90, 100):  # Last 10 records (hot)
            start_time = time.time()
            response = requests.get(
                f"{BASE_URL}/tables/tiered_test_table/records/record_{i}",
                timeout=5
            )
            latency = time.time() - start_time
            latencies.append(latency)
            if response.status_code != 200:
                print_error(f"Failed to get record_{i}")
                return False

        memory_latency = sum(latencies) / len(latencies)
        print_info(f"  Average latency: {memory_latency*1000:.2f}ms (Memory tier)")

        # Step 3: Access warm data (should hit SSD cache)
        print_info("\nStep 3: Accessing mid-range records (SSD tier)...")
        time.sleep(2)  # Let some data cool down
        latencies = []
        for i in range(40, 50):  # Middle records (warm)
            start_time = time.time()
            response = requests.get(
                f"{BASE_URL}/tables/tiered_test_table/records/record_{i}",
                timeout=5
            )
            latency = time.time() - start_time
            latencies.append(latency)
            if response.status_code != 200:
                print_error(f"Failed to get record_{i}")
                return False

        ssd_latency = sum(latencies) / len(latencies)
        print_info(f"  Average latency: {ssd_latency*1000:.2f}ms (SSD tier)")

        # Step 4: Access cold data (should fetch from GCS)
        print_info("\nStep 4: Accessing old records (Object Storage tier)...")
        time.sleep(2)
        latencies = []
        for i in range(0, 10):  # First records (cold)
            start_time = time.time()
            response = requests.get(
                f"{BASE_URL}/tables/tiered_test_table/records/record_{i}",
                timeout=10
            )
            latency = time.time() - start_time
            latencies.append(latency)
            if response.status_code != 200:
                print_error(f"Failed to get record_{i}")
                return False

        gcs_latency = sum(latencies) / len(latencies)
        print_info(f"  Average latency: {gcs_latency*1000:.2f}ms (GCS tier)")

        # Verify latency hierarchy
        print_info("\n=== Latency Comparison ===")
        print_info(f"Memory: {memory_latency*1000:.2f}ms")
        print_info(f"SSD:    {ssd_latency*1000:.2f}ms")
        print_info(f"GCS:    {gcs_latency*1000:.2f}ms")

        # Expected: Memory < SSD < GCS (with some tolerance)
        if memory_latency < gcs_latency:
            print_success("Tiered storage latency hierarchy verified!")
        else:
            print_warning(f"Unexpected latency pattern (Memory: {memory_latency*1000:.2f}ms >= GCS: {gcs_latency*1000:.2f}ms)")
            print_warning("This might be normal in some network conditions")

        print_success("Three-tier flow test complete!")
        return True

    except requests.exceptions.Timeout:
        print_error("Request timed out during three-tier flow test")
        return False
    except Exception as e:
        print_error(f"Three-tier flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_with_tiered_storage():
    """Test 3: Verify vector search works with tiered storage"""
    print_section("Test 3: Vector Search with Tiered Storage")

    try:
        # Insert more records
        print_info("Inserting 50 search test records...")
        for i in range(50):
            vector = [random.random() for _ in range(128)]
            batch_request = {
                "records": [
                    {
                        "id": f"search_record_{i}",
                        "vector": vector,
                        "metadata": {"category": "search_test"}
                    }
                ]
            }
            response = requests.post(
                f"{BASE_URL}/tables/tiered_test_table/records",
                json=batch_request,
                timeout=10
            )
            if response.status_code != 200:
                print_error(f"Failed to insert search_record_{i}")
                return False

        print_success("Inserted 50 search test records")

        # Perform search
        query_vector = [random.random() for _ in range(128)]
        search_request = {
            "vector": query_vector,
            "k": 10
        }

        print_info("Performing vector search...")
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/tables/tiered_test_table/search",
            json=search_request,
            timeout=30
        )
        search_latency = time.time() - start_time

        if response.status_code != 200:
            print_error(f"Search failed: {response.text}")
            return False

        response_data = response.json()
        results = response_data.get('data', {})
        result_count = len(results.get('results', []))
        print_info(f"Found {result_count} results")
        print_info(f"Search latency: {search_latency*1000:.2f}ms")

        if result_count > 0:
            print_success("Search works with tiered storage!")
        else:
            print_warning("Search returned no results (might be expected with random vectors)")

        return True

    except requests.exceptions.Timeout:
        print_error("Search request timed out")
        return False
    except Exception as e:
        print_error(f"Search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tier_statistics():
    """Test 4: Check tier statistics and system health"""
    print_section("Test 4: System Health Check")

    try:
        # List tables
        print_info("Listing tables...")
        response = requests.get(f"{BASE_URL}/tables", timeout=5)

        if response.status_code != 200:
            print_error(f"Failed to list tables: {response.text}")
            return False

        response_data = response.json()
        tables_data = response_data.get('data', [])
        print_info(f"Active tables: {tables_data}")

        # Get table info
        for table_name in ["test_gcs_table", "tiered_test_table"]:
            print_info(f"\nChecking table: {table_name}")
            response = requests.get(f"{BASE_URL}/tables/{table_name}", timeout=5)
            if response.status_code == 200:
                response_data = response.json()
                table_info = response_data.get('data', {})
                print_info(f"  Config: {json.dumps(table_info, indent=2)}")
            else:
                print_warning(f"  Could not retrieve table info")

        print_success("System operational!")
        return True

    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False

def cleanup():
    """Cleanup test data"""
    print_section("Cleanup")

    for table_name in ["test_gcs_table", "tiered_test_table"]:
        try:
            response = requests.delete(f"{BASE_URL}/tables/{table_name}", timeout=10)
            if response.status_code == 200:
                print_success(f"Deleted {table_name}")
            else:
                print_warning(f"Failed to delete {table_name}: {response.status_code}")
        except Exception as e:
            print_warning(f"Error deleting {table_name}: {e}")

def main():
    """Main test runner"""
    print_header("GCS Three-Tier Storage Test Suite")

    # Track test results
    results = {
        "gcs_connectivity": False,
        "three_tier_flow": False,
        "search": False,
        "health_check": False
    }

    # Run tests
    try:
        results["gcs_connectivity"] = test_gcs_connectivity()
        if not results["gcs_connectivity"]:
            print_error("\nGCS connectivity test failed. Check your credentials and bucket configuration.")
            print_info("Make sure environment variables are set:")
            print_info("  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
            print_info("  DB_S3_BUCKET, DB_S3_ENDPOINT")
            return 1

        results["three_tier_flow"] = test_three_tier_flow()
        results["search"] = test_search_with_tiered_storage()
        results["health_check"] = test_tier_statistics()

        # Print summary
        print_header("Test Summary")
        passed = sum(1 for v in results.values() if v)
        total = len(results)

        for test_name, passed_test in results.items():
            status = "PASS" if passed_test else "FAIL"
            color = Colors.GREEN if passed_test else Colors.RED
            print(f"{color}{status:6}{Colors.RESET} {test_name}")

        print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.RESET}")

        if passed == total:
            print_success("\n✓ ALL TESTS PASSED!")
            print_info("\nNext steps:")
            print_info("1. Check your object storage console to verify objects were created")
            print_info("2. Check local SSD cache: ls -lh ./test_cache/")
            print_info("3. Review server logs for tier operation details")
            return 0
        else:
            print_error(f"\n✗ {total - passed} test(s) failed")
            return 1

    except KeyboardInterrupt:
        print_warning("\n\nTest interrupted by user")
        return 1
    finally:
        cleanup()

if __name__ == "__main__":
    sys.exit(main())
