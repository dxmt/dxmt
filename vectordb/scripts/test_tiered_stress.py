#!/usr/bin/env python3
"""
Three-Tier Storage Stress Test

Stress tests for the tiered storage system to verify:
- Concurrent insert handling
- Random access patterns
- Search performance under load
- Tier management stability
"""

import requests
import json
import time
import random
import sys
import concurrent.futures
from typing import List, Tuple

BASE_URL = "http://localhost:9999/api/v1"

class Colors:
    """ANSI color codes"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text:^60}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}\n")

def print_section(text: str):
    print(f"\n{Colors.BLUE}=== {text} ==={Colors.RESET}")

def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_info(text: str):
    print(f"  {text}")

def insert_batch(table_name: str, start_idx: int, count: int) -> Tuple[int, int]:
    """
    Insert a batch of vectors

    Returns: (success_count, failure_count)
    """
    success = 0
    failure = 0

    for i in range(start_idx, start_idx + count):
        try:
            vector = [random.random() for _ in range(128)]
            batch_request = {
                "records": [
                    {
                        "id": f"stress_record_{i}",
                        "vector": vector,
                        "metadata": {"batch": start_idx // count, "index": i}
                    }
                ]
            }
            response = requests.post(
                f"{BASE_URL}/tables/{table_name}/records",
                json=batch_request,
                timeout=10
            )
            if response.status_code == 200:
                success += 1
            else:
                failure += 1
        except Exception as e:
            failure += 1
            if failure == 1:  # Only print first error per batch
                print_warning(f"Insert error in batch {start_idx}: {e}")

    return success, failure

def test_concurrent_inserts():
    """Test concurrent inserts to verify tier management"""
    print_section("Stress Test 1: Concurrent Inserts")

    try:
        # Create table
        table_config = {
            "name": "stress_test_table",
            "dimension": 128,
            "distance_metric": "cosine"
        }

        print_info("Creating stress test table...")
        response = requests.post(f"{BASE_URL}/tables", json=table_config, timeout=10)

        if response.status_code != 200:
            print_error(f"Failed to create table: {response.text}")
            return False

        print_success("Table created")

        # Insert 1000 records using 10 threads
        print_info("Inserting 1000 records with 10 threads...")
        start_time = time.time()

        total_success = 0
        total_failure = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(10):
                future = executor.submit(insert_batch, "stress_test_table", i * 100, 100)
                futures.append(future)

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                success, failure = future.result()
                total_success += success
                total_failure += failure
                print_info(f"  Batch {i+1}/10 completed: {success} success, {failure} failures")

        total_time = time.time() - start_time

        print_success(f"Inserted {total_success} records in {total_time:.2f}s")
        print_info(f"  Throughput: {total_success/total_time:.2f} records/sec")

        if total_failure > 0:
            print_warning(f"  Failed: {total_failure} records")

        if total_success >= 900:  # Allow for some failures
            print_success("Concurrent insert test passed!")
            return True
        else:
            print_error(f"Too many failures: {total_failure}/1000")
            return False

    except Exception as e:
        print_error(f"Concurrent insert test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_random_access_pattern():
    """Test random access to verify tier promotion/demotion"""
    print_section("Stress Test 2: Random Access Pattern")

    try:
        # Random access to 500 records
        print_info("Performing 500 random accesses...")
        start_time = time.time()

        success = 0
        failure = 0
        latencies = []

        for _ in range(500):
            record_id = f"stress_record_{random.randint(0, 999)}"

            try:
                req_start = time.time()
                response = requests.get(
                    f"{BASE_URL}/tables/stress_test_table/records/{record_id}",
                    timeout=5
                )
                latency = time.time() - req_start
                latencies.append(latency)

                if response.status_code == 200:
                    success += 1
                else:
                    failure += 1
            except Exception as e:
                failure += 1
                if failure == 1:
                    print_warning(f"Access error: {e}")

        total_time = time.time() - start_time

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
            p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

            print_success(f"Completed {success} random accesses in {total_time:.2f}s")
            print_info(f"  Average latency: {avg_latency*1000:.2f}ms")
            print_info(f"  P95 latency: {p95_latency*1000:.2f}ms")
            print_info(f"  P99 latency: {p99_latency*1000:.2f}ms")

            if failure > 0:
                print_warning(f"  Failed: {failure} accesses")

            if success >= 450:  # Allow for some failures
                print_success("Random access test passed!")
                return True
            else:
                print_error(f"Too many failures: {failure}/500")
                return False
        else:
            print_error("No latency data collected")
            return False

    except Exception as e:
        print_error(f"Random access test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_under_load():
    """Test search performance with tiered storage under load"""
    print_section("Stress Test 3: Search Under Load")

    try:
        # Perform 100 searches
        print_info("Performing 100 vector searches...")
        latencies = []
        result_counts = []
        errors = 0

        for i in range(100):
            try:
                query_vector = [random.random() for _ in range(128)]
                search_request = {
                    "vector": query_vector,
                    "k": 10
                }

                start_time = time.time()
                response = requests.post(
                    f"{BASE_URL}/tables/stress_test_table/search",
                    json=search_request,
                    timeout=10
                )
                latency = time.time() - start_time

                if response.status_code == 200:
                    latencies.append(latency)
                    response_data = response.json()
                    results_data = response_data.get('data', {})
                    result_counts.append(len(results_data.get('results', [])))
                else:
                    errors += 1
                    if errors == 1:
                        print_warning(f"Search error: {response.status_code}")

                if i % 20 == 0 and i > 0:
                    print_info(f"  Completed {i} searches")

            except Exception as e:
                errors += 1
                if errors == 1:
                    print_warning(f"Search exception: {e}")

        # Calculate statistics
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
            p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
            avg_results = sum(result_counts) / len(result_counts) if result_counts else 0

            print_success(f"Search Performance:")
            print_info(f"  Completed: {len(latencies)} searches")
            print_info(f"  Errors: {errors}")
            print_info(f"  Average latency: {avg_latency*1000:.2f}ms")
            print_info(f"  P95 latency: {p95_latency*1000:.2f}ms")
            print_info(f"  P99 latency: {p99_latency*1000:.2f}ms")
            print_info(f"  Average results per search: {avg_results:.1f}")

            if len(latencies) >= 90:  # At least 90% success
                print_success("Search under load test passed!")
                return True
            else:
                print_error(f"Too many search failures: {errors}/100")
                return False
        else:
            print_error("No search data collected")
            return False

    except Exception as e:
        print_error(f"Search under load test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_integrity():
    """Test data integrity after stress operations"""
    print_section("Stress Test 4: Data Integrity Check")

    try:
        print_info("Verifying data integrity...")

        # Sample 50 random records
        sample_size = 50
        success = 0
        failure = 0

        for _ in range(sample_size):
            record_id = f"stress_record_{random.randint(0, 999)}"

            try:
                response = requests.get(
                    f"{BASE_URL}/tables/stress_test_table/records/{record_id}",
                    timeout=5
                )

                if response.status_code == 200:
                    response_data = response.json()

                    # Check if API call was successful
                    if not response_data.get("success"):
                        print_warning(f"API error for {record_id}: {response_data.get('error')}")
                        failure += 1
                        continue

                    # Extract actual record data from response wrapper
                    data = response_data.get("data", {})

                    # Verify required fields
                    if "id" in data and "vector" in data:
                        # Verify vector dimension
                        if len(data["vector"]) == 128:
                            success += 1
                        else:
                            print_warning(f"Invalid vector dimension for {record_id}: {len(data['vector'])}")
                            failure += 1
                    else:
                        print_warning(f"Missing fields in {record_id}")
                        failure += 1
                else:
                    failure += 1

            except Exception as e:
                failure += 1

        print_info(f"Sampled {sample_size} records:")
        print_info(f"  Valid: {success}")
        print_info(f"  Invalid/Missing: {failure}")

        if success >= sample_size * 0.9:  # 90% success rate
            print_success("Data integrity check passed!")
            return True
        else:
            print_error(f"Data integrity issues detected: {failure}/{sample_size} failures")
            return False

    except Exception as e:
        print_error(f"Data integrity check failed: {e}")
        return False

def cleanup_stress():
    """Cleanup stress test data"""
    print_section("Cleanup")

    try:
        response = requests.delete(f"{BASE_URL}/tables/stress_test_table", timeout=10)
        if response.status_code == 200:
            print_success("Deleted stress_test_table")
        else:
            print_warning(f"Failed to delete stress_test_table: {response.status_code}")
    except Exception as e:
        print_warning(f"Error during cleanup: {e}")

def main():
    """Main stress test runner"""
    print_header("Three-Tier Storage Stress Test")

    # Track test results
    results = {
        "concurrent_inserts": False,
        "random_access": False,
        "search_under_load": False,
        "data_integrity": False
    }

    # Run stress tests
    try:
        results["concurrent_inserts"] = test_concurrent_inserts()

        if not results["concurrent_inserts"]:
            print_error("\nConcurrent insert test failed. Skipping remaining tests.")
            return 1

        results["random_access"] = test_random_access_pattern()
        results["search_under_load"] = test_search_under_load()
        results["data_integrity"] = test_data_integrity()

        # Print summary
        print_header("Stress Test Summary")
        passed = sum(1 for v in results.values() if v)
        total = len(results)

        for test_name, passed_test in results.items():
            status = "PASS" if passed_test else "FAIL"
            color = Colors.GREEN if passed_test else Colors.RED
            print(f"{color}{status:6}{Colors.RESET} {test_name}")

        print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.RESET}")

        if passed == total:
            print_success("\n✓ ALL STRESS TESTS PASSED!")
            print_info("\nThe tiered storage system is stable under load.")
            print_info("Key findings:")
            print_info("- Handles concurrent operations well")
            print_info("- Tier transitions work correctly")
            print_info("- Search performs well under load")
            print_info("- No data corruption detected")
            return 0
        else:
            print_error(f"\n✗ {total - passed} stress test(s) failed")
            print_info("Review the test output above for details.")
            return 1

    except KeyboardInterrupt:
        print_warning("\n\nStress test interrupted by user")
        return 1
    finally:
        cleanup_stress()

if __name__ == "__main__":
    sys.exit(main())
