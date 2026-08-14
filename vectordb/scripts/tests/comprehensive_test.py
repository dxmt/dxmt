#!/usr/bin/env python3
"""
Comprehensive functional and performance test suite for DXMT VectorDB
Uses GIST-like synthetic dataset for realistic testing
"""

import requests
import numpy as np
import time
import json
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import sys
from dataclasses import dataclass, asdict
from datetime import datetime

# Configuration
API_URL = "http://localhost:9999/api/v1"
REPORT_FILE = f"/tmp/vectordb_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

@dataclass
class TestResult:
    """Test result container"""
    name: str
    passed: bool
    duration_ms: float
    details: Dict[str, Any]
    error: str = ""

class VectorDBComprehensiveTest:
    """Comprehensive test suite for VectorDB"""

    def __init__(self):
        self.results: List[TestResult] = []
        self.lock = threading.Lock()
        self.test_table = f"test_{int(time.time())}"

    def log(self, message: str, level: str = "INFO"):
        """Log a message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def generate_gist_like_vectors(self, num_vectors: int, dim: int = 960) -> np.ndarray:
        """
        Generate GIST-like synthetic vectors
        GIST descriptors are typically 960-dimensional with specific characteristics
        """
        self.log(f"Generating {num_vectors:,} GIST-like vectors ({dim}D)...")
        np.random.seed(42)

        vectors = []
        for i in range(num_vectors):
            # GIST vectors have specific distribution patterns
            # Using mixture of Gaussians to simulate real GIST features
            vec = np.random.randn(dim) * 0.3
            vec += np.random.exponential(0.5, dim)

            # Normalize to unit length (common preprocessing)
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            vectors.append(vec.astype(np.float32))

            if (i + 1) % 10000 == 0:
                self.log(f"  Generated {i+1:,} vectors...", "DEBUG")

        self.log(f"✓ Generated {len(vectors):,} vectors")
        return np.array(vectors)

    def check_server_health(self) -> TestResult:
        """Test 1: Server health check"""
        self.log("\n" + "="*70)
        self.log("TEST 1: Server Health Check")
        self.log("="*70)

        start = time.time()
        try:
            resp = requests.get(f"{API_URL}/status", timeout=5)
            duration = (time.time() - start) * 1000

            if resp.status_code == 200:
                data = resp.json()
                self.log(f"✓ Server is healthy")
                self.log(f"  Status: {data.get('status', 'N/A')}")
                self.log(f"  Response time: {duration:.2f}ms")

                return TestResult(
                    name="Server Health",
                    passed=True,
                    duration_ms=duration,
                    details=data
                )
            else:
                raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")

        except Exception as e:
            duration = (time.time() - start) * 1000
            self.log(f"✗ Server health check failed: {e}", "ERROR")
            return TestResult(
                name="Server Health",
                passed=False,
                duration_ms=duration,
                details={},
                error=str(e)
            )

    def test_table_operations(self) -> List[TestResult]:
        """Test 2: Table CRUD operations"""
        self.log("\n" + "="*70)
        self.log("TEST 2: Table CRUD Operations")
        self.log("="*70)

        results = []

        # Test 2.1: Create table
        self.log("\n2.1: Create Table")
        start = time.time()
        try:
            payload = {
                "name": self.test_table,
                "dimension": 960,
                "distance_metric": "euclidean",
                "index_type": "hnsw"
            }
            resp = requests.post(f"{API_URL}/tables", json=payload, timeout=10)
            duration = (time.time() - start) * 1000

            if resp.status_code in [200, 201]:
                self.log(f"✓ Table '{self.test_table}' created")
                results.append(TestResult(
                    name="Create Table",
                    passed=True,
                    duration_ms=duration,
                    details={"table_name": self.test_table}
                ))
            else:
                raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.log(f"✗ Failed to create table: {e}", "ERROR")
            results.append(TestResult(
                name="Create Table",
                passed=False,
                duration_ms=duration,
                details={},
                error=str(e)
            ))
            return results  # Cannot continue without table

        # Test 2.2: List tables
        self.log("\n2.2: List Tables")
        start = time.time()
        try:
            resp = requests.get(f"{API_URL}/tables", timeout=5)
            duration = (time.time() - start) * 1000

            if resp.status_code == 200:
                tables = resp.json()
                self.log(f"✓ Listed {len(tables)} table(s)")
                results.append(TestResult(
                    name="List Tables",
                    passed=True,
                    duration_ms=duration,
                    details={"num_tables": len(tables)}
                ))
            else:
                raise Exception(f"HTTP {resp.status_code}")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.log(f"✗ Failed to list tables: {e}", "ERROR")
            results.append(TestResult(
                name="List Tables",
                passed=False,
                duration_ms=duration,
                details={},
                error=str(e)
            ))

        # Test 2.3: Get table info
        self.log("\n2.3: Get Table Info")
        start = time.time()
        try:
            resp = requests.get(f"{API_URL}/tables/{self.test_table}", timeout=5)
            duration = (time.time() - start) * 1000

            if resp.status_code == 200:
                info = resp.json()
                self.log(f"✓ Retrieved table info")
                self.log(f"  Dimension: {info.get('dimension', 'N/A')}")
                self.log(f"  Distance metric: {info.get('distance_metric', 'N/A')}")
                results.append(TestResult(
                    name="Get Table Info",
                    passed=True,
                    duration_ms=duration,
                    details=info
                ))
            else:
                raise Exception(f"HTTP {resp.status_code}")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.log(f"✗ Failed to get table info: {e}", "ERROR")
            results.append(TestResult(
                name="Get Table Info",
                passed=False,
                duration_ms=duration,
                details={},
                error=str(e)
            ))

        return results

    def test_record_operations(self, vectors: np.ndarray) -> List[TestResult]:
        """Test 3: Record CRUD operations"""
        self.log("\n" + "="*70)
        self.log("TEST 3: Record CRUD Operations")
        self.log("="*70)

        results = []
        test_vectors = vectors[:100]  # Use 100 vectors for CRUD tests

        # Test 3.1: Insert single record
        self.log("\n3.1: Insert Single Record")
        start = time.time()
        try:
            payload = {
                "records": [{
                    "id": "test_single",
                    "vector": test_vectors[0].tolist(),
                    "payload": {"type": "single", "index": 0}
                }]
            }
            resp = requests.post(
                f"{API_URL}/tables/{self.test_table}/records",
                json=payload,
                timeout=10
            )
            duration = (time.time() - start) * 1000

            if resp.status_code in [200, 201]:
                self.log(f"✓ Inserted single record ({duration:.2f}ms)")
                results.append(TestResult(
                    name="Insert Single Record",
                    passed=True,
                    duration_ms=duration,
                    details={"num_records": 1}
                ))
            else:
                raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.log(f"✗ Failed to insert single record: {e}", "ERROR")
            results.append(TestResult(
                name="Insert Single Record",
                passed=False,
                duration_ms=duration,
                details={},
                error=str(e)
            ))

        # Test 3.2: Insert batch records
        self.log("\n3.2: Insert Batch Records")
        start = time.time()
        try:
            batch_records = []
            for i, vec in enumerate(test_vectors[1:50]):  # 49 more records
                batch_records.append({
                    "id": f"test_batch_{i}",
                    "vector": vec.tolist(),
                    "payload": {"type": "batch", "index": i}
                })

            resp = requests.post(
                f"{API_URL}/tables/{self.test_table}/records",
                json={"records": batch_records},
                timeout=30
            )
            duration = (time.time() - start) * 1000

            if resp.status_code in [200, 201]:
                throughput = len(batch_records) / (duration / 1000)
                self.log(f"✓ Inserted {len(batch_records)} records ({duration:.2f}ms, {throughput:.0f} rec/s)")
                results.append(TestResult(
                    name="Insert Batch Records",
                    passed=True,
                    duration_ms=duration,
                    details={"num_records": len(batch_records), "throughput_rps": throughput}
                ))
            else:
                raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.log(f"✗ Failed to insert batch records: {e}", "ERROR")
            results.append(TestResult(
                name="Insert Batch Records",
                passed=False,
                duration_ms=duration,
                details={},
                error=str(e)
            ))

        # Test 3.3: Get single record
        self.log("\n3.3: Get Single Record")
        start = time.time()
        try:
            resp = requests.get(
                f"{API_URL}/tables/{self.test_table}/records/test_single",
                timeout=5
            )
            duration = (time.time() - start) * 1000

            if resp.status_code == 200:
                record = resp.json()
                self.log(f"✓ Retrieved single record ({duration:.2f}ms)")
                results.append(TestResult(
                    name="Get Single Record",
                    passed=True,
                    duration_ms=duration,
                    details={"record_id": "test_single"}
                ))
            else:
                raise Exception(f"HTTP {resp.status_code}")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.log(f"✗ Failed to get single record: {e}", "ERROR")
            results.append(TestResult(
                name="Get Single Record",
                passed=False,
                duration_ms=duration,
                details={},
                error=str(e)
            ))

        # Test 3.4: Update record
        self.log("\n3.4: Update Record")
        start = time.time()
        try:
            payload = {
                "vector": test_vectors[50].tolist(),
                "payload": {"type": "updated", "timestamp": time.time()}
            }
            resp = requests.put(
                f"{API_URL}/tables/{self.test_table}/records/test_single",
                json=payload,
                timeout=10
            )
            duration = (time.time() - start) * 1000

            if resp.status_code in [200, 204]:
                self.log(f"✓ Updated record ({duration:.2f}ms)")
                results.append(TestResult(
                    name="Update Record",
                    passed=True,
                    duration_ms=duration,
                    details={"record_id": "test_single"}
                ))
            else:
                raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.log(f"✗ Failed to update record: {e}", "ERROR")
            results.append(TestResult(
                name="Update Record",
                passed=False,
                duration_ms=duration,
                details={},
                error=str(e)
            ))

        # Test 3.5: Delete record
        self.log("\n3.5: Delete Record")
        start = time.time()
        try:
            resp = requests.delete(
                f"{API_URL}/tables/{self.test_table}/records",
                json={"ids": ["test_batch_0"]},
                timeout=10
            )
            duration = (time.time() - start) * 1000

            if resp.status_code in [200, 204]:
                self.log(f"✓ Deleted record ({duration:.2f}ms)")
                results.append(TestResult(
                    name="Delete Record",
                    passed=True,
                    duration_ms=duration,
                    details={"deleted_ids": ["test_batch_0"]}
                ))
            else:
                raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.log(f"✗ Failed to delete record: {e}", "ERROR")
            results.append(TestResult(
                name="Delete Record",
                passed=False,
                duration_ms=duration,
                details={},
                error=str(e)
            ))

        return results

    def test_search_operations(self, vectors: np.ndarray) -> List[TestResult]:
        """Test 4: Search operations"""
        self.log("\n" + "="*70)
        self.log("TEST 4: Search Operations")
        self.log("="*70)

        results = []
        query_vectors = vectors[:10]  # Use 10 vectors as queries

        # First, insert more vectors for meaningful search
        self.log("\n4.0: Preparing search dataset (inserting 1000 vectors)...")
        start = time.time()
        try:
            batch_size = 100
            for i in range(0, min(1000, len(vectors)), batch_size):
                batch = []
                for j, vec in enumerate(vectors[i:i+batch_size]):
                    batch.append({
                        "id": f"search_vec_{i+j}",
                        "vector": vec.tolist(),
                        "payload": {"batch": i//batch_size, "index": i+j, "category": f"cat_{(i+j) % 5}"}
                    })

                resp = requests.post(
                    f"{API_URL}/tables/{self.test_table}/records",
                    json={"records": batch},
                    timeout=30
                )

                if resp.status_code not in [200, 201]:
                    raise Exception(f"HTTP {resp.status_code}")

            duration = (time.time() - start) * 1000
            self.log(f"✓ Inserted search dataset ({duration:.2f}ms)")

            # Wait for indexing
            time.sleep(2)

        except Exception as e:
            self.log(f"✗ Failed to prepare search dataset: {e}", "ERROR")
            return results

        # Test 4.1: Basic vector search
        self.log("\n4.1: Basic Vector Search")
        start = time.time()
        try:
            payload = {
                "vector": query_vectors[0].tolist(),
                "k": 10
            }
            resp = requests.post(
                f"{API_URL}/tables/{self.test_table}/search",
                json=payload,
                timeout=10
            )
            duration = (time.time() - start) * 1000

            if resp.status_code == 200:
                results_data = resp.json()
                num_results = len(results_data.get('results', []))
                self.log(f"✓ Search returned {num_results} results ({duration:.2f}ms)")
                results.append(TestResult(
                    name="Basic Vector Search",
                    passed=True,
                    duration_ms=duration,
                    details={"num_results": num_results, "k": 10}
                ))
            else:
                raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.log(f"✗ Basic search failed: {e}", "ERROR")
            results.append(TestResult(
                name="Basic Vector Search",
                passed=False,
                duration_ms=duration,
                details={},
                error=str(e)
            ))

        # Test 4.2: Search with payload filter
        self.log("\n4.2: Search with Payload Filter")
        start = time.time()
        try:
            payload = {
                "vector": query_vectors[1].tolist(),
                "k": 10,
                "filters": [
                    {
                        "field": "category",
                        "operator": "eq",
                        "value": "cat_1"
                    }
                ]
            }
            resp = requests.post(
                f"{API_URL}/tables/{self.test_table}/search",
                json=payload,
                timeout=10
            )
            duration = (time.time() - start) * 1000

            if resp.status_code == 200:
                results_data = resp.json()
                num_results = len(results_data.get('results', []))
                self.log(f"✓ Filtered search returned {num_results} results ({duration:.2f}ms)")
                results.append(TestResult(
                    name="Search with Filter",
                    passed=True,
                    duration_ms=duration,
                    details={"num_results": num_results, "filter": "category=cat_1"}
                ))
            else:
                raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.log(f"✗ Filtered search failed: {e}", "ERROR")
            results.append(TestResult(
                name="Search with Filter",
                passed=False,
                duration_ms=duration,
                details={},
                error=str(e)
            ))

        # Test 4.3: Search with different K values
        self.log("\n4.3: Search with Different K Values")
        for k in [1, 5, 20, 50]:
            start = time.time()
            try:
                payload = {
                    "vector": query_vectors[2].tolist(),
                    "k": k
                }
                resp = requests.post(
                    f"{API_URL}/tables/{self.test_table}/search",
                    json=payload,
                    timeout=10
                )
                duration = (time.time() - start) * 1000

                if resp.status_code == 200:
                    results_data = resp.json()
                    num_results = len(results_data.get('results', []))
                    self.log(f"✓ Search k={k} returned {num_results} results ({duration:.2f}ms)")
                    results.append(TestResult(
                        name=f"Search K={k}",
                        passed=True,
                        duration_ms=duration,
                        details={"num_results": num_results, "k": k}
                    ))
                else:
                    raise Exception(f"HTTP {resp.status_code}")
            except Exception as e:
                duration = (time.time() - start) * 1000
                self.log(f"✗ Search k={k} failed: {e}", "ERROR")
                results.append(TestResult(
                    name=f"Search K={k}",
                    passed=False,
                    duration_ms=duration,
                    details={},
                    error=str(e)
                ))

        return results

    def test_performance_benchmarks(self, vectors: np.ndarray) -> List[TestResult]:
        """Test 5: Performance benchmarks"""
        self.log("\n" + "="*70)
        self.log("TEST 5: Performance Benchmarks")
        self.log("="*70)

        results = []

        # Test 5.1: Batch insertion throughput
        self.log("\n5.1: Batch Insertion Throughput")
        test_vectors = vectors[1000:5000]  # Use 4000 vectors
        batch_size = 500

        start = time.time()
        inserted = 0
        try:
            for i in range(0, len(test_vectors), batch_size):
                batch = []
                for j, vec in enumerate(test_vectors[i:i+batch_size]):
                    batch.append({
                        "id": f"perf_vec_{i+j}",
                        "vector": vec.tolist(),
                        "payload": {"batch": i//batch_size}
                    })

                resp = requests.post(
                    f"{API_URL}/tables/{self.test_table}/records",
                    json={"records": batch},
                    timeout=30
                )

                if resp.status_code in [200, 201]:
                    inserted += len(batch)

            duration = (time.time() - start) * 1000
            throughput = inserted / (duration / 1000)

            self.log(f"✓ Inserted {inserted} vectors in {duration:.2f}ms")
            self.log(f"  Throughput: {throughput:.0f} vectors/sec")
            results.append(TestResult(
                name="Batch Insertion Throughput",
                passed=True,
                duration_ms=duration,
                details={
                    "num_vectors": inserted,
                    "batch_size": batch_size,
                    "throughput_vps": throughput
                }
            ))
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.log(f"✗ Batch insertion failed: {e}", "ERROR")
            results.append(TestResult(
                name="Batch Insertion Throughput",
                passed=False,
                duration_ms=duration,
                details={},
                error=str(e)
            ))

        # Wait for indexing
        time.sleep(3)

        # Test 5.2: Search QPS and latency
        self.log("\n5.2: Search QPS and Latency")
        query_vectors = vectors[5000:6000]  # 1000 query vectors
        num_threads = 8

        latencies = []
        successful = 0
        failed = 0

        def search_single(vec):
            nonlocal successful, failed
            start = time.time()
            try:
                resp = requests.post(
                    f"{API_URL}/tables/{self.test_table}/search",
                    json={"vector": vec.tolist(), "k": 10},
                    timeout=5
                )
                latency_ms = (time.time() - start) * 1000

                if resp.status_code == 200:
                    with self.lock:
                        latencies.append(latency_ms)
                        successful += 1
                    return True
                else:
                    with self.lock:
                        failed += 1
                    return False
            except:
                with self.lock:
                    failed += 1
                return False

        start = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(search_single, vec) for vec in query_vectors]
            for future in as_completed(futures):
                future.result()

        duration = (time.time() - start) * 1000

        if latencies:
            latencies_sorted = sorted(latencies)
            p50 = latencies_sorted[len(latencies_sorted)//2]
            p95 = latencies_sorted[int(len(latencies_sorted)*0.95)]
            p99 = latencies_sorted[int(len(latencies_sorted)*0.99)]
            avg = np.mean(latencies)
            qps = successful / (duration / 1000)

            self.log(f"✓ Executed {successful} searches in {duration:.2f}ms")
            self.log(f"  QPS: {qps:.0f} queries/sec")
            self.log(f"  Avg latency: {avg:.2f}ms")
            self.log(f"  P50: {p50:.2f}ms, P95: {p95:.2f}ms, P99: {p99:.2f}ms")

            results.append(TestResult(
                name="Search QPS and Latency",
                passed=True,
                duration_ms=duration,
                details={
                    "num_queries": successful,
                    "failed": failed,
                    "qps": qps,
                    "avg_latency_ms": avg,
                    "p50_latency_ms": p50,
                    "p95_latency_ms": p95,
                    "p99_latency_ms": p99
                }
            ))
        else:
            self.log(f"✗ All searches failed", "ERROR")
            results.append(TestResult(
                name="Search QPS and Latency",
                passed=False,
                duration_ms=duration,
                details={"failed": failed},
                error="All searches failed"
            ))

        return results

    def cleanup(self) -> TestResult:
        """Clean up test resources"""
        self.log("\n" + "="*70)
        self.log("CLEANUP: Removing Test Table")
        self.log("="*70)

        start = time.time()
        try:
            resp = requests.delete(f"{API_URL}/tables/{self.test_table}", timeout=10)
            duration = (time.time() - start) * 1000

            if resp.status_code in [200, 204]:
                self.log(f"✓ Deleted test table '{self.test_table}'")
                return TestResult(
                    name="Cleanup",
                    passed=True,
                    duration_ms=duration,
                    details={"table": self.test_table}
                )
            else:
                raise Exception(f"HTTP {resp.status_code}")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.log(f"⚠ Cleanup failed: {e}", "WARN")
            return TestResult(
                name="Cleanup",
                passed=False,
                duration_ms=duration,
                details={},
                error=str(e)
            )

    def generate_report(self):
        """Generate comprehensive test report"""
        self.log("\n" + "="*70)
        self.log("GENERATING TEST REPORT")
        self.log("="*70)

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests

        # Calculate statistics
        total_duration = sum(r.duration_ms for r in self.results)

        # Categorize results
        categories = {
            "Health": [],
            "Table Operations": [],
            "Record Operations": [],
            "Search Operations": [],
            "Performance": [],
            "Cleanup": []
        }

        for result in self.results:
            if "Health" in result.name:
                categories["Health"].append(result)
            elif "Table" in result.name:
                categories["Table Operations"].append(result)
            elif "Record" in result.name or "Insert" in result.name or "Update" in result.name or "Delete" in result.name or "Get" in result.name:
                categories["Record Operations"].append(result)
            elif "Search" in result.name and "QPS" not in result.name:
                categories["Search Operations"].append(result)
            elif "Throughput" in result.name or "QPS" in result.name or "Performance" in result.name:
                categories["Performance"].append(result)
            elif "Cleanup" in result.name:
                categories["Cleanup"].append(result)

        # Print summary
        print("\n" + "="*70)
        print(" TEST SUMMARY")
        print("="*70)
        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        print(f"Total Duration: {total_duration/1000:.2f}s")

        print("\n" + "="*70)
        print(" DETAILED RESULTS")
        print("="*70)

        for category, results in categories.items():
            if results:
                print(f"\n{category}:")
                for result in results:
                    status = "✓" if result.passed else "✗"
                    print(f"  {status} {result.name}: {result.duration_ms:.2f}ms")
                    if result.error:
                        print(f"      Error: {result.error}")
                    if result.details:
                        key_details = {k: v for k, v in result.details.items() if k in ['qps', 'throughput_vps', 'p95_latency_ms', 'avg_latency_ms']}
                        if key_details:
                            for k, v in key_details.items():
                                if isinstance(v, float):
                                    print(f"      {k}: {v:.2f}")
                                else:
                                    print(f"      {k}: {v}")

        # Performance highlights
        print("\n" + "="*70)
        print(" PERFORMANCE HIGHLIGHTS")
        print("="*70)

        for result in self.results:
            if result.passed:
                if 'qps' in result.details:
                    print(f"\n🔍 {result.name}:")
                    print(f"   QPS: {result.details['qps']:,.0f} queries/sec")
                    if 'avg_latency_ms' in result.details:
                        print(f"   Avg Latency: {result.details['avg_latency_ms']:.2f}ms")
                    if 'p95_latency_ms' in result.details:
                        print(f"   P95 Latency: {result.details['p95_latency_ms']:.2f}ms")

                if 'throughput_vps' in result.details:
                    print(f"\n📊 {result.name}:")
                    print(f"   Throughput: {result.details['throughput_vps']:,.0f} vectors/sec")

        # Save JSON report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "duration_ms": total_duration
            },
            "results": [asdict(r) for r in self.results]
        }

        with open(REPORT_FILE, 'w') as f:
            json.dump(report, f, indent=2)

        self.log(f"\n✓ Full report saved to: {REPORT_FILE}")

        return passed_tests == total_tests

    def run(self):
        """Run all tests"""
        self.log("="*70)
        self.log(" DXMT VectorDB Comprehensive Test Suite")
        self.log("="*70)
        self.log(f"API URL: {API_URL}")
        self.log(f"Report: {REPORT_FILE}")

        # Generate test data
        self.log("\nGenerating GIST-like test dataset...")
        vectors = self.generate_gist_like_vectors(num_vectors=10000, dim=960)

        # Run test suites
        self.results.append(self.check_server_health())
        self.results.extend(self.test_table_operations())
        self.results.extend(self.test_record_operations(vectors))
        self.results.extend(self.test_search_operations(vectors))
        self.results.extend(self.test_performance_benchmarks(vectors))
        self.results.append(self.cleanup())

        # Generate report
        all_passed = self.generate_report()

        if all_passed:
            self.log("\n✓ ALL TESTS PASSED", "SUCCESS")
            return 0
        else:
            self.log("\n✗ SOME TESTS FAILED", "ERROR")
            return 1


def main():
    """Main entry point"""
    test_suite = VectorDBComprehensiveTest()
    exit_code = test_suite.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
