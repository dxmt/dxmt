"""Improved diagnostic script to identify server bottlenecks and issues.

This enhanced version includes:
1. Performance metrics collection (latency, throughput)
2. Load gradient testing (find breaking points)
3. Concurrent insert testing
4. Improved diagnostic logic (handles all-pass scenarios)
5. JSON report output option

Usage:
    python3 diagnose_server_improved.py
    python3 diagnose_server_improved.py --json-output report.json
"""

import time
import random
import json
import statistics
import threading
import argparse
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from dxmt import Client


class PerformanceMetrics:
    """Track and calculate performance metrics."""

    def __init__(self):
        self.latencies = []
        self.start_time = None
        self.end_time = None
        self.record_count = 0

    def start(self):
        """Start timing."""
        self.start_time = time.time()

    def end(self):
        """End timing."""
        self.end_time = time.time()

    def add_latency(self, latency: float):
        """Add a latency measurement."""
        self.latencies.append(latency)

    def add_records(self, count: int):
        """Add record count."""
        self.record_count += count

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.latencies:
            return {}

        duration = self.end_time - self.start_time if self.end_time else 0
        throughput = self.record_count / duration if duration > 0 else 0

        return {
            "latency_p50": statistics.median(self.latencies),
            "latency_p95": statistics.quantiles(self.latencies, n=20)[18] if len(self.latencies) >= 20 else max(self.latencies),
            "latency_p99": statistics.quantiles(self.latencies, n=100)[98] if len(self.latencies) >= 100 else max(self.latencies),
            "latency_avg": statistics.mean(self.latencies),
            "latency_min": min(self.latencies),
            "latency_max": max(self.latencies),
            "throughput": throughput,
            "total_records": self.record_count,
            "total_duration": duration,
        }


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


def test_insert_after_heavy_load(client: Client, table_name: str, dimension: int) -> Tuple[bool, Dict]:
    """Test if inserts fail after heavy load."""
    print("\n" + "=" * 80)
    print("TEST 1: Insert After Heavy Load")
    print("=" * 80)

    metrics = PerformanceMetrics()

    # Heavy load: Insert 10k records
    print("\nStep 1: Creating heavy load (10,000 records)...")
    records = generate_records(1, 10000, dimension)

    metrics.start()
    chunk_size = 1000
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        chunk_start = time.time()
        client.insert_records(table_name=table_name, records=chunk)
        chunk_duration = time.time() - chunk_start
        metrics.add_latency(chunk_duration)
        metrics.add_records(len(chunk))
        print(f"  Inserted chunk {i // chunk_size + 1}/10 ({chunk_duration:.3f}s)")

    metrics.end()
    print("✅ Heavy load completed")

    # Immediate insert (this should replicate the issue)
    print("\nStep 2: Immediate insert (50 records)...")
    test_records = generate_records(10001, 50, dimension)

    test_success = False
    test_attempts = []

    for attempt in range(3):
        try:
            start = time.time()
            client.insert_records(table_name=table_name, records=test_records)
            duration = time.time() - start
            print(f"  ✅ Insert succeeded (attempt {attempt + 1}, {duration:.4f}s)")
            test_success = True
            test_attempts.append({"attempt": attempt + 1, "success": True, "duration": duration})
            break
        except Exception as e:
            duration = time.time() - start
            print(f"  ❌ Insert failed (attempt {attempt + 1}): {str(e)[:80]}")
            test_attempts.append({"attempt": attempt + 1, "success": False, "error": str(e)[:80], "duration": duration})
            if attempt < 2:
                wait = (attempt + 1) * 0.5
                print(f"     Waiting {wait}s before retry...")
                time.sleep(wait)

    return test_success, {
        "heavy_load_metrics": metrics.get_summary(),
        "test_insert_attempts": test_attempts,
        "test_insert_success": test_success,
    }


def test_load_gradient(client: Client, table_name: str, dimension: int) -> Dict:
    """Test different load levels to find performance characteristics."""
    print("\n" + "=" * 80)
    print("TEST 2: Load Gradient Analysis")
    print("=" * 80)
    print("\nTesting performance at different load levels...")

    load_levels = [1000, 5000, 10000, 20000]
    results = {}

    for load_size in load_levels:
        print(f"\n  Testing {load_size} records...")

        # Recreate table for clean test
        try:
            client.delete_table(table_name)
            time.sleep(0.5)
        except Exception:
            pass

        client.create_table(
            name=table_name,
            dimension=dimension,
            distance_metric="cosine",
            index_type="hnsw",
        )

        metrics = PerformanceMetrics()
        records = generate_records(1, load_size, dimension)

        metrics.start()
        chunk_size = 1000
        success = True

        try:
            for i in range(0, len(records), chunk_size):
                chunk = records[i:i + chunk_size]
                chunk_start = time.time()
                client.insert_records(table_name=table_name, records=chunk)
                chunk_duration = time.time() - chunk_start
                metrics.add_latency(chunk_duration)
                metrics.add_records(len(chunk))

            metrics.end()
            print(f"    ✅ Completed in {metrics.end_time - metrics.start_time:.2f}s")

        except Exception as e:
            metrics.end()
            print(f"    ❌ Failed: {str(e)[:60]}")
            success = False

        summary = metrics.get_summary()
        summary["success"] = success
        results[load_size] = summary

    return results


def test_concurrent_inserts(client: Client, table_name: str, dimension: int) -> Dict:
    """Test concurrent insert performance."""
    print("\n" + "=" * 80)
    print("TEST 3: Concurrent Insert Performance")
    print("=" * 80)

    # Recreate table
    try:
        client.delete_table(table_name)
        time.sleep(0.5)
    except Exception:
        pass

    client.create_table(
        name=table_name,
        dimension=dimension,
        distance_metric="cosine",
        index_type="hnsw",
    )

    num_threads = 4
    records_per_thread = 1000

    print(f"\nTesting {num_threads} concurrent clients, {records_per_thread} records each...")

    results = {"threads": [], "errors": []}

    def worker(thread_id: int):
        """Worker function for concurrent inserts."""
        try:
            worker_client = Client(host="localhost", port=9999)
            records = generate_records(thread_id * 10000 + 1, records_per_thread, dimension)

            start = time.time()
            worker_client.insert_records(table_name=table_name, records=records)
            duration = time.time() - start

            results["threads"].append({
                "thread_id": thread_id,
                "success": True,
                "duration": duration,
                "throughput": records_per_thread / duration,
            })

            worker_client.close()
            print(f"  Thread {thread_id}: ✅ {duration:.3f}s ({records_per_thread/duration:.1f} rec/s)")

        except Exception as e:
            results["errors"].append({
                "thread_id": thread_id,
                "error": str(e)[:100],
            })
            print(f"  Thread {thread_id}: ❌ {str(e)[:60]}")

    start_time = time.time()
    threads = []

    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    total_duration = time.time() - start_time

    results["total_duration"] = total_duration
    results["success_rate"] = len(results["threads"]) / num_threads

    if results["threads"]:
        results["avg_throughput"] = statistics.mean([t["throughput"] for t in results["threads"]])
        results["total_throughput"] = sum(t["throughput"] for t in results["threads"])

    print(f"\n  Total duration: {total_duration:.3f}s")
    print(f"  Success rate: {results['success_rate']*100:.1f}%")
    if results["threads"]:
        print(f"  Combined throughput: {results['total_throughput']:.1f} rec/s")

    return results


def test_insert_with_delays(client: Client, table_name: str, dimension: int) -> Tuple[bool, Dict]:
    """Test if adding delays prevents failures."""
    print("\n" + "=" * 80)
    print("TEST 4: Insert With Delays Between Operations")
    print("=" * 80)

    # Create table
    try:
        client.delete_table(table_name)
        time.sleep(0.5)
    except Exception:
        pass

    client.create_table(
        name=table_name,
        dimension=dimension,
        distance_metric="cosine",
        index_type="hnsw",
    )

    metrics = PerformanceMetrics()

    # Heavy load with delays
    print("\nInserting 10,000 records with 1s delay between chunks...")
    metrics.start()

    for i in range(10):
        records = generate_records(i * 1000 + 1, 1000, dimension)
        chunk_start = time.time()
        client.insert_records(table_name=table_name, records=records)
        chunk_duration = time.time() - chunk_start
        metrics.add_latency(chunk_duration)
        metrics.add_records(1000)
        print(f"  Chunk {i + 1}/10 ({chunk_duration:.3f}s)")
        time.sleep(1.0)  # 1 second delay

    metrics.end()
    print("✅ Heavy load with delays completed")

    # Immediate insert
    print("\nImmediate insert (50 records)...")
    test_records = generate_records(10001, 50, dimension)
    try:
        start = time.time()
        client.insert_records(table_name=table_name, records=test_records)
        duration = time.time() - start
        print(f"  ✅ Insert succeeded ({duration:.4f}s)")
        success = True
    except Exception as e:
        duration = time.time() - start
        print(f"  ❌ Insert failed: {str(e)[:80]}")
        success = False

    return success, {
        "heavy_load_metrics": metrics.get_summary(),
        "test_insert_success": success,
        "test_insert_duration": duration,
    }


def analyze_results(test_results: Dict) -> Dict[str, Any]:
    """Analyze test results and generate diagnosis."""
    diagnosis = {
        "status": "healthy",
        "issues": [],
        "recommendations": [],
        "metrics": {},
    }

    # Analyze Test 1: Insert after heavy load
    test1 = test_results.get("test1", {})
    if test1.get("test_insert_success"):
        diagnosis["metrics"]["test1_status"] = "passed"
        # Check if retries were needed
        attempts = test1.get("test_insert_attempts", [])
        if len(attempts) > 1:
            diagnosis["issues"].append("Insert required retries after heavy load")
            diagnosis["recommendations"].append("Implement exponential backoff retry logic in production")
    else:
        diagnosis["status"] = "degraded"
        diagnosis["issues"].append("Insert failed after heavy load (critical issue)")
        diagnosis["recommendations"].append("Add delays between large batch operations")

    # Analyze Test 2: Load gradient
    test2 = test_results.get("test2", {})
    if test2:
        failed_loads = [size for size, result in test2.items() if not result.get("success")]
        if failed_loads:
            diagnosis["status"] = "degraded"
            diagnosis["issues"].append(f"System fails at load levels: {failed_loads}")
            diagnosis["recommendations"].append(f"Limit batch size to below {min(failed_loads)} records")

        # Check throughput degradation
        throughputs = {size: result.get("throughput", 0) for size, result in test2.items() if result.get("success")}
        if len(throughputs) >= 2:
            throughput_list = list(throughputs.values())
            if throughput_list[-1] < throughput_list[0] * 0.5:  # 50% degradation
                diagnosis["issues"].append("Significant throughput degradation at higher loads")
                diagnosis["recommendations"].append("Consider adding more server resources or sharding")

        diagnosis["metrics"]["load_gradient"] = test2

    # Analyze Test 3: Concurrent inserts
    test3 = test_results.get("test3", {})
    if test3:
        success_rate = test3.get("success_rate", 0)
        if success_rate < 1.0:
            diagnosis["status"] = "degraded"
            diagnosis["issues"].append(f"Concurrent insert success rate: {success_rate*100:.1f}% (below 100%)")
            diagnosis["recommendations"].append("Server may have concurrency issues; consider connection pooling limits")

        diagnosis["metrics"]["concurrent_throughput"] = test3.get("total_throughput", 0)

    # Analyze Test 4: Delays
    test4 = test_results.get("test4", {})
    if test4 and not test1.get("test_insert_success") and test4.get("test_insert_success"):
        diagnosis["issues"].append("Delays resolve insert failures")
        diagnosis["recommendations"].append("Add 1-2s delays between large batch operations in production")

    # Overall assessment
    if not diagnosis["issues"]:
        diagnosis["status"] = "healthy"
        diagnosis["recommendations"].append("System appears stable under current test loads")
        diagnosis["recommendations"].append("Consider periodic load testing to ensure continued stability")

    return diagnosis


def print_diagnostic_summary(test_results: Dict, diagnosis: Dict):
    """Print comprehensive diagnostic summary."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    # Test results
    print("\nTest Results:")
    test1 = test_results.get("test1", {})
    print(f"  1. Insert After Heavy Load: {'✅ PASSED' if test1.get('test_insert_success') else '❌ FAILED'}")

    test2 = test_results.get("test2", {})
    if test2:
        passed = sum(1 for r in test2.values() if r.get("success"))
        print(f"  2. Load Gradient: {passed}/{len(test2)} load levels passed")

    test3 = test_results.get("test3", {})
    if test3:
        success_rate = test3.get("success_rate", 0) * 100
        print(f"  3. Concurrent Inserts: {success_rate:.1f}% success rate")

    test4 = test_results.get("test4", {})
    print(f"  4. Insert With Delays: {'✅ PASSED' if test4.get('test_insert_success') else '❌ FAILED'}")

    # Performance metrics
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)

    if test1 and "heavy_load_metrics" in test1:
        metrics = test1["heavy_load_metrics"]
        print("\nTest 1 - Heavy Load Performance:")
        print(f"  Throughput: {metrics.get('throughput', 0):.1f} records/sec")
        print(f"  Latency P50: {metrics.get('latency_p50', 0):.4f}s")
        print(f"  Latency P95: {metrics.get('latency_p95', 0):.4f}s")
        print(f"  Latency P99: {metrics.get('latency_p99', 0):.4f}s")

    if test2:
        print("\nTest 2 - Load Gradient Performance:")
        for size, metrics in sorted(test2.items()):
            if metrics.get("success"):
                print(f"  {size:,} records: {metrics.get('throughput', 0):.1f} rec/s")

    if test3:
        print("\nTest 3 - Concurrent Performance:")
        print(f"  Combined throughput: {test3.get('total_throughput', 0):.1f} rec/s")
        print(f"  Average per thread: {test3.get('avg_throughput', 0):.1f} rec/s")

    # Diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    status_icon = "✅" if diagnosis["status"] == "healthy" else "⚠️"
    print(f"\n{status_icon} System Status: {diagnosis['status'].upper()}")

    if diagnosis["issues"]:
        print("\nIssues Identified:")
        for issue in diagnosis["issues"]:
            print(f"  • {issue}")
    else:
        print("\n✅ No critical issues detected")
        print("   Server appears stable under tested load conditions")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if diagnosis["recommendations"]:
        print()
        for i, rec in enumerate(diagnosis["recommendations"], 1):
            print(f"{i}. {rec}")

    # Additional context
    print("\n" + "=" * 80)
    print("ADDITIONAL CONTEXT")
    print("=" * 80)
    print("""
For production deployments:
- Monitor server logs for HNSW index building or memory pressure
- Consider rate limiting in client SDK for large batch operations
- Use separate tables for different workload types
- Schedule heavy operations during low-traffic periods
- Implement health checks and circuit breakers
- Set up alerting for throughput degradation
    """)


def save_json_report(test_results: Dict, diagnosis: Dict, output_file: str):
    """Save diagnostic report as JSON."""
    report = {
        "timestamp": time.time(),
        "test_results": test_results,
        "diagnosis": diagnosis,
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ JSON report saved to: {output_file}")


def main():
    """Run improved diagnostic tests."""
    parser = argparse.ArgumentParser(description="DXMT Server Diagnostic Tool (Improved)")
    parser.add_argument("--json-output", help="Save JSON report to file", metavar="FILE")
    parser.add_argument("--skip-concurrent", action="store_true", help="Skip concurrent test")
    parser.add_argument("--skip-gradient", action="store_true", help="Skip load gradient test")
    args = parser.parse_args()

    print("=" * 80)
    print("DXMT Server Diagnostic Tool (Improved)")
    print("=" * 80)
    print("\nThis enhanced tool includes:")
    print("  • Performance metrics collection (latency, throughput)")
    print("  • Load gradient testing (find breaking points)")
    print("  • Concurrent insert testing")
    print("  • Improved diagnostic analysis")
    print("\nExpected runtime: ~7-10 minutes")

    client = Client(host="localhost", port=9999)
    dimension = 128
    table_name = "diagnostic_test"

    test_results = {}

    try:
        # Test 1: Replicate the issue with metrics
        try:
            client.delete_table(table_name)
            time.sleep(0.5)
        except Exception:
            pass

        client.create_table(
            name=table_name,
            dimension=dimension,
            distance_metric="cosine",
            index_type="hnsw",
        )

        success, data = test_insert_after_heavy_load(client, table_name, dimension)
        test_results["test1"] = data

        # Test 2: Load gradient (unless skipped)
        if not args.skip_gradient:
            test_results["test2"] = test_load_gradient(client, table_name, dimension)

        # Test 3: Concurrent inserts (unless skipped)
        if not args.skip_concurrent:
            test_results["test3"] = test_concurrent_inserts(client, table_name, dimension)

        # Test 4: With delays
        success, data = test_insert_with_delays(client, table_name, dimension)
        test_results["test4"] = data

        # Analyze and print summary
        diagnosis = analyze_results(test_results)
        print_diagnostic_summary(test_results, diagnosis)

        # Save JSON report if requested
        if args.json_output:
            save_json_report(test_results, diagnosis, args.json_output)

    finally:
        # Cleanup
        print("\n" + "=" * 80)
        print("Cleaning up...")
        try:
            client.delete_table(table_name)
            print("✅ Deleted test table")
        except Exception as e:
            print(f"⚠️  Cleanup failed: {e}")

        client.close()
        print("\n✅ Diagnostic complete!")


if __name__ == "__main__":
    main()
