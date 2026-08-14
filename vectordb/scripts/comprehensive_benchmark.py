#!/usr/bin/env python3
"""
全面的性能基准测试 - 修复后的版本
包括单线程和并发测试
"""

import requests
import time
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://localhost:9999/api/v1"

def generate_vector(dim=128):
    return [random.random() for _ in range(dim)]

def test_insert_throughput():
    """测试插入吞吐量（不同批次大小）"""
    print("\n" + "="*70)
    print("测试 1: 单线程插入吞吐量（Rayon 并行化效果）")
    print("="*70)

    # Create table
    table_name = f"insert_test_{int(time.time())}"
    requests.post(f"{BASE_URL}/tables", json={
        "name": table_name,
        "dimension": 128,
        "distance_metric": "cosine"
    })

    batch_sizes = [10, 50, 100, 500, 1000, 2000]

    for batch_size in batch_sizes:
        records = [
            {
                "id": f"vec_{i}",
                "vector": generate_vector(128),
                "payload": {"index": i}
            }
            for i in range(batch_size)
        ]

        start = time.time()
        response = requests.post(
            f"{BASE_URL}/tables/{table_name}/records",
            json={"records": records}
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            throughput = batch_size / elapsed
            print(f"批次大小: {batch_size:5d} | "
                  f"耗时: {elapsed:6.3f}s | "
                  f"吞吐量: {throughput:8.0f} VPS")
        else:
            print(f"批次大小: {batch_size:5d} | 失败: {response.status_code}")

def test_concurrent_insert():
    """测试并发插入（DashMap 效果）"""
    print("\n" + "="*70)
    print("测试 2: 并发批量插入（DashMap 无锁优化）")
    print("="*70)

    # Create multiple tables for concurrent testing
    tables = []
    for i in range(8):
        table_name = f"concurrent_test_{int(time.time())}_{i}"
        response = requests.post(f"{BASE_URL}/tables", json={
            "name": table_name,
            "dimension": 128,
            "distance_metric": "cosine"
        })
        if response.status_code == 200:
            tables.append(table_name)

    def insert_batch(table_name, batch_size):
        records = [
            {
                "id": f"vec_{i}",
                "vector": generate_vector(128),
                "payload": {"index": i}
            }
            for i in range(batch_size)
        ]

        start = time.time()
        response = requests.post(
            f"{BASE_URL}/tables/{table_name}/records",
            json={"records": records}
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            return batch_size, elapsed
        return 0, elapsed

    batch_size = 500
    for num_threads in [1, 2, 4, 8]:
        start = time.time()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(insert_batch, tables[i % len(tables)], batch_size)
                for i in range(num_threads)
            ]

            results = [f.result() for f in as_completed(futures)]

        total_time = time.time() - start
        total_inserted = sum(r[0] for r in results)
        throughput = total_inserted / total_time

        print(f"线程数: {num_threads} | "
              f"总插入: {total_inserted:5d} | "
              f"耗时: {total_time:6.3f}s | "
              f"吞吐量: {throughput:8.0f} VPS")

def test_search_performance():
    """测试搜索性能（SIMD 效果）"""
    print("\n" + "="*70)
    print("测试 3: 搜索延迟（SIMD 加速）")
    print("="*70)

    # Create table and insert data
    table_name = f"search_test_{int(time.time())}"
    requests.post(f"{BASE_URL}/tables", json={
        "name": table_name,
        "dimension": 128,
        "distance_metric": "cosine"
    })

    # Insert 10K vectors
    print(f"预插入 10,000 个向量...")
    for i in range(10):
        records = [
            {
                "id": f"vec_{i*1000 + j}",
                "vector": generate_vector(128),
                "payload": {"index": i*1000 + j}
            }
            for j in range(1000)
        ]
        requests.post(
            f"{BASE_URL}/tables/{table_name}/records",
            json={"records": records}
        )

    # Test search
    print(f"执行 100 次搜索查询...")
    latencies = []

    for _ in range(100):
        query_vector = generate_vector(128)

        start = time.time()
        response = requests.post(
            f"{BASE_URL}/tables/{table_name}/search",
            json={
                "vector": query_vector,
                "top_k": 10
            }
        )
        elapsed = (time.time() - start) * 1000  # ms

        if response.status_code == 200:
            latencies.append(elapsed)

    if latencies:
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]
        p99 = statistics.quantiles(latencies, n=100)[98]
        avg = statistics.mean(latencies)
        qps = 1000 / avg

        print(f"\n查询延迟统计:")
        print(f"  平均延迟: {avg:6.2f}ms")
        print(f"  P50 延迟: {p50:6.2f}ms")
        print(f"  P95 延迟: {p95:6.2f}ms")
        print(f"  P99 延迟: {p99:6.2f}ms")
        print(f"  预估 QPS: {qps:8.0f}")

def test_concurrent_search():
    """测试并发搜索"""
    print("\n" + "="*70)
    print("测试 4: 并发搜索性能")
    print("="*70)

    # Reuse table from previous test
    table_name = f"concurrent_search_{int(time.time())}"
    requests.post(f"{BASE_URL}/tables", json={
        "name": table_name,
        "dimension": 128,
        "distance_metric": "cosine"
    })

    # Insert 5K vectors
    print(f"预插入 5,000 个向量...")
    for i in range(5):
        records = [
            {
                "id": f"vec_{i*1000 + j}",
                "vector": generate_vector(128),
                "payload": {"index": i*1000 + j}
            }
            for j in range(1000)
        ]
        requests.post(
            f"{BASE_URL}/tables/{table_name}/records",
            json={"records": records}
        )

    def search_query():
        query_vector = generate_vector(128)
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/tables/{table_name}/search",
            json={
                "vector": query_vector,
                "top_k": 10
            }
        )
        elapsed = (time.time() - start) * 1000
        return elapsed if response.status_code == 200 else -1

    queries_per_thread = 50

    for num_threads in [1, 2, 4, 8]:
        start = time.time()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(search_query)
                for _ in range(num_threads * queries_per_thread)
            ]

            latencies = [f.result() for f in as_completed(futures) if f.result() > 0]

        total_time = time.time() - start
        total_queries = len(latencies)
        qps = total_queries / total_time
        avg_latency = statistics.mean(latencies) if latencies else 0

        print(f"线程数: {num_threads} | "
              f"查询数: {total_queries:4d} | "
              f"耗时: {total_time:6.3f}s | "
              f"QPS: {qps:8.0f} | "
              f"平均延迟: {avg_latency:6.2f}ms")

def main():
    print("\n" + "="*70)
    print("DXMT VectorDB - Phase 1 全面性能验证")
    print("="*70)

    # Check server
    try:
        response = requests.get("http://localhost:9999/status")
        if response.status_code != 200:
            print("\n❌ 服务器未响应")
            return
    except Exception as e:
        print(f"\n❌ 无法连接服务器: {e}")
        return

    print("\n✓ 服务器运行正常")

    test_insert_throughput()
    test_concurrent_insert()
    test_search_performance()
    test_concurrent_search()

    print("\n" + "="*70)
    print("性能评估")
    print("="*70)
    print("\n目标:")
    print("  • 单线程插入: > 5,000 VPS")
    print("  • 8线程并发插入: > 20,000 VPS")
    print("  • 搜索 P95 延迟: < 50ms")
    print("  • 并发搜索 QPS: > 1,000")
    print("\n如果达到目标 → ✅ 可以部署")
    print("如果未达目标 → ⚠️  考虑 Phase 2 优化")

if __name__ == "__main__":
    main()
