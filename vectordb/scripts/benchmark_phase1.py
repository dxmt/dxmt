#!/usr/bin/env python3
"""
Phase 1 性能验证测试
测试 SIMD、DashMap、Rayon 并行化的实际效果
"""

import requests
import time
import random
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

BASE_URL = "http://localhost:9999/api/v1"

def generate_vector(dim: int = 128) -> List[float]:
    """生成随机向量"""
    return [random.random() for _ in range(dim)]

def create_table(name: str, dimension: int = 128) -> bool:
    """创建表"""
    response = requests.post(
        f"{BASE_URL}/tables",
        json={
            "name": name,
            "dimension": dimension,
            "distance_metric": "cosine"
        }
    )
    return response.status_code == 200

def insert_batch(table_name: str, batch_size: int, start_id: int = 0) -> Tuple[float, int]:
    """插入批量向量并返回耗时和实际插入数"""
    records = [
        {
            "id": f"vec_{start_id + i}",
            "vector": generate_vector(128),
            "payload": {"index": start_id + i}
        }
        for i in range(batch_size)
    ]

    start = time.time()
    response = requests.post(
        f"{BASE_URL}/tables/{table_name}/records",
        json=records
    )
    elapsed = time.time() - start

    if response.status_code == 200:
        return elapsed, batch_size
    else:
        print(f"插入失败: {response.status_code} - {response.text}")
        return elapsed, 0

def search_query(table_name: str, top_k: int = 10) -> float:
    """执行搜索并返回延迟"""
    query_vector = generate_vector(128)

    start = time.time()
    response = requests.post(
        f"{BASE_URL}/tables/{table_name}/search",
        json={
            "vector": query_vector,
            "top_k": top_k
        }
    )
    elapsed = time.time() - start

    if response.status_code == 200:
        return elapsed * 1000  # 转换为毫秒
    else:
        print(f"搜索失败: {response.status_code}")
        return -1

def test_1_single_thread_insert():
    """测试 1: 单线程批量插入性能"""
    print("\n" + "="*60)
    print("测试 1: 单线程批量插入性能")
    print("="*60)

    table_name = "test_single_thread"
    create_table(table_name)

    batch_sizes = [100, 500, 1000]

    for batch_size in batch_sizes:
        elapsed, count = insert_batch(table_name, batch_size)
        throughput = count / elapsed

        print(f"批次大小: {batch_size:4d} | "
              f"耗时: {elapsed:6.3f}s | "
              f"吞吐量: {throughput:7.0f} VPS")

    return True

def test_2_concurrent_insert():
    """测试 2: 并发批量插入（测试 DashMap 效果）"""
    print("\n" + "="*60)
    print("测试 2: 并发批量插入（DashMap 无锁优化）")
    print("="*60)

    table_name = "test_concurrent"
    create_table(table_name)

    num_threads_list = [1, 2, 4, 8]
    batch_size = 200

    for num_threads in num_threads_list:
        start = time.time()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(insert_batch, table_name, batch_size, i * batch_size)
                for i in range(num_threads)
            ]

            total_inserted = sum(f.result()[1] for f in as_completed(futures))

        elapsed = time.time() - start
        throughput = total_inserted / elapsed

        print(f"线程数: {num_threads} | "
              f"总插入: {total_inserted:4d} | "
              f"耗时: {elapsed:6.3f}s | "
              f"吞吐量: {throughput:7.0f} VPS")

    return True

def test_3_search_latency():
    """测试 3: 搜索延迟（测试 SIMD 效果）"""
    print("\n" + "="*60)
    print("测试 3: 搜索延迟（SIMD 加速）")
    print("="*60)

    table_name = "test_search"
    create_table(table_name)

    # 预先插入数据
    print("预先插入 10,000 个向量...")
    for i in range(10):
        insert_batch(table_name, 1000, i * 1000)

    # 执行搜索测试
    num_queries = 100
    latencies = []

    print(f"执行 {num_queries} 次搜索查询...")
    for _ in range(num_queries):
        latency = search_query(table_name)
        if latency > 0:
            latencies.append(latency)

    if latencies:
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        avg = statistics.mean(latencies)
        qps = 1000 / avg  # 基于平均延迟计算 QPS

        print(f"\n查询延迟统计:")
        print(f"  平均延迟: {avg:6.2f}ms")
        print(f"  P50 延迟: {p50:6.2f}ms")
        print(f"  P95 延迟: {p95:6.2f}ms")
        print(f"  P99 延迟: {p99:6.2f}ms")
        print(f"  预估 QPS: {qps:7.0f}")

    return True

def test_4_concurrent_search():
    """测试 4: 并发搜索（测试整体系统性能）"""
    print("\n" + "="*60)
    print("测试 4: 并发搜索性能")
    print("="*60)

    table_name = "test_concurrent_search"
    create_table(table_name)

    # 预先插入数据
    print("预先插入 5,000 个向量...")
    for i in range(5):
        insert_batch(table_name, 1000, i * 1000)

    num_threads_list = [1, 2, 4, 8]
    queries_per_thread = 50

    for num_threads in num_threads_list:
        start = time.time()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(search_query, table_name)
                for _ in range(num_threads * queries_per_thread)
            ]

            latencies = [f.result() for f in as_completed(futures) if f.result() > 0]

        elapsed = time.time() - start
        total_queries = len(latencies)
        qps = total_queries / elapsed
        avg_latency = statistics.mean(latencies) if latencies else 0

        print(f"线程数: {num_threads} | "
              f"查询数: {total_queries:3d} | "
              f"耗时: {elapsed:6.3f}s | "
              f"QPS: {qps:7.0f} | "
              f"平均延迟: {avg_latency:5.2f}ms")

    return True

def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("DXMT VectorDB - Phase 1 性能验证测试")
    print("="*60)
    print("\n测试项目:")
    print("  1. 单线程批量插入（测试 Rayon 并行化）")
    print("  2. 并发批量插入（测试 DashMap 无锁）")
    print("  3. 搜索延迟（测试 SIMD 加速）")
    print("  4. 并发搜索（测试整体性能）")

    # 检查服务器状态
    try:
        response = requests.get("http://localhost:9999/status")
        if response.status_code != 200:
            print("\n❌ 错误: 服务器未运行或未响应")
            print("   请先启动服务器: ./target/release/vectordb serve")
            return
    except Exception as e:
        print(f"\n❌ 错误: 无法连接到服务器: {e}")
        print("   请先启动服务器: ./target/release/vectordb serve")
        return

    print("\n✓ 服务器运行正常，开始测试...\n")

    # 运行测试
    tests = [
        test_1_single_thread_insert,
        test_2_concurrent_insert,
        test_3_search_latency,
        test_4_concurrent_search,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n❌ 测试失败: {test_func.__name__}")
            print(f"   错误: {e}")
            results.append((test_func.__name__, False))

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} - {name}")

    print(f"\n总计: {passed}/{total} 测试通过")

    print("\n" + "="*60)
    print("性能评估建议")
    print("="*60)
    print("\n基于测试结果，如果达到以下目标则认为优化成功：")
    print("  • 单线程插入: > 5,000 VPS")
    print("  • 8线程并发插入: > 20,000 VPS")
    print("  • 搜索 P95 延迟: < 50ms")
    print("  • 并发搜索 QPS: > 1,000")
    print("\n如果达到目标 → ✅ 可以部署到生产环境")
    print("如果未达目标 → ⚠️  需要继续 Phase 2 优化")

if __name__ == "__main__":
    main()
