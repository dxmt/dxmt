#!/usr/bin/env python3
"""GIST-960 快速基准测试（通过 HTTP API）"""

import requests
import time
import numpy as np
import statistics
import sys

BASE_URL = "http://localhost:9999/api/v1"
DIM = 960

print("=" * 80)
print("  GIST-960 快速基准测试（通过 HTTP API）")
print("=" * 80)
print()

def generate_vector(seed=None):
    if seed is not None:
        np.random.seed(seed)
    vec = np.random.randn(DIM).astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    return vec.tolist()

# 测试 1: 插入性能
print("测试 1: 插入性能")
print("-" * 80)

table_name = f"gist_quick_{int(time.time())}"
response = requests.post(f"{BASE_URL}/tables", json={
    "name": table_name,
    "dimension": DIM,
    "distance_metric": "euclidean",
    "index_type": "hnsw"
})

if response.status_code != 200:
    print(f"✗ 创建表失败: {response.text}")
    sys.exit(1)

print(f"✓ 表创建成功\n")

batch_sizes = [100, 500, 1000, 5000]

print(f"{'批次大小':<12} {'耗时(s)':<12} {'吞吐量(vec/s)':<15}")
print("-" * 50)

for batch_size in batch_sizes:
    records = [{
        "id": f"vec_{i}",
        "vector": generate_vector(seed=i),
        "payload": {"idx": i}
    } for i in range(batch_size)]

    start = time.time()
    response = requests.post(
        f"{BASE_URL}/tables/{table_name}/records",
        json={"records": records}
    )
    elapsed = time.time() - start

    if response.status_code == 200:
        throughput = batch_size / elapsed
        print(f"{batch_size:<12} {elapsed:<12.3f} {throughput:<15.0f}")
    else:
        print(f"{batch_size:<12} 失败")

# 测试 2: 搜索性能
print(f"\n测试 2: 搜索性能")
print("-" * 80)

# 先插入更多数据
print("准备测试数据 (20,000 向量)...")
sys.stdout.flush()

for batch_idx in range(4):
    batch_records = [{
        "id": f"vec2_{i}",
        "vector": generate_vector(seed=10000 + batch_idx * 5000 + i),
        "payload": {"idx": i}
    } for i in range(5000)]

    response = requests.post(
        f"{BASE_URL}/tables/{table_name}/records",
        json={"records": batch_records}
    )
    print(f"  批次 {batch_idx+1}/4 完成")
    sys.stdout.flush()

print("✓ 数据准备完成\n")

# 执行搜索测试
num_queries = 100
print(f"执行 {num_queries} 次搜索...")
sys.stdout.flush()

latencies = []
for i in range(num_queries):
    query_vec = generate_vector(seed=1000000 + i)

    start = time.time()
    response = requests.post(
        f"{BASE_URL}/tables/{table_name}/search",
        json={
            "vector": query_vec,
            "k": 10,
            "with_payload": False
        }
    )
    latency = (time.time() - start) * 1000

    if response.status_code == 200:
        latencies.append(latency)

if latencies:
    print(f"\n搜索性能 (20,000 向量, k=10):")
    print(f"  平均延迟: {statistics.mean(latencies):8.2f} ms")
    print(f"  P50 延迟: {statistics.median(latencies):8.2f} ms")
    print(f"  P95 延迟: {np.percentile(latencies, 95):8.2f} ms")
    print(f"  P99 延迟: {np.percentile(latencies, 99):8.2f} ms")
    print(f"  QPS:      {1000/statistics.mean(latencies):8.0f}")

# 清理
print(f"\n清理测试表...")
requests.delete(f"{BASE_URL}/tables/{table_name}")

print("\n" + "=" * 80)
print("✓ 测试完成")
print("=" * 80)
