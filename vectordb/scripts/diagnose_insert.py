#!/usr/bin/env python3
"""
诊断插入性能 - 隔离测试
每个批次大小使用新表，排除累积数据影响
"""
import requests
import random
import time

BASE_URL = "http://localhost:9999/api/v1"

def test_batch_size(batch_size, dimension=128):
    """测试单个批次大小"""
    # Create new table for each test
    table_name = f"diag_{batch_size}_{int(time.time())}"

    start_create = time.time()
    response = requests.post(
        f"{BASE_URL}/tables",
        json={
            "name": table_name,
            "dimension": dimension,
            "distance_metric": "cosine"
        }
    )
    create_time = time.time() - start_create

    if response.status_code != 200:
        return None

    # Generate records
    start_gen = time.time()
    records = [
        {
            "id": f"vec_{i}",
            "vector": [random.random() for _ in range(dimension)],
            "payload": {"index": i}
        }
        for i in range(batch_size)
    ]
    gen_time = time.time() - start_gen

    # Insert records
    start_insert = time.time()
    response = requests.post(
        f"{BASE_URL}/tables/{table_name}/records",
        json={"records": records}
    )
    insert_time = time.time() - start_insert

    if response.status_code != 200:
        print(f"    插入失败: {response.text[:200]}")
        return None

    return {
        "batch_size": batch_size,
        "create_time": create_time,
        "gen_time": gen_time,
        "insert_time": insert_time,
        "throughput": batch_size / insert_time
    }

print("\n" + "="*70)
print("插入性能诊断测试（每批次使用新表）")
print("="*70)
print("\n{:>10} {:>12} {:>12} {:>12} {:>12}".format(
    "批次大小", "创建表(ms)", "生成(ms)", "插入(s)", "吞吐量(VPS)"
))
print("-"*70)

batch_sizes = [10, 50, 100, 200, 500, 1000, 2000, 5000]

results = []
for batch_size in batch_sizes:
    result = test_batch_size(batch_size)

    if result:
        results.append(result)
        print("{:>10} {:>12.1f} {:>12.1f} {:>12.3f} {:>12.0f}".format(
            result["batch_size"],
            result["create_time"] * 1000,
            result["gen_time"] * 1000,
            result["insert_time"],
            result["throughput"]
        ))
    else:
        print(f"{batch_size:>10} {'失败':>12}")

print("\n" + "="*70)
print("分析")
print("="*70)

if len(results) >= 2:
    # 分析趋势
    small_batch = [r for r in results if r["batch_size"] <= 100]
    large_batch = [r for r in results if r["batch_size"] >= 500]

    if small_batch and large_batch:
        avg_small = sum(r["throughput"] for r in small_batch) / len(small_batch)
        avg_large = sum(r["throughput"] for r in large_batch) / len(large_batch)

        print(f"\n小批次平均吞吐量 (≤100): {avg_small:.0f} VPS")
        print(f"大批次平均吞吐量 (≥500): {avg_large:.0f} VPS")
        print(f"性能比率: {avg_large / avg_small:.2f}x")

        if avg_large < avg_small:
            print("\n⚠️ 警告: 大批次性能低于小批次")
            print("可能原因:")
            print("  1. HTTP 序列化/反序列化开销随批次大小线性增长")
            print("  2. 索引构建时间随数据量增加")
            print("  3. 内存分配和拷贝开销")
            print("  4. JSON 解析性能瓶颈")
        else:
            print("\n✓ 大批次性能符合预期")
