#!/usr/bin/env python3
"""
修复后的插入性能测试 - 使用唯一ID
"""
import requests
import random
import time
import uuid

BASE_URL = "http://localhost:9999/api/v1"

# Create table with unique name
table_name = f"perf_test_{int(time.time())}"
print(f"\n创建测试表: {table_name}")
response = requests.post(
    f"{BASE_URL}/tables",
    json={
        "name": table_name,
        "dimension": 128,
        "distance_metric": "cosine"
    }
)
print(f"状态: {response.status_code}")

# Test different batch sizes with unique IDs
print("\n" + "="*70)
print("测试不同批次大小的插入性能")
print("="*70)

batch_sizes = [10, 50, 100, 500, 1000, 2000, 5000]
offset = 0

for batch_size in batch_sizes:
    print(f"\n批次大小: {batch_size}")

    # Generate unique IDs
    records = [
        {
            "id": f"vec_{offset + i}",
            "vector": [random.random() for _ in range(128)],
            "payload": {"index": offset + i, "batch": batch_size}
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
        print(f"  ✓ 成功: {elapsed:.3f}s | {throughput:.0f} VPS")
        offset += batch_size
    else:
        print(f"  ✗ 失败: {response.status_code}")
        print(f"  错误: {response.text[:300]}")
        break

print("\n" + "="*70)
print("测试完成")
print("="*70)
