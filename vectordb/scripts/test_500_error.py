#!/usr/bin/env python3
"""
测试触发 500 错误的脚本
"""
import requests
import random

BASE_URL = "http://localhost:9999/api/v1"

# Create table
print("创建测试表...")
response = requests.post(
    f"{BASE_URL}/tables",
    json={
        "name": "error_test",
        "dimension": 128,
        "distance_metric": "cosine"
    }
)
print(f"创建表: {response.status_code}")

# Test different batch sizes
batch_sizes = [10, 50, 100, 500, 1000]

for batch_size in batch_sizes:
    print(f"\n测试批次大小: {batch_size}")

    records = [
        {
            "id": f"vec_{i}",
            "vector": [random.random() for _ in range(128)],
            "payload": {"index": i, "batch": batch_size}
        }
        for i in range(batch_size)
    ]

    response = requests.post(
        f"{BASE_URL}/tables/error_test/records",
        json={"records": records}
    )

    print(f"状态码: {response.status_code}")
    if response.status_code != 200:
        print(f"错误: {response.text[:500]}")
        print("\n检查服务器日志以获取详细错误信息...")
        break
    else:
        print(f"成功插入 {batch_size} 条记录")
