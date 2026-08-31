#!/usr/bin/env python3
"""
测试同步插入 vs 异步插入
"""
import requests
import random
import time
import json

BASE_URL = "http://localhost:9999/api/v1"

def test_insert(table_name, batch_size, sync_mode):
    """测试插入性能"""
    records = [
        {
            "id": f"vec_{i}",
            "vector": [random.random() for _ in range(128)],
            "payload": {"index": i}
        }
        for i in range(batch_size)
    ]

    # 构建URL
    url = f"{BASE_URL}/tables/{table_name}/records"
    if sync_mode:
        url += "?sync=true"

    start = time.time()
    response = requests.post(url, json={"records": records})
    elapsed = time.time() - start

    if response.status_code == 200:
        result = response.json()
        mode = result['data'].get('mode', 'unknown')
        count = result['data'].get('count', 0)
        return elapsed, count, mode
    else:
        print(f"插入失败: {response.status_code}")
        return None, None, None

# 创建两个测试表
print(f"\n{'='*70}")
print("同步插入 vs 异步插入性能对比")
print('='*70)

timestamp = int(time.time())
table_async = f"async_test_{timestamp}"
table_sync = f"sync_test_{timestamp}"

for table_name in [table_async, table_sync]:
    response = requests.post(
        f"{BASE_URL}/tables",
        json={
            "name": table_name,
            "dimension": 128,
            "distance_metric": "cosine"
        }
    )
    print(f"创建表 {table_name}: {response.status_code}")

print()

# 测试不同批次大小
batch_sizes = [50, 100, 500, 1000, 2000]

print(f"\n{'='*70}")
print("异步插入（默认）")
print('='*70)
print(f"{'批次大小':>10} {'延迟(s)':>12} {'吞吐量(VPS)':>15} {'模式':>10}")
print('-'*70)

async_results = {}
for batch_size in batch_sizes:
    elapsed, count, mode = test_insert(table_async, batch_size, sync_mode=False)
    if elapsed:
        throughput = count / elapsed
        async_results[batch_size] = (elapsed, throughput)
        print(f"{batch_size:>10} {elapsed:>12.3f} {throughput:>15.0f} {mode:>10}")
    time.sleep(0.1)  # 等待后台索引完成

print(f"\n{'='*70}")
print("同步插入（sync=true）")
print('='*70)
print(f"{'批次大小':>10} {'延迟(s)':>12} {'吞吐量(VPS)':>15} {'模式':>10}")
print('-'*70)

sync_results = {}
offset = 0
for batch_size in batch_sizes:
    # 生成不同的ID避免冲突
    records = [
        {
            "id": f"sync_vec_{offset + i}",
            "vector": [random.random() for _ in range(128)],
            "payload": {"index": offset + i}
        }
        for i in range(batch_size)
    ]

    url = f"{BASE_URL}/tables/{table_sync}/records?sync=true"
    start = time.time()
    response = requests.post(url, json={"records": records})
    elapsed = time.time() - start

    if response.status_code == 200:
        result = response.json()
        mode = result['data'].get('mode', 'unknown')
        count = result['data'].get('count', 0)
        throughput = count / elapsed
        sync_results[batch_size] = (elapsed, throughput)
        print(f"{batch_size:>10} {elapsed:>12.3f} {throughput:>15.0f} {mode:>10}")
        offset += batch_size

# 对比分析
print(f"\n{'='*70}")
print("性能对比分析")
print('='*70)
print(f"{'批次大小':>10} {'异步(VPS)':>15} {'同步(VPS)':>15} {'异步/同步':>12}")
print('-'*70)

for batch_size in batch_sizes:
    if batch_size in async_results and batch_size in sync_results:
        async_throughput = async_results[batch_size][1]
        sync_throughput = sync_results[batch_size][1]
        ratio = async_throughput / sync_throughput if sync_throughput > 0 else 0
        print(f"{batch_size:>10} {async_throughput:>15.0f} {sync_throughput:>15.0f} {ratio:>12.2f}x")

# 验证索引状态
print(f"\n{'='*70}")
print("验证索引状态")
print('='*70)

for table_name, label in [(table_async, "异步表"), (table_sync, "同步表")]:
    response = requests.get(f"{BASE_URL}/tables/{table_name}/index_status")
    if response.status_code == 200:
        status = response.json()['data']
        print(f"\n{label} ({table_name}):")
        print(f"  总记录数: {status['total_records']}")
        print(f"  已索引: {status['indexed_vectors']}")
        print(f"  待处理: {status['pending_operations']}")
        print(f"  索引滞后: {status['index_lag']}")

# 测试搜索
print(f"\n{'='*70}")
print("验证搜索功能")
print('='*70)

query_vector = [random.random() for _ in range(128)]
for table_name, label in [(table_async, "异步表"), (table_sync, "同步表")]:
    response = requests.post(
        f"{BASE_URL}/tables/{table_name}/search",
        json={"vector": query_vector, "top_k": 5}
    )
    if response.status_code == 200:
        hits = response.json()['data']['hits']
        print(f"{label}: 搜索返回 {len(hits)} 条结果 ✓")
    else:
        print(f"{label}: 搜索失败 ✗")

print(f"\n{'='*70}")
print("测试完成!")
print('='*70)
