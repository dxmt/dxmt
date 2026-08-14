#!/usr/bin/env python3
"""
测试索引状态监控API
"""
import requests
import random
import time
import json

BASE_URL = "http://localhost:9999/api/v1"

# 创建测试表
table_name = f"status_test_{int(time.time())}"
print(f"\n{'='*70}")
print(f"创建测试表: {table_name}")
print('='*70)

response = requests.post(
    f"{BASE_URL}/tables",
    json={
        "name": table_name,
        "dimension": 128,
        "distance_metric": "cosine"
    }
)
print(f"创建表状态: {response.status_code}")

# 查询初始索引状态
print(f"\n{'='*70}")
print("1. 初始索引状态（无数据）")
print('='*70)
response = requests.get(f"{BASE_URL}/tables/{table_name}/index_status")
if response.status_code == 200:
    status = response.json()
    print(json.dumps(status['data'], indent=2))
else:
    print(f"获取状态失败: {response.status_code}")
    exit(1)

# 插入小批次数据
print(f"\n{'='*70}")
print("2. 插入 50 条记录")
print('='*70)
records = [
    {
        "id": f"vec_{i}",
        "vector": [random.random() for _ in range(128)],
        "payload": {"index": i}
    }
    for i in range(50)
]

start = time.time()
response = requests.post(
    f"{BASE_URL}/tables/{table_name}/records",
    json={"records": records}
)
elapsed = time.time() - start

if response.status_code == 200:
    print(f"✓ 插入成功: {elapsed:.3f}s")

    # 立即查询索引状态（可能有pending操作）
    print("\n立即查询索引状态:")
    response = requests.get(f"{BASE_URL}/tables/{table_name}/index_status")
    if response.status_code == 200:
        status = response.json()
        print(json.dumps(status['data'], indent=2))

    # 等待一段时间后再查询
    time.sleep(0.5)
    print("\n等待 0.5s 后查询索引状态:")
    response = requests.get(f"{BASE_URL}/tables/{table_name}/index_status")
    if response.status_code == 200:
        status = response.json()
        print(json.dumps(status['data'], indent=2))
else:
    print(f"✗ 插入失败: {response.status_code}")

# 插入大批次数据
print(f"\n{'='*70}")
print("3. 插入 2000 条记录（测试延迟索引）")
print('='*70)
records = [
    {
        "id": f"vec_{50 + i}",
        "vector": [random.random() for _ in range(128)],
        "payload": {"index": 50 + i}
    }
    for i in range(2000)
]

start = time.time()
response = requests.post(
    f"{BASE_URL}/tables/{table_name}/records",
    json={"records": records}
)
elapsed = time.time() - start

if response.status_code == 200:
    print(f"✓ 插入成功: {elapsed:.3f}s, {2000/elapsed:.0f} VPS")

    # 立即查询（应该看到 pending > 0）
    print("\n立即查询索引状态（应有待处理操作）:")
    response = requests.get(f"{BASE_URL}/tables/{table_name}/index_status")
    if response.status_code == 200:
        status = response.json()
        data = status['data']
        print(json.dumps(data, indent=2))
        if data['pending_operations'] > 0:
            print(f"✓ 检测到 {data['pending_operations']} 个待处理操作")
        if data['index_lag'] > 0:
            print(f"✓ 索引滞后 {data['index_lag']} 条记录")

    # 轮询直到索引完成
    print("\n轮询索引状态直到完成...")
    max_attempts = 20
    for attempt in range(max_attempts):
        time.sleep(0.2)
        response = requests.get(f"{BASE_URL}/tables/{table_name}/index_status")
        if response.status_code == 200:
            data = response.json()['data']
            pending = data['pending_operations']
            lag = data['index_lag']
            print(f"  第 {attempt+1:2d} 次: pending={pending}, lag={lag}")
            if pending == 0 and lag == 0:
                print("\n✓ 索引构建完成!")
                break
    else:
        print("\n⚠ 轮询超时，但这可能是正常的（索引仍在后台运行）")

    # 最终状态
    print("\n最终索引状态:")
    response = requests.get(f"{BASE_URL}/tables/{table_name}/index_status")
    if response.status_code == 200:
        status = response.json()
        print(json.dumps(status['data'], indent=2))
else:
    print(f"✗ 插入失败: {response.status_code}")

# 验证搜索功能
print(f"\n{'='*70}")
print("4. 验证搜索功能（确保索引可用）")
print('='*70)
query_vector = [random.random() for _ in range(128)]
response = requests.post(
    f"{BASE_URL}/tables/{table_name}/search",
    json={
        "vector": query_vector,
        "top_k": 5
    }
)

if response.status_code == 200:
    result = response.json()
    hits = result['data']['hits']
    print(f"✓ 搜索成功: 返回 {len(hits)} 条结果")
    for i, hit in enumerate(hits[:3]):
        print(f"  {i+1}. id={hit['id']}, score={hit['score']:.4f}")
else:
    print(f"✗ 搜索失败: {response.status_code}")

print(f"\n{'='*70}")
print("测试完成!")
print('='*70)
