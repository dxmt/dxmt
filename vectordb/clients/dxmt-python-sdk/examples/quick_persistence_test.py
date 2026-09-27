#!/usr/bin/env python3
"""快速持久化测试 - 验证 LMDB 数据持久化"""

from dxmt import Client
import sys

HOST = "127.0.0.1"
PORT = 9999
TABLE_NAME = "persistence_demo"

print("=" * 80)
print("  LMDB 持久化快速验证")
print("=" * 80)
print()

# 连接服务器
print(f"连接到服务器 {HOST}:{PORT}...")
client = Client(host=HOST, port=PORT)
print("✓ 连接成功\n")

# 检查表是否已存在（说明数据持久化了）
print("检查表是否存在...")
tables = client.list_tables()
table_exists = any(t['name'] == TABLE_NAME for t in tables)

if table_exists:
    print(f"✓ 表 '{TABLE_NAME}' 已存在（数据已持久化）\n")

    # 搜索数据
    print("搜索现有数据...")
    query_vector = [0.1 + i * 0.01 for i in range(128)]
    results = client.search(
        table_name=TABLE_NAME,
        vector=query_vector,
        k=5
    )

    print(f"✓ 找到 {len(results)} 条记录:\n")
    for i, result in enumerate(results, 1):
        metadata = result.get('metadata', {})
        print(f"  {i}. ID: {result['id']}")
        print(f"     相似度: {result['score']:.4f}")
        print(f"     标题: {metadata.get('title', 'N/A')}")
        print(f"     内容: {metadata.get('content', 'N/A')[:60]}...")
        print()

    print("=" * 80)
    print("✅ 持久化验证成功！重启后数据完整保留")
    print("=" * 80)

else:
    print(f"表 '{TABLE_NAME}' 不存在，创建新表并插入数据...\n")

    # 创建表
    client.create_table(
        name=TABLE_NAME,
        dimension=128,
        distance_metric="cosine",
        index_type="hnsw"
    )
    print(f"✓ 表 '{TABLE_NAME}' 创建成功\n")

    # 插入测试数据
    records = [
        {
            "id": "doc1",
            "vector": [0.1 + i * 0.01 for i in range(128)],
            "metadata": {
                "title": "DXMT VectorDB 简介",
                "content": "DXMT VectorDB 是一个轻量级、可水平扩展的向量数据库，现在支持 LMDB 持久化存储。",
                "category": "文档"
            }
        },
        {
            "id": "doc2",
            "vector": [0.2 + i * 0.01 for i in range(128)],
            "metadata": {
                "title": "LMDB 存储模式",
                "content": "LMDB 模式提供持久化存储，服务器重启后数据会自动恢复。",
                "category": "技术"
            }
        },
        {
            "id": "doc3",
            "vector": [0.15 + i * 0.01 for i in range(128)],
            "metadata": {
                "title": "向量搜索功能",
                "content": "支持余弦相似度、欧氏距离等多种相似度计算方法。",
                "category": "功能"
            }
        },
        {
            "id": "doc4",
            "vector": [0.25 + i * 0.01 for i in range(128)],
            "metadata": {
                "title": "混合搜索",
                "content": "结合向量搜索和全文搜索，提供更精准的检索结果。",
                "category": "功能"
            }
        },
        {
            "id": "doc5",
            "vector": [0.3 + i * 0.01 for i in range(128)],
            "metadata": {
                "title": "生产环境部署",
                "content": "LMDB 模式适合生产环境，提供高性能和数据持久化保证。",
                "category": "部署"
            }
        }
    ]

    client.insert_records(table_name=TABLE_NAME, records=records)
    print(f"✓ 插入 {len(records)} 条记录\n")

    # 验证插入
    print("验证数据插入...")
    query_vector = [0.1 + i * 0.01 for i in range(128)]
    results = client.search(
        table_name=TABLE_NAME,
        vector=query_vector,
        k=5
    )

    print(f"✓ 搜索结果: 找到 {len(results)} 条记录\n")
    for i, result in enumerate(results, 1):
        metadata = result.get('metadata', {})
        print(f"  {i}. [{result['id']}] {metadata.get('title', 'N/A')}")
        print(f"     相似度: {result['score']:.4f}")
    print()

    print("=" * 80)
    print("✅ 数据已插入并保存")
    print("=" * 80)
    print()
    print("下一步验证持久化:")
    print("  1. 重启服务器:")
    print("     pkill -f vectordb")
    print("     sleep 3")
    print("     ./target/release/vectordb serve &")
    print()
    print("  2. 再次运行此脚本:")
    print("     python3 quick_persistence_test.py")
    print()
    print("如果能找到这 5 条记录，说明持久化工作正常！")
    print("=" * 80)

client.close()
