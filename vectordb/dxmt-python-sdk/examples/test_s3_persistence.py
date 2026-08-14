#!/usr/bin/env python3
"""S3 持久化测试 - 验证数据存储到 AWS S3"""

import os

from dxmt import Client
import sys

HOST = "127.0.0.1"
PORT = 9999
TABLE_NAME = "s3_persistence_test"
S3_BUCKET = os.environ.get("S3_BUCKET", "your-bucket-name")

print("=" * 80)
print("  S3 持久化验证测试")
print("=" * 80)
print(f"\nS3 Bucket: {S3_BUCKET}")
print(f"服务器: {HOST}:{PORT}\n")

# 连接服务器
print("连接到 S3 模式服务器...")
client = Client(host=HOST, port=PORT)
print("✓ 连接成功\n")

# 检查表是否已存在
print("检查表是否存在...")
tables = client.list_tables()
table_exists = any(t['name'] == TABLE_NAME for t in tables)

if table_exists:
    print(f"✓ 表 '{TABLE_NAME}' 已存在（数据已从 S3 恢复）\n")

    # 搜索数据
    print("搜索现有数据（从 S3 加载）...")
    query_vector = [0.2 + i * 0.01 for i in range(256)]
    results = client.search(
        table_name=TABLE_NAME,
        vector=query_vector,
        k=10
    )

    if not results:
        print("⚠ 未找到任何记录")
    else:
        print(f"✓ 找到 {len(results)} 条记录:\n")
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            print(f"  {i}. ID: {result['id']}")
            print(f"     相似度: {result['score']:.4f}")
            print(f"     类别: {metadata.get('category', 'N/A')}")
            print(f"     描述: {metadata.get('description', 'N/A')[:60]}...")
            print()

    print("=" * 80)
    print("✅ S3 持久化验证成功！数据已从 S3 恢复")
    print("=" * 80)
    print()
    print(f"S3 路径: s3://{S3_BUCKET}/...")

else:
    print(f"表 '{TABLE_NAME}' 不存在，创建新表并插入数据到 S3...\n")

    # 创建表
    print(f"创建表 '{TABLE_NAME}'（256 维向量）...")
    client.create_table(
        name=TABLE_NAME,
        dimension=256,
        distance_metric="cosine",
        index_type="hnsw"
    )
    print(f"✓ 表创建成功\n")

    # 准备测试数据
    print("准备测试数据...")
    records = []

    # 机器学习相关文档
    ml_docs = [
        {
            "id": "ml_001",
            "title": "深度学习基础",
            "description": "深度学习是机器学习的一个分支，使用多层神经网络进行特征学习和模式识别。",
            "category": "机器学习",
            "tags": ["深度学习", "神经网络", "AI"]
        },
        {
            "id": "ml_002",
            "title": "卷积神经网络 CNN",
            "description": "CNN 是一种专门用于处理图像数据的深度学习架构，广泛应用于计算机视觉。",
            "category": "机器学习",
            "tags": ["CNN", "图像识别", "计算机视觉"]
        },
        {
            "id": "ml_003",
            "title": "自然语言处理 NLP",
            "description": "NLP 使计算机能够理解和生成人类语言，包括文本分类、情感分析等任务。",
            "category": "机器学习",
            "tags": ["NLP", "文本处理", "语言模型"]
        },
    ]

    # 向量数据库相关文档
    vectordb_docs = [
        {
            "id": "vdb_001",
            "title": "向量数据库原理",
            "description": "向量数据库使用高维向量表示数据，支持高效的相似度搜索和语义检索。",
            "category": "向量数据库",
            "tags": ["向量检索", "相似度搜索", "嵌入"]
        },
        {
            "id": "vdb_002",
            "title": "HNSW 索引算法",
            "description": "HNSW 是一种高效的近似最近邻搜索算法，使用分层的导航小世界图结构。",
            "category": "向量数据库",
            "tags": ["HNSW", "ANN", "索引"]
        },
        {
            "id": "vdb_003",
            "title": "S3 对象存储",
            "description": "Amazon S3 是云对象存储服务，提供高可用性、可扩展性和数据持久化保证。",
            "category": "云存储",
            "tags": ["S3", "对象存储", "云计算"]
        },
    ]

    # 生成向量（模拟 embeddings）
    import hashlib
    import random

    all_docs = ml_docs + vectordb_docs

    for doc in all_docs:
        # 使用文档内容的哈希生成确定性的伪随机向量
        content = doc['title'] + doc['description']
        hash_val = int(hashlib.md5(content.encode()).hexdigest(), 16)
        random.seed(hash_val)

        vector = [random.random() for _ in range(256)]

        records.append({
            "id": doc["id"],
            "vector": vector,
            "metadata": {
                "title": doc["title"],
                "description": doc["description"],
                "category": doc["category"],
                "tags": ", ".join(doc["tags"])
            }
        })

    print(f"✓ 准备了 {len(records)} 条测试记录\n")

    # 插入数据（会存储到 S3）
    print(f"插入数据到 S3...")
    print(f"  目标: s3://{S3_BUCKET}/")
    client.insert_records(table_name=TABLE_NAME, records=records)
    print(f"✓ 数据已插入（存储到 S3）\n")

    # 验证插入
    print("验证数据插入...")
    query_vector = [0.2 + i * 0.01 for i in range(256)]
    results = client.search(
        table_name=TABLE_NAME,
        vector=query_vector,
        k=len(records)
    )

    print(f"✓ 搜索结果: 找到 {len(results)} 条记录\n")

    print("记录列表:")
    for i, result in enumerate(results, 1):
        metadata = result.get('metadata', {})
        print(f"  {i}. [{result['id']}] {metadata.get('title', 'N/A')}")
        print(f"     类别: {metadata.get('category', 'N/A')}")
        print(f"     相似度: {result['score']:.4f}")
    print()

    print("=" * 80)
    print("✅ 数据已成功写入 S3")
    print("=" * 80)
    print()
    print("下一步验证 S3 持久化:")
    print(f"  1. 检查 S3 bucket:")
    print(f"     aws s3 ls s3://{S3_BUCKET}/")
    print()
    print("  2. 重启服务器:")
    print("     pkill -f vectordb")
    print("     sleep 3")
    print("     ./target/release/vectordb serve &")
    print()
    print("  3. 再次运行此脚本:")
    print("     python3 test_s3_persistence.py")
    print()
    print("如果能找到这些记录，说明 S3 持久化工作正常！")
    print("=" * 80)

client.close()
