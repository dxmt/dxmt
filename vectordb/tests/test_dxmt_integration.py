#!/usr/bin/env python3
"""
直接测试 DXMT VectorDBBench 集成
"""

import sys
import numpy as np

# 测试导入
print("1. 测试导入...")
try:
    from vectordb_bench.backend.clients import DB
    from vectordb_bench.backend.clients.dxmt import DXMT, DXMTHNSWConfig, DXMTConfig
    from vectordb_bench.backend.clients.api import MetricType
    print("✅ 导入成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 测试客户端创建
print("\n2. 测试客户端创建...")
try:
    config = {
        "host": "localhost",
        "port": 9999,
    }

    case_config = DXMTHNSWConfig(
        metric_type=MetricType.COSINE,
        m=16,
        ef_construction=200,
        ef_search=100,
    )

    client = DXMT(
        dim=128,
        db_config=config,
        db_case_config=case_config,
        collection_name="test_collection",
        drop_old=True,
    )
    print("✅ 客户端创建成功")
except Exception as e:
    print(f"❌ 客户端创建失败: {e}")
    sys.exit(1)

# 测试 init
print("\n3. 测试 init()...")
try:
    with client.init():
        print("✅ init() 成功")

        # 测试插入
        print("\n4. 测试插入数据...")
        embeddings = np.random.rand(100, 128).tolist()
        metadata = list(range(100))

        count, error = client.insert_embeddings(embeddings, metadata)
        if error:
            print(f"❌ 插入失败: {error}")
            sys.exit(1)
        print(f"✅ 插入成功: {count} 条记录")

        # 测试搜索
        print("\n5. 测试搜索...")
        query = np.random.rand(128).tolist()
        results = client.search_embedding(query, k=10)
        if not results:
            print("❌ 搜索失败: 没有结果")
            sys.exit(1)
        print(f"✅ 搜索成功: 返回 {len(results)} 个结果")
        print(f"   结果 IDs: {results[:5]}...")

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("✅ 所有测试通过！DXMT 集成工作正常！")
print("="*50)
