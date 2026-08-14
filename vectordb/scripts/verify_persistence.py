#!/usr/bin/env python3
"""
LMDB 持久化验证脚本

验证步骤:
1. 连接到 DXMT 服务器
2. 创建测试表并插入数据
3. 搜索验证数据已插入
4. 提示用户重启服务器
5. 重启后再次搜索，验证数据是否持久化

使用方法:
    python3 verify_persistence.py
"""

import sys
import time
from dxmt import Client

# 配置
HOST = "127.0.0.1"
PORT = 9999
TABLE_NAME = "persistence_test"
DIMENSION = 128

# 测试数据
TEST_RECORDS = [
    {
        "id": "test_1",
        "vector": [0.1 + i * 0.01 for i in range(DIMENSION)],
        "metadata": {
            "name": "测试记录 1",
            "description": "这是第一条测试记录，用于验证 LMDB 持久化",
            "timestamp": "2026-01-25T21:50:00Z"
        }
    },
    {
        "id": "test_2",
        "vector": [0.2 + i * 0.01 for i in range(DIMENSION)],
        "metadata": {
            "name": "测试记录 2",
            "description": "这是第二条测试记录",
            "timestamp": "2026-01-25T21:51:00Z"
        }
    },
    {
        "id": "test_3",
        "vector": [0.3 + i * 0.01 for i in range(DIMENSION)],
        "metadata": {
            "name": "测试记录 3",
            "description": "这是第三条测试记录",
            "timestamp": "2026-01-25T21:52:00Z"
        }
    }
]


def print_section(title):
    """打印分隔线"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def connect_client():
    """连接到 DXMT 服务器"""
    try:
        client = Client(host=HOST, port=PORT)
        print(f"✓ 成功连接到 DXMT 服务器 ({HOST}:{PORT})")
        return client
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        print(f"\n请确保 DXMT 服务器正在运行:")
        print(f"  cd /localdisk/code/dxmt/vectordb")
        print(f"  ./target/release/vectordb serve")
        sys.exit(1)


def phase1_insert_data():
    """阶段 1: 插入测试数据"""
    print_section("阶段 1: 插入测试数据")

    client = connect_client()

    # 删除旧表（如果存在）
    try:
        client.delete_table(TABLE_NAME)
        print(f"已删除旧表: {TABLE_NAME}")
    except:
        pass

    # 创建表
    print(f"\n创建表: {TABLE_NAME}")
    client.create_table(
        name=TABLE_NAME,
        dimension=DIMENSION,
        distance_metric="cosine",
        index_type="hnsw"
    )
    print(f"✓ 表创建成功")

    # 插入记录
    print(f"\n插入 {len(TEST_RECORDS)} 条测试记录...")
    client.insert_records(table_name=TABLE_NAME, records=TEST_RECORDS)
    print(f"✓ 记录插入成功")

    # 验证插入
    print(f"\n验证数据已插入...")
    query_vector = [0.15 + i * 0.01 for i in range(DIMENSION)]
    results = client.search(
        table_name=TABLE_NAME,
        vector=query_vector,
        k=3
    )

    print(f"\n搜索结果: 找到 {len(results)} 条记录")
    for i, result in enumerate(results, 1):
        metadata = result.get('metadata', {})
        print(f"\n  {i}. ID: {result['id']}")
        print(f"     相似度: {result['score']:.4f}")
        print(f"     名称: {metadata.get('name', 'N/A')}")
        print(f"     描述: {metadata.get('description', 'N/A')}")

    client.close()

    print_section("阶段 1 完成")
    print("测试数据已成功插入到数据库")
    print(f"\n数据应该持久化在: ./data/lmdb")
    print(f"\n请执行以下步骤验证持久化:")
    print(f"\n  1. 停止当前运行的 vectordb 服务器:")
    print(f"     pkill -f vectordb")
    print(f"\n  2. 等待 2-3 秒")
    print(f"\n  3. 重新启动服务器:")
    print(f"     cd /localdisk/code/dxmt/vectordb")
    print(f"     ./target/release/vectordb serve &")
    print(f"\n  4. 等待服务器启动（约 3-5 秒）")
    print(f"\n  5. 运行阶段 2 验证:")
    print(f"     python3 verify_persistence.py --phase2")
    print(f"\n或者使用自动化脚本:")
    print(f"     bash verify_persistence.sh")


def phase2_verify_persistence():
    """阶段 2: 验证持久化"""
    print_section("阶段 2: 验证数据持久化")

    print("正在连接到重启后的服务器...")
    time.sleep(1)

    client = connect_client()

    # 搜索之前插入的数据
    print(f"\n在重启后的服务器中搜索数据...")
    query_vector = [0.15 + i * 0.01 for i in range(DIMENSION)]

    try:
        results = client.search(
            table_name=TABLE_NAME,
            vector=query_vector,
            k=3
        )

        if not results:
            print("✗ 未找到任何记录 - 数据未持久化!")
            print("\n可能的原因:")
            print("  1. 服务器未使用 LMDB 存储模式")
            print("  2. .env 配置未正确加载")
            print("  3. 数据目录权限问题")
            client.close()
            sys.exit(1)

        print(f"✓ 成功找到 {len(results)} 条记录 - 数据已持久化!")

        print(f"\n重启后的搜索结果:")
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            print(f"\n  {i}. ID: {result['id']}")
            print(f"     相似度: {result['score']:.4f}")
            print(f"     名称: {metadata.get('name', 'N/A')}")
            print(f"     描述: {metadata.get('description', 'N/A')}")

        # 验证数据完整性
        print(f"\n数据完整性验证:")
        original_ids = {r['id'] for r in TEST_RECORDS}
        result_ids = {r['id'] for r in results}

        if original_ids == result_ids:
            print("  ✓ 所有原始记录都已恢复")
        else:
            missing = original_ids - result_ids
            if missing:
                print(f"  ⚠ 缺失记录: {missing}")

        client.close()

        print_section("阶段 2 完成")
        print("✓ LMDB 持久化验证成功!")
        print("\n持久化已正常工作，数据在服务器重启后成功恢复。")

        # 检查 LMDB 文件
        print(f"\nLMDB 数据文件:")
        import os
        lmdb_path = "./data/lmdb"
        if os.path.exists(lmdb_path):
            for file in os.listdir(lmdb_path):
                file_path = os.path.join(lmdb_path, file)
                size = os.path.getsize(file_path)
                print(f"  {file}: {size / 1024:.2f} KB")

    except Exception as e:
        print(f"✗ 验证失败: {e}")
        client.close()
        sys.exit(1)


def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "--phase2":
        phase2_verify_persistence()
    else:
        phase1_insert_data()


if __name__ == "__main__":
    main()
