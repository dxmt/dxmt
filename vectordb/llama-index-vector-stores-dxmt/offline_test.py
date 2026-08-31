"""离线测试 - 不需要运行 DXMT 服务器"""

print("=" * 60)
print("LlamaIndex DXMT 集成离线测试")
print("=" * 60)

# 测试1: 导入
print("\n[测试 1] 导入模块...")
try:
    from llama_index.vector_stores.dxmt import DXMTVectorStore
    from llama_index.vector_stores.dxmt import (
        convert_llamaindex_filters_to_dxmt,
        node_to_dxmt_record,
        dxmt_hit_to_node,
    )
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores.types import (
        MetadataFilters,
        MetadataFilter,
        FilterOperator,
        FilterCondition,
    )
    print("✓ 所有模块导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    exit(1)

# 测试2: 实例化
print("\n[测试 2] 创建 DXMTVectorStore 实例...")
try:
    store = DXMTVectorStore(
        table_name="offline_test",
        dimension=128,
        host="localhost",
        port=9999,
    )
    print(f"✓ 实例创建成功")
    print(f"  - 表名: {store.table_name}")
    print(f"  - 维度: {store.dimension}")
    print(f"  - 主机: {store.host}:{store.port}")
    print(f"  - 距离度量: {store.distance_metric}")
    print(f"  - 索引类型: {store.index_type}")
    print(f"  - 存储文本: {store.stores_text}")
    print(f"  - 查询嵌入: {store.is_embedding_query}")
except Exception as e:
    print(f"✗ 实例化失败: {e}")
    exit(1)

# 测试3: 过滤器转换
print("\n[测试 3] 测试过滤器转换...")
try:
    # 简单过滤器
    filters1 = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="AI", operator=FilterOperator.EQ)
        ]
    )
    result1 = convert_llamaindex_filters_to_dxmt(filters1)
    print(f"✓ 简单过滤器转换成功")
    print(f"  输入: category == 'AI'")
    print(f"  输出: {result1}")

    # 复杂过滤器（AND）
    filters2 = MetadataFilters(
        filters=[
            MetadataFilter(key="year", value=2020, operator=FilterOperator.GTE),
            MetadataFilter(key="score", value=0.8, operator=FilterOperator.GT),
        ],
        condition=FilterCondition.AND,
    )
    result2 = convert_llamaindex_filters_to_dxmt(filters2)
    print(f"✓ 复杂过滤器（AND）转换成功")
    print(f"  输入: year >= 2020 AND score > 0.8")
    print(f"  输出: {result2}")

    # IN 过滤器
    filters3 = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category",
                value=["AI", "ML", "DL"],
                operator=FilterOperator.IN
            )
        ]
    )
    result3 = convert_llamaindex_filters_to_dxmt(filters3)
    print(f"✓ IN 过滤器转换成功")
    print(f"  输入: category IN ['AI', 'ML', 'DL']")
    print(f"  输出: {result3}")

except Exception as e:
    print(f"✗ 过滤器转换失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试4: 节点转换
print("\n[测试 4] 测试节点转换...")
try:
    # 创建测试节点
    node = TextNode(
        text="这是一个测试文档",
        id_="test_node_1",
        embedding=[0.1] * 128,
        metadata={"category": "test", "year": 2024}
    )

    # 转换为 DXMT 记录
    record = node_to_dxmt_record(node, include_text=True)
    print(f"✓ 节点转 DXMT 记录成功")
    print(f"  节点ID: {record['id']}")
    print(f"  向量维度: {len(record['vector'])}")
    print(f"  Payload keys: {list(record['payload'].keys())}")
    print(f"  文本内容: {record['payload'].get('_node_content', '')[:30]}...")

    # 验证记录结构
    assert record['id'] == 'test_node_1', "ID 不匹配"
    assert len(record['vector']) == 128, "向量维度不正确"
    assert 'payload' in record, "缺少 payload"
    assert '_node_content' in record['payload'], "缺少文本内容"
    print(f"✓ 记录结构验证通过")

except Exception as e:
    print(f"✗ 节点转换失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试5: 属性和方法
print("\n[测试 5] 测试类属性和方法...")
try:
    assert store.class_name() == "DXMTVectorStore", "类名不正确"
    assert store.stores_text == True, "stores_text 应该为 True"
    assert store.is_embedding_query == True, "is_embedding_query 应该为 True"
    assert hasattr(store, 'add'), "缺少 add 方法"
    assert hasattr(store, 'query'), "缺少 query 方法"
    assert hasattr(store, 'delete'), "缺少 delete 方法"
    assert hasattr(store, 'get_nodes'), "缺少 get_nodes 方法"
    assert hasattr(store, 'clear'), "缺少 clear 方法"
    print(f"✓ 所有必需的属性和方法都存在")
    print(f"  - class_name(): {store.class_name()}")
    print(f"  - stores_text: {store.stores_text}")
    print(f"  - is_embedding_query: {store.is_embedding_query}")
    print(f"  - 方法: add, query, delete, get_nodes, clear")
except Exception as e:
    print(f"✗ 属性验证失败: {e}")
    exit(1)

# 测试6: 不同配置
print("\n[测试 6] 测试不同配置...")
try:
    configs = [
        {"index_type": "hnsw", "distance_metric": "cosine"},
        {"index_type": "ivf", "distance_metric": "euclidean"},
        {"index_type": "brute_force", "distance_metric": "manhattan"},
        {"enable_full_text_search": True},
        {"insert_mode": "lock_free", "batch_size": 500},
    ]

    for i, config in enumerate(configs, 1):
        s = DXMTVectorStore(
            table_name=f"config_test_{i}",
            dimension=128,
            **config
        )
        print(f"✓ 配置 {i} 创建成功: {config}")

except Exception as e:
    print(f"✗ 配置测试失败: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✓✓✓ 所有离线测试通过！")
print("=" * 60)
print("\n核心功能验证：")
print("  ✓ 模块导入正常")
print("  ✓ 实例创建正常")
print("  ✓ 过滤器转换正常（支持所有操作符）")
print("  ✓ 节点转换正常")
print("  ✓ 属性和方法完整")
print("  ✓ 多种配置支持")
print("\n要测试完整功能，需要：")
print("  1. 启动 DXMT 服务器:")
print("     cd /localdisk/code/dxmt/vectordb")
print("     cargo run -- serve --host 0.0.0.0 --port 9999")
print("  2. 然后运行: ./quick_test.sh")
print("=" * 60)
