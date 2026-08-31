#!/bin/bash
# LlamaIndex DXMT 集成快速测试脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "LlamaIndex DXMT 集成快速测试"
echo "=========================================="
echo ""

# 检查是否在正确的目录
if [ ! -f "pyproject.toml" ]; then
    echo "错误：请在 llama-index-vector-stores-dxmt 目录下运行此脚本"
    exit 1
fi

# 激活虚拟环境
if [ ! -d "venv" ]; then
    echo "错误：虚拟环境不存在，请先安装："
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -e ../dxmt-python-sdk"
    echo "  pip install -e ."
    exit 1
fi

source venv/bin/activate

echo "✓ 虚拟环境已激活"
echo ""

# 检查服务器连接
echo "[1/5] 检查 DXMT 服务器连接..."
if curl -s http://localhost:9999/health > /dev/null 2>&1; then
    echo "✓ DXMT 服务器运行正常"
else
    echo "✗ 无法连接到 DXMT 服务器"
    echo ""
    echo "请在另一个终端启动服务器："
    echo "  cd /localdisk/code/dxmt/vectordb"
    echo "  cargo run -- serve --host 0.0.0.0 --port 9999"
    exit 1
fi
echo ""

# 测试导入
echo "[2/5] 测试导入..."
python << 'EOF'
try:
    from llama_index.vector_stores.dxmt import DXMTVectorStore
    from llama_index.vector_stores.dxmt import (
        convert_llamaindex_filters_to_dxmt,
        node_to_dxmt_record,
        dxmt_hit_to_node,
    )
    print("✓ 所有模块导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    exit(1)
EOF
echo ""

# 测试实例化
echo "[3/5] 测试实例化..."
python << 'EOF'
try:
    from llama_index.vector_stores.dxmt import DXMTVectorStore

    store = DXMTVectorStore(
        table_name="test_quick",
        dimension=128,
        host="localhost",
        port=9999,
    )

    print(f"✓ 实例创建成功")
    print(f"  - 表名: {store.table_name}")
    print(f"  - 维度: {store.dimension}")
    print(f"  - 类名: {store.class_name()}")
except Exception as e:
    print(f"✗ 实例化失败: {e}")
    exit(1)
EOF
echo ""

# 测试基本功能
echo "[4/5] 测试基本功能（添加、查询、删除）..."
python << 'EOF'
try:
    from llama_index.vector_stores.dxmt import DXMTVectorStore
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores.types import VectorStoreQuery

    # 创建向量存储
    store = DXMTVectorStore(
        table_name="quick_test_functional",
        dimension=128,
        host="localhost",
        port=9999,
    )

    # 添加节点
    nodes = [
        TextNode(
            text="测试文档 1",
            id_="node1",
            embedding=[0.1] * 128,
            metadata={"category": "test", "id": 1}
        ),
        TextNode(
            text="测试文档 2",
            id_="node2",
            embedding=[0.2] * 128,
            metadata={"category": "test", "id": 2}
        ),
    ]

    ids = store.add(nodes)
    print(f"✓ 成功添加 {len(ids)} 个节点")

    # 查询
    query = VectorStoreQuery(
        query_embedding=[0.15] * 128,
        similarity_top_k=2,
    )
    result = store.query(query)
    print(f"✓ 查询成功，返回 {len(result.nodes)} 个结果")

    # 获取节点
    nodes_retrieved = store.get_nodes(node_ids=["node1"])
    print(f"✓ 成功获取节点")

    # 删除
    store.delete_nodes(node_ids=["node1"])
    print(f"✓ 成功删除节点")

    # 清理
    store.clear()
    print(f"✓ 清理完成")

except Exception as e:
    print(f"✗ 功能测试失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF
echo ""

# 测试过滤器
echo "[5/5] 测试元数据过滤..."
python << 'EOF'
try:
    from llama_index.vector_stores.dxmt import DXMTVectorStore
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores.types import (
        VectorStoreQuery,
        MetadataFilters,
        MetadataFilter,
        FilterOperator,
    )

    # 创建向量存储
    store = DXMTVectorStore(
        table_name="quick_test_filters",
        dimension=128,
        host="localhost",
        port=9999,
    )

    # 添加测试数据
    nodes = [
        TextNode(
            text=f"文档 {i}",
            id_=f"node_{i}",
            embedding=[0.1] * 128,
            metadata={"year": 2020 + i, "category": "test"}
        )
        for i in range(5)
    ]
    store.add(nodes)

    # 测试过滤查询
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="year", value=2022, operator=FilterOperator.GTE)
        ]
    )

    query = VectorStoreQuery(
        query_embedding=[0.1] * 128,
        similarity_top_k=10,
        filters=filters,
    )

    result = store.query(query)

    # 验证过滤结果
    for node in result.nodes:
        year = node.metadata.get("year", 0)
        assert year >= 2022, f"过滤失败：year={year}"

    print(f"✓ 过滤查询成功，返回 {len(result.nodes)} 个符合条件的结果")
    print(f"✓ 过滤验证通过（所有结果 year >= 2022）")

    # 清理
    store.clear()

except Exception as e:
    print(f"✗ 过滤器测试失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF
echo ""

echo "=========================================="
echo "✓✓✓ 所有快速测试通过！✓✓✓"
echo "=========================================="
echo ""
echo "集成工作正常！你可以："
echo "  1. 运行单元测试: pytest tests/test_filters.py -v"
echo "  2. 运行示例: python examples/basic_usage.py"
echo "  3. 查看完整测试指南: cat TESTING_GUIDE_CN.md"
echo ""
