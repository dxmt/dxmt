# DXMT Python SDK - 快速开始

## 安装

```bash
cd /localdisk/code/dxmt/vectordb/dxmt-python-sdk
pip install -e ".[dev]"
```

## 基本使用

### 1. 导入SDK

```python
from dxmt import Client
```

### 2. 创建客户端

```python
# 连接到本地DXMT服务器
client = Client(host="localhost", port=9999)

# 或使用上下文管理器（自动清理）
with Client(host="localhost", port=9999) as client:
    # ... 使用客户端 ...
```

### 3. 创建向量表

```python
client.create_table(
    name="my_documents",
    dimension=1536,           # 向量维度
    distance_metric="cosine", # 距离度量
    index_type="hnsw"         # 索引类型
)
```

### 4. 插入向量

```python
client.insert_records(
    table_name="my_documents",
    records=[
        {
            "id": "doc1",
            "vector": [0.1, 0.2, 0.3, ...],  # 1536维向量
            "metadata": {
                "title": "Document 1",
                "text": "This is document 1 content"
            }
        },
        {
            "id": "doc2",
            "vector": [0.4, 0.5, 0.6, ...],
            "metadata": {
                "title": "Document 2",
                "text": "This is document 2 content"
            }
        }
    ]
)
```

### 5. 搜索相似向量

```python
# 向量搜索
results = client.search(
    table_name="my_documents",
    vector=[0.15, 0.25, 0.35, ...],  # 查询向量
    k=10  # 返回前10个结果
)

# 查看结果
for result in results:
    print(f"ID: {result['id']}")
    print(f"Score: {result['score']}")
    print(f"Title: {result['metadata']['title']}")
    print()
```

### 6. 混合搜索（向量 + 文本）

```python
results = client.hybrid_search(
    table_name="my_documents",
    vector=[0.1, 0.2, ...],        # 向量查询
    text_query="machine learning", # 文本查询
    alpha=0.5,                     # 权重：0=仅文本，1=仅向量
    k=10
)
```

### 7. 清理

```python
# 删除表
client.delete_table("my_documents")

# 关闭连接
client.close()
```

## 异步使用

```python
import asyncio

async def main():
    # 使用异步上下文管理器
    async with Client(host="localhost", port=9999) as client:
        # 创建表
        await client.acreate_table(
            name="async_docs",
            dimension=128
        )

        # 插入记录
        await client.ainsert_records(
            table_name="async_docs",
            records=[...]
        )

        # 搜索
        results = await client.asearch(
            table_name="async_docs",
            vector=[0.1, 0.2, ...],
            k=10
        )

# 运行
asyncio.run(main())
```

## 完整示例

参见 `examples/` 目录：
- `basic_usage.py` - 基本使用
- `async_usage.py` - 异步操作
- `hybrid_search.py` - 混合搜索

## API 参考

完整API文档请参见 `README.md`

## 需要帮助？

- 文档: `README.md`
- 验证报告: `SDK_VERIFICATION_REPORT.md`
- 示例: `examples/`
- GitHub Issues: https://github.com/dxmt/vectordb/issues
