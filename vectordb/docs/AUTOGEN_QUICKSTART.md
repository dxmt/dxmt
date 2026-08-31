# DXMT-AutoGen 集成快速开始指南

**5 分钟快速上手 DXMT VectorDB 与 AutoGen 集成**

---

## 前置要求

- Python 3.10+
- Rust 工具链 (用于编译 DXMT)
- (可选) OpenAI API Key

---

## Step 1: 启动 DXMT 服务器

```bash
# 在 DXMT 项目根目录
cargo run -- serve --host 0.0.0.0 --port 9999
```

**输出应显示:**
```
🚀 DXMT VectorDB Server started
📍 Listening on http://0.0.0.0:9999
```

---

## Step 2: 安装依赖

```bash
# 安装 AutoGen 和 DXMT Python SDK
pip install autogen-ext autogen-agentchat

# 安装 DXMT Python SDK (从源码)
cd dxmt-python-sdk
pip install -e .
```

---

## Step 3: 基础使用

创建文件 `test_dxmt_memory.py`:

```python
import asyncio
from autogen_core.memory import MemoryContent
from autogen_ext.memory.dxmt import DXMTVectorMemory, HttpDXMTVectorMemoryConfig

async def main():
    # 配置 DXMT 内存
    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="my_first_memory",
        k=3  # 返回前 3 个结果
    )

    memory = DXMTVectorMemory(config=config)

    print("✅ 连接到 DXMT 服务器成功！\n")

    # 添加一些记忆
    print("📝 添加记忆...")
    memories = [
        "User's name is Alice",
        "Alice lives in San Francisco",
        "Alice is a Python developer",
        "Alice enjoys hiking"
    ]

    for content in memories:
        await memory.add(MemoryContent(
            content=content,
            mime_type="text/plain",
            metadata={"user_id": "alice_123"}
        ))
        print(f"  ✓ {content}")

    # 查询记忆
    print("\n🔍 查询记忆...")
    queries = [
        "What is the user's name?",
        "Where does Alice live?",
        "What does Alice do?"
    ]

    for query in queries:
        print(f"\n  Query: {query}")
        results = await memory.query(query, user_id="alice_123")

        for i, result in enumerate(results.results, 1):
            score = result.metadata.get("score", 0.0)
            print(f"    {i}. {result.content} (score: {score:.3f})")

    # 清理
    await memory.clear()
    await memory.close()

    print("\n✅ 测试完成！")

if __name__ == "__main__":
    asyncio.run(main())
```

**运行:**
```bash
python test_dxmt_memory.py
```

**预期输出:**
```
✅ 连接到 DXMT 服务器成功！

📝 添加记忆...
  ✓ User's name is Alice
  ✓ Alice lives in San Francisco
  ✓ Alice is a Python developer
  ✓ Alice enjoys hiking

🔍 查询记忆...

  Query: What is the user's name?
    1. User's name is Alice (score: 0.892)
    2. Alice is a Python developer (score: 0.654)
    3. Alice lives in San Francisco (score: 0.621)

  Query: Where does Alice live?
    1. Alice lives in San Francisco (score: 0.901)
    2. Alice enjoys hiking (score: 0.543)
    3. User's name is Alice (score: 0.487)

  Query: What does Alice do?
    1. Alice is a Python developer (score: 0.878)
    2. Alice enjoys hiking (score: 0.712)
    3. User's name is Alice (score: 0.456)

✅ 测试完成！
```

---

## Step 4: 与 AutoGen Agent 集成

创建文件 `test_agent_memory.py`:

```python
import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.memory import MemoryContent
from autogen_ext.memory.dxmt import DXMTVectorMemory, HttpDXMTVectorMemoryConfig

async def main():
    # 检查 OpenAI API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  请设置 OPENAI_API_KEY 环境变量")
        return

    # 配置内存
    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="agent_memory",
        k=3
    )

    memory = DXMTVectorMemory(config=config)

    # 添加背景知识
    print("📝 添加背景知识到记忆...")
    knowledge = [
        "The user's company uses Python and Django for backend",
        "The team is migrating from PostgreSQL to DynamoDB",
        "The main project is a real-time analytics platform",
    ]

    for content in knowledge:
        await memory.add(MemoryContent(
            content=content,
            mime_type="text/plain"
        ))
        print(f"  ✓ {content}")

    # 创建带记忆的 Agent
    print("\n🤖 创建 Agent...")
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        memory=[memory],
        system_message="You are a helpful assistant with memory."
    )

    # 测试对话
    print("\n💬 开始对话...\n")

    questions = [
        "What technology stack does my company use?",
        "Tell me about the database migration",
    ]

    for question in questions:
        print(f"User: {question}")
        result = await agent.run(task=question)
        print(f"Agent: {result.messages[-1].content}\n")

    # 清理
    await memory.clear()
    await memory.close()

    print("✅ 测试完成！")

if __name__ == "__main__":
    asyncio.run(main())
```

**运行:**
```bash
export OPENAI_API_KEY="sk-..."
python test_agent_memory.py
```

**预期输出:**
```
📝 添加背景知识到记忆...
  ✓ The user's company uses Python and Django for backend
  ✓ The team is migrating from PostgreSQL to DynamoDB
  ✓ The main project is a real-time analytics platform

🤖 创建 Agent...

💬 开始对话...

User: What technology stack does my company use?
Agent: Based on the memory, your company uses Python and Django
for the backend. The main project is a real-time analytics platform.

User: Tell me about the database migration
Agent: Your team is currently migrating from PostgreSQL to DynamoDB.
This is likely to support the real-time analytics platform with better
scalability and performance.

✅ 测试完成！
```

---

## Step 5: 多用户记忆隔离

创建文件 `test_multi_user.py`:

```python
import asyncio
from autogen_core.memory import MemoryContent
from autogen_ext.memory.dxmt import DXMTVectorMemory, HttpDXMTVectorMemoryConfig

async def main():
    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="multi_user_memory"
    )

    memory = DXMTVectorMemory(config=config)

    # 为不同用户添加记忆
    print("📝 添加多用户记忆...\n")

    users = {
        "alice": ["Alice prefers Python", "Alice works at Google"],
        "bob": ["Bob likes Java", "Bob works at Microsoft"],
        "charlie": ["Charlie uses Rust", "Charlie works at Mozilla"]
    }

    for user_id, memories in users.items():
        print(f"User: {user_id}")
        for content in memories:
            await memory.add(MemoryContent(
                content=content,
                mime_type="text/plain",
                metadata={"user_id": user_id}
            ))
            print(f"  ✓ {content}")
        print()

    # 按用户查询
    print("🔍 按用户查询...\n")

    for user_id in ["alice", "bob", "charlie"]:
        print(f"Query for {user_id}: What programming language?")
        results = await memory.query(
            "programming language",
            user_id=user_id,
            limit=2
        )

        for result in results.results:
            score = result.metadata.get("score", 0.0)
            print(f"  → {result.content} (score: {score:.3f})")
        print()

    # 清理
    await memory.clear()
    await memory.close()

    print("✅ 测试完成！")

if __name__ == "__main__":
    asyncio.run(main())
```

**运行:**
```bash
python test_multi_user.py
```

**预期输出:**
```
📝 添加多用户记忆...

User: alice
  ✓ Alice prefers Python
  ✓ Alice works at Google

User: bob
  ✓ Bob likes Java
  ✓ Bob works at Microsoft

User: charlie
  ✓ Charlie uses Rust
  ✓ Charlie works at Mozilla

🔍 按用户查询...

Query for alice: What programming language?
  → Alice prefers Python (score: 0.887)
  → Alice works at Google (score: 0.456)

Query for bob: What programming language?
  → Bob likes Java (score: 0.892)
  → Bob works at Microsoft (score: 0.421)

Query for charlie: What programming language?
  → Charlie uses Rust (score: 0.901)
  → Charlie works at Mozilla (score: 0.398)

✅ 测试完成！
```

---

## 常见问题

### 1. Connection refused 错误

**问题:**
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**解决:**
确保 DXMT 服务器正在运行:
```bash
cargo run -- serve --host 0.0.0.0 --port 9999
```

### 2. 模块未找到

**问题:**
```
ModuleNotFoundError: No module named 'dxmt'
```

**解决:**
安装 DXMT Python SDK:
```bash
cd dxmt-python-sdk
pip install -e .
```

### 3. 查询返回空结果

**问题:** 添加了记忆但查询返回空结果

**解决:**
1. 检查表是否成功创建: `curl http://localhost:9999/tables`
2. 降低或移除 `score_threshold`
3. 检查 `user_id` 等元数据过滤器是否正确

---

## 下一步

### 运行完整示例

```bash
# 运行完整的基础示例
python examples/autogen_dxmt_basic.py
```

### 运行测试

```bash
# 安装 pytest
pip install pytest pytest-asyncio

# 运行测试
pytest tests/test_dxmt_autogen.py -v
```

### 阅读文档

- **使用指南:** `reference/autogen/.../dxmt/README.md`
- **设计文档:** `DXMT_AUTOGEN_INTEGRATION_DESIGN.md`
- **完成总结:** `AUTOGEN_INTEGRATION_PHASE1_COMPLETE.md`

---

## 配置选项

### 启用全文搜索

```python
config = HttpDXMTVectorMemoryConfig(
    enable_full_text_search=True  # 使用 Tantivy + BM25
)
```

### 设置分数阈值

```python
config = HttpDXMTVectorMemoryConfig(
    score_threshold=0.5  # 只返回 score >= 0.5 的结果
)
```

### 调整返回数量

```python
config = HttpDXMTVectorMemoryConfig(
    k=10  # 返回前 10 个结果
)

# 或在查询时覆盖
results = await memory.query("query", limit=5)
```

---

## 性能优化

### DXMT 服务器配置

在 `.env` 文件中:

```bash
# 存储配置
DB_STORAGE_TYPE=tiered  # 使用分层存储
DB_MEMORY_CACHE_SIZE=1024  # 1GB 内存缓存

# S3 配置 (用于持久化)
DB_OBJECT_STORAGE_PROVIDER=s3
DB_S3_BUCKET=my-memory-bucket
DB_S3_REGION=us-east-1

# 压缩
DB_COMPRESSION=zstd
```

### Python 客户端优化

```python
config = HttpDXMTVectorMemoryConfig(
    timeout=60,  # 增加超时时间
    k=5,  # 减少返回数量
    score_threshold=0.3  # 过滤低相关度结果
)
```

---

## 成功！🎉

你现在已经成功集成了 DXMT VectorDB 与 AutoGen！

**你可以:**
- ✅ 添加和查询记忆
- ✅ 创建带记忆的 AutoGen Agent
- ✅ 支持多用户记忆隔离
- ✅ 使用全文搜索
- ✅ 利用分层存储节省成本

**继续学习:**
- 查看更多示例: `examples/`
- 阅读完整文档: `reference/autogen/.../dxmt/README.md`
- 运行测试: `pytest tests/test_dxmt_autogen.py`

---

**需要帮助?**
- GitHub Issues: https://github.com/dxmt/dxmt/issues
- 文档: `DXMT_AUTOGEN_INTEGRATION_DESIGN.md`
