# OpenAI Embeddings 快速开始

## 5分钟上手指南

### 步骤 1: 获取 OpenAI API 密钥

1. 访问 https://platform.openai.com/api-keys
2. 登录或注册账号
3. 点击 "Create new secret key"
4. 复制密钥（格式：`sk-...`）

**成本**: 新用户通常有免费额度，之后非常便宜（$0.02 / 1M tokens）

### 步骤 2: 设置环境变量

```bash
# Linux/Mac
export OPENAI_API_KEY='sk-your-key-here'

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-your-key-here"

# 或者在 Python 中设置
import os
os.environ['OPENAI_API_KEY'] = 'sk-your-key-here'
```

### 步骤 3: 安装依赖

```bash
pip install openai
```

### 步骤 4: 启动 DXMT 服务器

```bash
# 在另一个终端
cd /path/to/vectordb
cargo run --features text-bm25 -- serve --port 9999
```

### 步骤 5: 运行示例

```bash
cd dxmt-python-sdk/examples
python3 chinese_text_search_openai.py
```

### 预期输出

```
================================================================================
示例 4: 不同查询方式对比（测试缩写理解）
================================================================================

技术缩写 [关键测试]: "NLP技术应用"
────────────────────────────────────────────────────────────────────────────────
1. 自然语言处理概述 (相似度: 0.8547)  ← ✅ 正确！
2. 中文分词技术 (相似度: 0.7621)
3. 理解向量嵌入 (相似度: 0.6034)

💡 OpenAI改进提示:
   ✅ OpenAI成功理解了'NLP'='自然语言处理'的关系！
```

---

## 对比测试

运行对比脚本查看差异：

```bash
python3 compare_embeddings.py
```

**示例输出**：

```
测试 OpenAI Embeddings (text-embedding-3-small)
================================================================================
✅ NLP vs 自然语言处理: 0.8547 (应该高)
✅ AI vs 人工智能: 0.8621 (应该高)
✅ ML vs 机器学习: 0.8423 (应该高)
✅ NLP vs 不相关术语: 0.3156 (应该低)

测试 SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2)
================================================================================
⚠️  NLP vs 自然语言处理: 0.4521 (应该高)
⚠️  AI vs 人工智能: 0.5234 (应该高)
⚠️  ML vs 机器学习: 0.4789 (应该高)
✅ NLP vs 不相关术语: 0.3421 (应该低)

对比总结
================================================================================
测试场景                        OpenAI    SentenceT        差异
--------------------------------------------------------------------------------
🎯 NLP vs 自然语言处理          0.8547         0.4521    +0.4026
🎯 AI vs 人工智能               0.8621         0.5234    +0.3387
🎯 ML vs 机器学习               0.8423         0.4789    +0.3634

✅ OpenAI在以下场景表现更好:
   • NLP vs 自然语言处理
   • AI vs 人工智能
   • ML vs 机器学习
```

---

## 集成到您的应用

### 方案 1: 直接使用 OpenAI

```python
from openai import OpenAI
from dxmt import Client

# 初始化
openai_client = OpenAI()
dxmt_client = Client()

# 创建表 (1536维 for text-embedding-3-small)
dxmt_client.create_table(
    name="docs",
    dimension=1536,  # OpenAI 维度
    distance_metric="cosine",
)

# 插入文档
texts = ["文档1", "文档2", "文档3"]
response = openai_client.embeddings.create(
    input=texts,
    model="text-embedding-3-small"
)

records = [
    {
        "id": f"doc_{i}",
        "vector": data.embedding,
        "payload": {"text": text}
    }
    for i, (text, data) in enumerate(zip(texts, response.data))
]

dxmt_client.insert_records("docs", records)

# 查询
query = "搜索查询"
query_response = openai_client.embeddings.create(
    input=query,
    model="text-embedding-3-small"
)

results = dxmt_client.search(
    table_name="docs",
    vector=query_response.data[0].embedding,
    k=5
)
```

### 方案 2: 使用示例中的 OpenAIEmbedder 类

```python
from chinese_text_search_openai import OpenAIEmbedder
from dxmt import Client

# 初始化
embedder = OpenAIEmbedder(model="text-embedding-3-small")
client = Client()

# 创建表
client.create_table(
    name="docs",
    dimension=embedder.dimension,  # 自动获取
    distance_metric="cosine",
)

# 批量编码（自动批处理）
texts = ["文档1", "文档2", "文档3"]
embeddings = embedder.encode(texts)

# 插入
records = [
    {
        "id": f"doc_{i}",
        "vector": embedding,
        "payload": {"text": text}
    }
    for i, (text, embedding) in enumerate(zip(texts, embeddings))
]
client.insert_records("docs", records)

# 查询
query_embedding = embedder.encode(["搜索查询"])[0]
results = client.search("docs", vector=query_embedding, k=5)
```

---

## 常见问题

### Q: 成本是多少？

**A**: 非常便宜！
- text-embedding-3-small: **$0.02 / 1M tokens**
- 1个中文字符 ≈ 2-3 tokens
- 1000次查询（每次20字） ≈ 40K-60K tokens ≈ **$0.001-0.002**

**示例计算**:
- 10,000篇文章索引（每篇500字）= 10M-15M tokens = **$0.20-0.30**
- 100,000次查询（每次20字）= 4M-6M tokens = **$0.08-0.12**

### Q: 延迟多少？

**A**:
- 单次查询: 50-100ms
- 批量查询（10条）: 200-300ms
- 批量查询（100条）: 500-800ms

**优化建议**: 使用批量API可以大幅降低平均延迟。

### Q: 在中国能用吗？

**A**:
- 直接使用: 需要代理或VPN
- 通过Azure: 可以使用 Azure OpenAI Service（需要申请）
- 替代方案: 使用支持中国的embedding服务或本地模型

### Q: 数据安全吗？

**A**:
- OpenAI承诺不使用API数据训练模型
- 数据通过HTTPS加密传输
- 建议: 敏感数据使用本地模型（SentenceTransformer）

### Q: 可以离线使用吗？

**A**:
- 不可以，OpenAI需要网络连接
- 离线场景请使用 SentenceTransformer
- 可以考虑混合方案：索引时用OpenAI，查询时用本地模型

---

## 性能优化技巧

### 1. 批量处理

```python
# ✅ 好：批量处理
embeddings = embedder.encode(texts)  # 一次API调用

# ❌ 差：逐个处理
embeddings = [embedder.encode([t])[0] for t in texts]  # N次API调用
```

### 2. 本地缓存

```python
import hashlib
import json

def get_cached_embedding(text, cache_dir="./cache"):
    # 生成缓存键
    key = hashlib.md5(text.encode()).hexdigest()
    cache_file = f"{cache_dir}/{key}.json"

    # 检查缓存
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    # 生成新embedding
    embedding = embedder.encode([text])[0]

    # 保存缓存
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(embedding, f)

    return embedding
```

### 3. 错误重试

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def get_embedding_with_retry(text):
    return embedder.encode([text])[0]
```

---

## 下一步

1. ✅ 运行 `compare_embeddings.py` 查看实际差异
2. 📖 阅读 [OPENAI_EMBEDDINGS_GUIDE.md](OPENAI_EMBEDDINGS_GUIDE.md) 了解详细信息
3. 🚀 在您的应用中集成 OpenAI embeddings
4. 💡 如有问题，查看 GitHub Issues 或联系支持

---

**成本计算器**: https://openai.com/api/pricing/
**API文档**: https://platform.openai.com/docs/guides/embeddings
**DXMT文档**: https://docs.dxmt.online
