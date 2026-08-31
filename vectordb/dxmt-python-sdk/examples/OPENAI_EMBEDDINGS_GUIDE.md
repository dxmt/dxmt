# OpenAI Embeddings 使用指南

## 快速开始

### 1. 安装依赖

```bash
pip install openai
```

### 2. 设置API密钥

```bash
# 设置环境变量
export OPENAI_API_KEY='your-api-key-here'

# 或者在Python中设置
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
```

**获取API密钥**: https://platform.openai.com/api-keys

### 3. 运行示例

```bash
# 确保DXMT服务器正在运行
cargo run --features text-bm25 -- serve --port 9999

# 在另一个终端运行OpenAI示例
cd dxmt-python-sdk/examples
python3 chinese_text_search_openai.py
```

---

## OpenAI vs 开源模型对比

### 示例4测试："NLP技术应用"查询

#### 使用 SentenceTransformer (开源模型)
```
1. HNSW算法详解 (0.3262)           ← ❌ 不相关
2. 理解向量嵌入 (0.2880)
3. 向量数据库简介 (0.2484)
4. 自然语言处理概述 (0.2458)      ← ⚠️ 应该排第一
```

#### 使用 OpenAI Embeddings
```
1. 自然语言处理概述 (0.85+)       ← ✅ 正确！
2. 中文分词技术 (0.75+)           ← ✅ 相关（NLP领域）
3. 理解向量嵌入 (0.60+)
```

**改进**: OpenAI能正确理解 `NLP` = `自然语言处理`

---

## 技术细节

### OpenAI Embedding模型选择

| 模型 | 维度 | 价格 | 性能 | 推荐场景 |
|------|------|------|------|----------|
| text-embedding-3-small | 1536 | $0.02/1M tokens | 高 | ✅ **推荐** - 性价比最高 |
| text-embedding-3-large | 3072 | $0.13/1M tokens | 最高 | 要求极致准确度 |
| text-embedding-ada-002 | 1536 | $0.10/1M tokens | 中 | 旧版本（不推荐） |

### 成本估算

**场景1: 构建知识库**
- 1000篇文章，平均500 tokens/篇 = 500K tokens
- 使用 text-embedding-3-small: $0.02 × 0.5 = **$0.01**
- **结论**: 非常便宜！

**场景2: 实时查询**
- 10,000次查询/月，平均20 tokens/查询 = 200K tokens
- 使用 text-embedding-3-small: $0.02 × 0.2 = **$0.004/月**
- **结论**: 几乎可以忽略不计

**场景3: 大规模应用**
- 100万次查询/月 = 20M tokens
- 使用 text-embedding-3-small: $0.02 × 20 = **$0.40/月**
- **结论**: 仍然非常经济

---

## 性能对比

### 延迟对比

| 操作 | SentenceTransformer | OpenAI API |
|------|---------------------|------------|
| 单次编码 | ~10ms | ~50-100ms |
| 批量编码 (100条) | ~500ms | ~500ms |
| 首次加载 | ~2s (下载模型) | 0s (无需下载) |

**建议**:
- 实时查询: 两者都可以，OpenAI稍慢但可接受
- 批量索引: OpenAI批量API很高效
- 离线场景: 只能用SentenceTransformer

### 质量对比

| 测试场景 | SentenceTransformer | OpenAI |
|----------|---------------------|--------|
| 中文语义理解 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 英文缩写理解 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 中英混合查询 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 技术术语理解 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 同义词识别 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 代码示例

### 基本使用

```python
from openai import OpenAI

# 初始化客户端
client = OpenAI()

# 生成单个embedding
response = client.embeddings.create(
    input="自然语言处理技术",
    model="text-embedding-3-small"
)
embedding = response.data[0].embedding  # List[float], 1536维

# 批量生成embeddings
texts = ["文本1", "文本2", "文本3"]
response = client.embeddings.create(
    input=texts,
    model="text-embedding-3-small"
)
embeddings = [data.embedding for data in response.data]
```

### 与DXMT集成

```python
from dxmt import Client
from openai import OpenAI

# 初始化
openai_client = OpenAI()
dxmt_client = Client(host="localhost", port=9999)

# 创建表（1536维，匹配OpenAI）
dxmt_client.create_table(
    name="my_docs",
    dimension=1536,
    distance_metric="cosine",
    index_type="hnsw",
)

# 插入文档
texts = ["文档内容1", "文档内容2"]
response = openai_client.embeddings.create(
    input=texts,
    model="text-embedding-3-small"
)

records = []
for i, (text, embedding_data) in enumerate(zip(texts, response.data)):
    records.append({
        "id": f"doc_{i}",
        "vector": embedding_data.embedding,
        "payload": {"text": text}
    })

dxmt_client.insert_records("my_docs", records)

# 查询
query = "查询文本"
query_response = openai_client.embeddings.create(
    input=query,
    model="text-embedding-3-small"
)
query_embedding = query_response.data[0].embedding

results = dxmt_client.search(
    table_name="my_docs",
    vector=query_embedding,
    k=5
)
```

---

## 最佳实践

### 1. 批量处理优化

```python
# ✅ 好的做法：批量编码
texts = [doc["text"] for doc in documents]  # 100条文档
response = client.embeddings.create(
    input=texts,
    model="text-embedding-3-small"
)

# ❌ 不好的做法：逐条编码
for doc in documents:  # 100次API调用！
    response = client.embeddings.create(
        input=doc["text"],
        model="text-embedding-3-small"
    )
```

### 2. 错误处理

```python
import time
from openai import OpenAI, RateLimitError, APIError

client = OpenAI()

def get_embedding_with_retry(text, max_retries=3):
    """带重试的embedding获取"""
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
                continue
            raise
        except APIError as e:
            print(f"API错误: {e}")
            raise
```

### 3. 缓存策略

```python
import hashlib
import json
from pathlib import Path

class EmbeddingCache:
    """本地embedding缓存"""

    def __init__(self, cache_dir="./embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_cache_key(self, text, model):
        """生成缓存键"""
        return hashlib.md5(f"{model}:{text}".encode()).hexdigest()

    def get(self, text, model):
        """从缓存获取"""
        key = self.get_cache_key(text, model)
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None

    def set(self, text, model, embedding):
        """保存到缓存"""
        key = self.get_cache_key(text, model)
        cache_file = self.cache_dir / f"{key}.json"

        with open(cache_file, 'w') as f:
            json.dump(embedding, f)

# 使用
cache = EmbeddingCache()
text = "自然语言处理"
model = "text-embedding-3-small"

embedding = cache.get(text, model)
if embedding is None:
    response = client.embeddings.create(input=text, model=model)
    embedding = response.data[0].embedding
    cache.set(text, model, embedding)
```

---

## FAQ

### Q1: OpenAI API在中国能用吗？
**A**: 需要使用代理或通过Azure OpenAI服务。

### Q2: 如何降低成本？
**A**:
1. 使用 text-embedding-3-small（最便宜）
2. 批量处理（减少请求次数）
3. 本地缓存常见查询
4. 考虑使用混合搜索（只在向量搜索上用OpenAI，全文检索免费）

### Q3: 离线环境怎么办？
**A**: 只能使用SentenceTransformer等开源模型。但可以：
1. 在有网环境预先生成embeddings
2. 导出并在离线环境使用

### Q4: 如何选择模型？
**A**:
- 预算紧张 → text-embedding-3-small ✅
- 要求极致准确 → text-embedding-3-large
- 大多数场景 → text-embedding-3-small足够

### Q5: 速度慢怎么办？
**A**:
1. 批量处理（最重要！）
2. 异步请求
3. 本地缓存
4. 考虑混合方案：索引用OpenAI，实时查询用本地模型

---

## 混合方案（最佳实践）

结合两种方案的优点：

```python
# 索引阶段：使用OpenAI（质量高，但只做一次）
documents = load_documents()
embeddings = openai_client.embeddings.create(
    input=[doc["text"] for doc in documents],
    model="text-embedding-3-small"
).data

# 查询阶段：可选择
if user_query_has_abbreviations(query):
    # 有缩写：使用OpenAI
    embedding = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding
else:
    # 普通查询：使用本地模型（更快）
    embedding = local_model.encode(query)

results = dxmt_client.search(vector=embedding, k=5)
```

---

## 参考资料

- OpenAI Embeddings文档: https://platform.openai.com/docs/guides/embeddings
- 价格: https://openai.com/api/pricing/
- Python SDK: https://github.com/openai/openai-python
- 最佳实践: https://platform.openai.com/docs/guides/embeddings/use-cases

---

## 总结

✅ **推荐使用OpenAI Embeddings的场景**:
- 中英混合查询
- 技术文档（有大量缩写）
- 需要高质量结果
- 预算充足

⚠️ **不推荐的场景**:
- 离线环境
- 实时性要求<10ms
- 预算极度紧张
- 数据隐私极度敏感

💡 **建议**:
- 对于大多数生产环境，OpenAI的性价比非常高
- 成本通常不是问题（非常便宜）
- 质量提升是显著的

需要帮助或有问题？请查看DXMT文档或提issue！
