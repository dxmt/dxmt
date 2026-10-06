"""Chinese text search example using OpenAI Embeddings for DXMT Python SDK.

This example demonstrates:
1. Using OpenAI's text-embedding-3-small model for better semantic understanding
2. Improved handling of abbreviations (NLP, AI, ML, etc.)
3. Full-text search with Chinese text using Jieba tokenization
4. Hybrid search combining Chinese text and OpenAI embeddings

Prerequisites:
    pip install openai
    export OPENAI_API_KEY="your-api-key"

Use case:
    - Chinese document search with better abbreviation understanding
    - Chinese Q&A systems
    - Bilingual search systems (Chinese + English abbreviations)
"""

import json
import os
from typing import List, Dict, Any

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from dxmt import Client
from environment import client_kwargs, skip


def create_chinese_articles() -> List[Dict[str, Any]]:
    """Create Chinese articles corpus."""
    articles = [
        {
            "id": "zh1",
            "text": "向量数据库是专门为存储和检索高维向量而设计的数据库系统。它在人工智能和机器学习应用中发挥着重要作用。",
            "metadata": {
                "title": "向量数据库简介",
                "category": "技术",
                "author": "张三",
            },
        },
        {
            "id": "zh2",
            "text": "HNSW（Hierarchical Navigable Small World）是一种基于图的近似最近邻搜索算法，具有优秀的性能和准确性。",
            "metadata": {
                "title": "HNSW算法详解",
                "category": "技术",
                "author": "李四",
            },
        },
        {
            "id": "zh3",
            "text": "语义搜索利用深度学习技术理解查询的真实意图，而不仅仅是匹配关键词。它能够提供更准确的搜索结果。",
            "metadata": {
                "title": "语义搜索的优势",
                "category": "技术",
                "author": "王五",
            },
        },
        {
            "id": "zh4",
            "text": "中文分词是自然语言处理的基础任务。Jieba是一个流行的中文分词工具，支持多种分词模式。",
            "metadata": {
                "title": "中文分词技术",
                "category": "NLP",
                "author": "赵六",
            },
        },
        {
            "id": "zh5",
            "text": "机器学习模型可以从大量数据中学习模式和规律，用于预测、分类和生成任务。",
            "metadata": {
                "title": "机器学习基础",
                "category": "AI",
                "author": "张三",
            },
        },
        {
            "id": "zh6",
            "text": "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的复杂特征表示。",
            "metadata": {
                "title": "深度学习原理",
                "category": "AI",
                "author": "李四",
            },
        },
        {
            "id": "zh7",
            "text": "自然语言处理让计算机能够理解、解释和生成人类语言。它是人工智能的重要应用领域。",
            "metadata": {
                "title": "自然语言处理概述",
                "category": "NLP",
                "author": "王五",
            },
        },
        {
            "id": "zh8",
            "text": "向量嵌入是将文本、图像等数据转换为固定长度的数值向量，以便计算机处理和理解。",
            "metadata": {
                "title": "理解向量嵌入",
                "category": "技术",
                "author": "赵六",
            },
        },
        {
            "id": "zh9",
            "text": "余弦相似度是衡量两个向量之间角度的指标，常用于文本相似度计算和推荐系统。",
            "metadata": {
                "title": "余弦相似度的应用",
                "category": "技术",
                "author": "张三",
            },
        },
        {
            "id": "zh10",
            "text": "全文检索系统使用倒排索引快速查找包含特定关键词的文档，在搜索引擎中广泛应用。",
            "metadata": {
                "title": "全文检索原理",
                "category": "技术",
                "author": "李四",
            },
        },
    ]
    return articles


class OpenAIEmbedder:
    """Wrapper for OpenAI embeddings."""

    def __init__(self, model: str = "text-embedding-3-small"):
        """Initialize OpenAI embedder.

        Args:
            model: OpenAI embedding model name
                - text-embedding-3-small: 1536 dimensions, fast and cost-effective
                - text-embedding-3-large: 3072 dimensions, highest quality
        """
        self.client = OpenAI()
        self.model = model
        self._dimension = None

        # Get dimension by encoding a test string
        test_embedding = self._encode_single("test")
        self._dimension = len(test_embedding)

    def _encode_single(self, text: str) -> List[float]:
        """Encode a single text."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    def encode(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        """Encode a list of texts.

        Args:
            texts: List of texts to encode
            show_progress: Whether to show progress (for compatibility)

        Returns:
            List of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # Batch encode for efficiency
        if show_progress:
            print(f"   Encoding {len(texts)} texts with OpenAI {self.model}...")

        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )

        return [data.embedding for data in response.data]

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


def print_chinese_results(query: str, results: List[Dict[str, Any]], records_map: Dict[str, Any], title: str) -> None:
    """Pretty print Chinese search results."""
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")
    print(f"查询: \"{query}\"")
    print(f"找到 {len(results)} 个结果:\n")

    for i, result in enumerate(results, 1):
        doc_id = result["id"]
        score = result.get("score", 0)

        # Get full document
        doc = records_map.get(doc_id)
        if not doc:
            continue

        # Parse payload
        payload = {}
        if "payload_json" in doc and doc["payload_json"] and doc["payload_json"] != "null":
            payload = json.loads(doc["payload_json"])

        print(f"{i}. {payload.get('title', doc_id)} (分数: {score:.4f})")
        print(f"   {payload.get('content', 'N/A')}")
        print(f"   分类: {payload.get('category', 'N/A')} | 作者: {payload.get('author', 'N/A')}")


def main() -> None:
    """Run Chinese text search examples with OpenAI embeddings."""
    print("=" * 80)
    print("DXMT 中文文本搜索示例 (使用 OpenAI Embeddings)")
    print("=" * 80)

    # Check dependencies
    if not OPENAI_AVAILABLE:
        skip("未安装 openai: pip install openai")

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        skip("未设置 OPENAI_API_KEY 环境变量: export OPENAI_API_KEY=...")

    # Initialize OpenAI embedder
    print("\n初始化 OpenAI embedding 模型...")
    print("   模型: text-embedding-3-small")
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    embedding_dim = embedder.dimension
    print(f"✅ 模型初始化完成 (维度: {embedding_dim})")

    # Create client
    print("\n连接 DXMT 服务器...")
    client = Client(**client_kwargs())

    try:
        # Create table with full-text search enabled
        print("\n创建中文文章表...")
        try:
            client.delete_table("chinese_articles_openai")
        except Exception:
            pass

        client.create_table(
            name="chinese_articles_openai",
            dimension=embedding_dim,
            distance_metric="cosine",
            index_type="hnsw",
            enable_full_text_search=True,
        )
        print("✅ 表创建成功")

        # Load Chinese articles
        print("\n加载中文文章...")
        articles = create_chinese_articles()

        # Generate embeddings using OpenAI
        print("\n使用 OpenAI API 生成嵌入向量...")
        texts = [article["text"] for article in articles]
        embeddings = embedder.encode(texts, show_progress=True)
        print(f"✅ 生成 {len(embeddings)} 个嵌入向量")

        # Prepare records
        records = []
        for article, embedding in zip(articles, embeddings):
            records.append({
                "id": article["id"],
                "vector": embedding,
                "payload": {
                    "content": article["text"],
                    "title": article["metadata"]["title"],
                    "category": article["metadata"]["category"],
                    "author": article["metadata"]["author"],
                },
            })

        # Insert records
        print("\n插入文章到 DXMT...")
        client.insert_records(table_name="chinese_articles_openai", records=records)
        print(f"✅ 插入 {len(records)} 篇文章")

        # Get all records for display
        all_records = client.get_records(table_name="chinese_articles_openai", ids=[a["id"] for a in articles])
        records_map = {record["id"]: record for record in all_records}

        # ================================================================
        # Example 1: Full-Text Search (Text Only)
        # ================================================================
        print("\n" + "=" * 80)
        print("示例 1: 全文检索（仅文本匹配）")
        print("=" * 80)

        text_query = "向量数据库"
        print(f"\n使用 Jieba 分词进行中文全文检索...")

        results = client.full_text_search(
            table_name="chinese_articles_openai",
            text_query=text_query,
            limit=3,
        )

        if results:
            print_chinese_results(text_query, results, records_map, "全文检索结果")
        else:
            print(f"\n未找到包含 '{text_query}' 的文章")

        # ================================================================
        # Example 2: Semantic Search (Vector Only)
        # ================================================================
        print("\n" + "=" * 80)
        print("示例 2: 语义搜索（仅向量相似度 - OpenAI）")
        print("=" * 80)

        semantic_query = "什么是向量存储和查询？"
        print(f"\n使用 OpenAI 模型进行语义搜索...")

        query_embedding = embedder.encode([semantic_query])[0]

        results = client.search(
            table_name="chinese_articles_openai",
            vector=query_embedding,
            k=3,
        )

        print_chinese_results(semantic_query, results, records_map, "语义搜索结果（OpenAI）")

        # ================================================================
        # Example 3: Hybrid Search (Text + Vector)
        # ================================================================
        print("\n" + "=" * 80)
        print("示例 3: 混合搜索（文本 + 向量 - OpenAI）")
        print("=" * 80)

        hybrid_query_text = "机器学习"
        hybrid_query_embedding = embedder.encode([hybrid_query_text])[0]

        print(f"\n混合搜索结合全文检索和语义相似度...")

        # Try different alpha values
        alphas = [0.3, 0.5, 0.7]

        for alpha in alphas:
            print(f"\n{'─' * 80}")
            print(f"Alpha = {alpha} ({'偏重文本' if alpha < 0.5 else '偏重向量' if alpha > 0.5 else '平衡'})")
            print(f"{'─' * 80}")

            results = client.hybrid_search(
                table_name="chinese_articles_openai",
                vector=hybrid_query_embedding,
                text_query=hybrid_query_text,
                alpha=alpha,
                limit=3,
            )

            for i, result in enumerate(results, 1):
                doc = records_map.get(result["id"])
                if doc and "payload_json" in doc:
                    payload = json.loads(doc["payload_json"])
                    print(f"{i}. {payload['title']} (分数: {result.get('score', 0):.4f})")

        # ================================================================
        # Example 4: Different Query Styles (Testing Abbreviations)
        # ================================================================
        print("\n" + "=" * 80)
        print("示例 4: 不同查询方式对比（测试缩写理解）")
        print("=" * 80)

        test_queries = [
            ("图算法", "关键词查询"),
            ("如何提高搜索速度？", "自然问题"),
            ("NLP技术应用", "技术缩写 [关键测试]"),
        ]

        for query_text, query_type in test_queries:
            print(f"\n{'─' * 80}")
            print(f"{query_type}: \"{query_text}\"")
            print(f"{'─' * 80}")

            query_embedding = embedder.encode([query_text])[0]
            results = client.search(
                table_name="chinese_articles_openai",
                vector=query_embedding,
                k=3,
            )

            for i, result in enumerate(results, 1):
                doc = records_map.get(result["id"])
                if doc and "payload_json" in doc:
                    payload = json.loads(doc["payload_json"])
                    print(f"{i}. {payload['title']} (相似度: {result['score']:.4f})")

            # Special note for NLP query
            if "NLP" in query_text:
                print("\n💡 OpenAI改进提示:")
                first_result = records_map.get(results[0]["id"])
                if first_result and "payload_json" in first_result:
                    first_payload = json.loads(first_result["payload_json"])
                    if "自然语言处理" in first_payload.get("title", ""):
                        print("   ✅ OpenAI成功理解了'NLP'='自然语言处理'的关系！")
                    else:
                        print("   ⚠️  排序可能还需要优化（但已比之前的模型好）")

        # ================================================================
        # Example 5: Category Filtering
        # ================================================================
        print("\n" + "=" * 80)
        print("示例 5: 分类过滤搜索")
        print("=" * 80)

        category_query = "深度学习和神经网络"
        query_embedding = embedder.encode([category_query])[0]

        print(f"\n查询: \"{category_query}\"")
        print("过滤条件: 分类 = 'AI'")

        results = client.search(
            table_name="chinese_articles_openai",
            vector=query_embedding,
            k=5,
            filter={"category": "AI"},
        )

        print_chinese_results(category_query, results, records_map, "分类过滤结果")

        # ================================================================
        # Summary
        # ================================================================
        print("\n" + "=" * 80)
        print("OpenAI Embeddings 优势总结")
        print("=" * 80)
        print("""
OpenAI text-embedding-3-small 相比开源模型的优势:

1. ✅ **更好的缩写理解**:
   - 能理解 NLP ↔ 自然语言处理
   - 能理解 AI ↔ 人工智能
   - 能理解 ML ↔ 机器学习

2. ✅ **更好的多语言支持**:
   - 中英混合查询效果更好
   - 技术术语理解更准确

3. ✅ **更高的语义理解能力**:
   - 同义词识别更准确
   - 上下文理解更深入

4. ⚠️  **需要考虑的因素**:
   - 需要API密钥和网络连接
   - 有API调用成本（但很低廉）
   - 稍有延迟（但可以批量处理优化）

5. 💰 **成本估算**:
   - text-embedding-3-small: $0.02 / 1M tokens
   - 10篇文章（~500 tokens）: $0.00001
   - 1000次查询: $0.02
   - 非常经济实惠！

推荐使用场景:
  ✅ 生产环境（需要高质量结果）
  ✅ 中英混合查询
  ✅ 技术文档搜索（有大量缩写）
  ❌ 离线环境（无网络）
  ❌ 实时性要求极高（<10ms）的场景
        """)

    finally:
        # Cleanup
        print("\n清理...")
        try:
            client.delete_table("chinese_articles_openai")
            print("✅ 删除表 'chinese_articles_openai'")
        except Exception as e:
            print(f"⚠️  清理失败: {e}")

        client.close()
        print("\n✅ 完成!")


if __name__ == "__main__":
    main()
