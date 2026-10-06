"""Chinese text search example for DXMT Python SDK.

This example demonstrates:
1. Full-text search with Chinese text using Jieba tokenization
2. Hybrid search combining Chinese text and vector embeddings
3. Multilingual embeddings for Chinese semantic search
4. Comparing text-only vs semantic vs hybrid search

Prerequisites:
    pip install sentence-transformers

Use case:
    - Chinese document search
    - Chinese Q&A systems
    - Chinese content recommendation
    - Bilingual search systems
"""

import json
from typing import List, Dict, Any

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

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
    """Run Chinese text search examples."""
    print("=" * 80)
    print("DXMT 中文文本搜索示例")
    print("=" * 80)

    # Check dependencies
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        skip("未安装 sentence-transformers: pip install sentence-transformers")

    # Load multilingual embedding model
    print("\n加载多语言嵌入模型 (paraphrase-multilingual-MiniLM-L12-v2)...")
    print("   首次运行可能需要下载模型...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"✅ 模型加载完成 (维度: {embedding_dim})")

    # Create client
    print("\n连接 DXMT 服务器...")
    client = Client(**client_kwargs())

    try:
        # Create table with full-text search enabled
        print("\n创建中文文章表...")
        try:
            client.delete_table("chinese_articles")
        except Exception:
            pass

        client.create_table(
            name="chinese_articles",
            dimension=embedding_dim,
            distance_metric="cosine",
            index_type="hnsw",
            enable_full_text_search=True,  # Enable full-text search
        )
        print("✅ 表创建成功")

        # Load Chinese articles
        print("\n加载中文文章...")
        articles = create_chinese_articles()

        # Generate embeddings
        texts = [article["text"] for article in articles]
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        print(f"✅ 生成 {len(embeddings)} 个嵌入向量")

        # Prepare records
        records = []
        for article, embedding in zip(articles, embeddings):
            records.append({
                "id": article["id"],
                "vector": embedding.tolist(),
                "payload": {
                    "content": article["text"],
                    "title": article["metadata"]["title"],
                    "category": article["metadata"]["category"],
                    "author": article["metadata"]["author"],
                },
            })

        # Insert records
        print("\n插入文章到 DXMT...")
        client.insert_records(table_name="chinese_articles", records=records)
        print(f"✅ 插入 {len(records)} 篇文章")

        # Get all records for display
        all_records = client.get_records(table_name="chinese_articles", ids=[a["id"] for a in articles])
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
            table_name="chinese_articles",
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
        print("示例 2: 语义搜索（仅向量相似度）")
        print("=" * 80)

        semantic_query = "什么是向量存储和查询？"
        print(f"\n使用多语言模型进行语义搜索...")

        query_embedding = model.encode(semantic_query, convert_to_numpy=True)

        results = client.search(
            table_name="chinese_articles",
            vector=query_embedding.tolist(),
            k=3,
        )

        print_chinese_results(semantic_query, results, records_map, "语义搜索结果")

        # ================================================================
        # Example 3: Hybrid Search (Text + Vector)
        # ================================================================
        print("\n" + "=" * 80)
        print("示例 3: 混合搜索（文本 + 向量）")
        print("=" * 80)

        hybrid_query_text = "机器学习"
        hybrid_query_embedding = model.encode(hybrid_query_text, convert_to_numpy=True)

        print(f"\n混合搜索结合全文检索和语义相似度...")

        # Try different alpha values
        alphas = [0.3, 0.5, 0.7]

        for alpha in alphas:
            print(f"\n{'─' * 80}")
            print(f"Alpha = {alpha} ({'偏重文本' if alpha < 0.5 else '偏重向量' if alpha > 0.5 else '平衡'})")
            print(f"{'─' * 80}")

            results = client.hybrid_search(
                table_name="chinese_articles",
                vector=hybrid_query_embedding.tolist(),
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
        # Example 4: Different Query Styles
        # ================================================================
        print("\n" + "=" * 80)
        print("示例 4: 不同查询方式对比")
        print("=" * 80)

        test_queries = [
            ("图算法", "关键词查询"),
            ("如何提高搜索速度？", "自然问题"),
            ("NLP技术应用", "技术缩写"),
        ]

        for query_text, query_type in test_queries:
            print(f"\n{'─' * 80}")
            print(f"{query_type}: \"{query_text}\"")
            print(f"{'─' * 80}")

            query_embedding = model.encode(query_text, convert_to_numpy=True)
            results = client.search(
                table_name="chinese_articles",
                vector=query_embedding.tolist(),
                k=3,
            )

            for i, result in enumerate(results, 1):
                doc = records_map.get(result["id"])
                if doc and "payload_json" in doc:
                    payload = json.loads(doc["payload_json"])
                    print(f"{i}. {payload['title']} (相似度: {result['score']:.4f})")

        # ================================================================
        # Example 5: Category Filtering
        # ================================================================
        print("\n" + "=" * 80)
        print("示例 5: 分类过滤搜索")
        print("=" * 80)

        category_query = "深度学习和神经网络"
        query_embedding = model.encode(category_query, convert_to_numpy=True)

        print(f"\n查询: \"{category_query}\"")
        print("过滤条件: 分类 = 'AI'")

        results = client.search(
            table_name="chinese_articles",
            vector=query_embedding.tolist(),
            k=5,
            filter={"category": "AI"},
        )

        print_chinese_results(category_query, results, records_map, "分类过滤结果")

        # ================================================================
        # Summary
        # ================================================================
        print("\n" + "=" * 80)
        print("中文搜索最佳实践")
        print("=" * 80)
        print("""
搜索方式对比:

1. 全文检索 (Full-Text Search):
   ✓ 精确匹配关键词
   ✓ 使用 Jieba 分词
   ✓ 适合已知关键词的查询
   ✗ 无法理解语义

2. 语义搜索 (Semantic Search):
   ✓ 理解查询意图
   ✓ 支持同义词和释义
   ✓ 适合自然语言问题
   ✗ 可能错过精确关键词

3. 混合搜索 (Hybrid Search):
   ✓ 结合两者优点
   ✓ 可调节权重平衡
   ✓ 更好的搜索效果
   ✓ 推荐用于生产环境

中文特殊处理:
  - 使用多语言嵌入模型 (paraphrase-multilingual-*)
  - 全文检索自动使用 Jieba 分词
  - 考虑简繁体转换
  - 处理中文标点符号

模型推荐:
  - paraphrase-multilingual-MiniLM-L12-v2: 快速，384维
  - paraphrase-multilingual-mpnet-base-v2: 高质量，768维
  - distiluse-base-multilingual-cased-v2: 平衡选择

应用场景:
  - 中文文档搜索
  - 知识库问答
  - 客服系统
  - 内容推荐
  - 法律/医疗文档检索

性能优化:
  - 缓存常用查询的嵌入
  - 使用元数据过滤减少搜索空间
  - 批量处理多个查询
  - 定期更新索引
        """)

    finally:
        # Cleanup
        print("\n清理...")
        try:
            client.delete_table("chinese_articles")
            print("✅ 删除表 'chinese_articles'")
        except Exception as e:
            print(f"⚠️  清理失败: {e}")

        client.close()
        print("\n✅ 完成!")


if __name__ == "__main__":
    main()
