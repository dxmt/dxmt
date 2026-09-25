"""Chinese text search example for langchain-dxmt.

This example demonstrates DXMT's support for Chinese text search
using Jieba tokenizer for full-text indexing.

Prerequisites:
- DXMT VectorDB server running on localhost:9999
- OpenAI API key set in environment
- DXMT server configured with Chinese text support (Jieba tokenizer)

Run:
    python examples/chinese_text_search.py
"""

from langchain_core.documents import Document
from embedding import default_embeddings, embed_dimension

from langchain_dxmt import DXMTVectorDB


def main() -> None:
    """Run Chinese text search example."""
    print("=== LangChain DXMT 中文文本搜索示例 ===\n")

    # 1. Initialize embeddings
    print("1. 初始化 OpenAI embeddings...")
    embeddings = default_embeddings()

    # 2. Create vector store with Chinese full-text search
    print("2. 创建支持中文全文检索的向量存储...")
    vector_store = DXMTVectorDB(
        table_name="chinese_docs",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
        host="localhost",
        port=9999,
        index_type="hnsw",
        distance_metric="cosine",
        enable_full_text_search=True,  # 启用全文检索
    )
    print(f"   ✓ 向量存储已创建: {vector_store._table_name}\n")

    # 3. Add Chinese documents
    print("3. 添加中文文档...")
    documents = [
        Document(
            page_content="人工智能是计算机科学的一个分支，致力于创建能够执行需要人类智能的任务的系统。",
            metadata={
                "category": "人工智能",
                "source": "ai_introduction",
                "language": "zh",
            },
        ),
        Document(
            page_content="机器学习是人工智能的核心技术之一，通过算法让计算机从数据中学习规律。",
            metadata={
                "category": "机器学习",
                "source": "ml_basics",
                "language": "zh",
            },
        ),
        Document(
            page_content="深度学习使用多层神经网络来学习数据的复杂表示，在图像识别、自然语言处理等领域取得了突破性进展。",
            metadata={
                "category": "深度学习",
                "source": "dl_overview",
                "language": "zh",
            },
        ),
        Document(
            page_content="自然语言处理（NLP）是人工智能的重要分支，专注于让计算机理解和生成人类语言。",
            metadata={
                "category": "自然语言处理",
                "source": "nlp_intro",
                "language": "zh",
            },
        ),
        Document(
            page_content="向量数据库是专门用于存储和查询高维向量的数据库系统，在语义搜索中扮演重要角色。",
            metadata={
                "category": "数据库",
                "source": "vectordb_basics",
                "language": "zh",
            },
        ),
        Document(
            page_content="DXMT 是一个轻量级的向量数据库，使用 Rust 编写，支持多种索引类型和距离度量。",
            metadata={
                "category": "数据库",
                "source": "dxmt_intro",
                "language": "zh",
            },
        ),
        Document(
            page_content="语义搜索通过理解查询的语义含义来检索相关文档，而不仅仅依赖关键词匹配。",
            metadata={
                "category": "搜索技术",
                "source": "semantic_search",
                "language": "zh",
            },
        ),
        Document(
            page_content="LangChain 是一个用于开发大语言模型应用的框架，提供了丰富的工具和抽象。",
            metadata={
                "category": "开发框架",
                "source": "langchain_intro",
                "language": "zh",
            },
        ),
    ]

    ids = vector_store.add_documents(documents)
    print(f"   ✓ 已添加 {len(ids)} 篇中文文档\n")

    # 4. Pure vector similarity search
    print("4. 纯向量相似度搜索:")
    print("   " + "=" * 60)
    query = "什么是人工智能？"
    results = vector_store.similarity_search(query, k=3)

    print(f"   查询: '{query}'")
    print(f"   结果:\n")
    for i, doc in enumerate(results, 1):
        print(f"   [{i}] {doc.page_content}")
        print(f"       类别: {doc.metadata.get('category', 'N/A')}\n")

    # 5. Hybrid search - Balanced
    print("5. 混合搜索 (50% 向量, 50% 文本):")
    print("   " + "=" * 60)
    query = "机器学习算法"
    results = vector_store.hybrid_search(
        query,
        k=3,
        vector_weight=0.5,  # 50% 向量, 50% 文本
    )

    print(f"   查询: '{query}'")
    print(f"   结果:\n")
    for i, doc in enumerate(results, 1):
        print(f"   [{i}] {doc.page_content}")
        print(f"       类别: {doc.metadata.get('category', 'N/A')}\n")

    # 6. Text-heavy hybrid search
    print("6. 文本为主的混合搜索 (20% 向量, 80% 文本):")
    print("   " + "=" * 60)
    query = "向量数据库 DXMT"
    results = vector_store.hybrid_search(
        query,
        k=3,
        vector_weight=0.2,  # 更依赖关键词匹配
    )

    print(f"   查询: '{query}'")
    print(f"   结果:\n")
    for i, doc in enumerate(results, 1):
        print(f"   [{i}] {doc.page_content}")
        print(f"       来源: {doc.metadata.get('source', 'N/A')}\n")

    # 7. Vector-heavy hybrid search
    print("7. 向量为主的混合搜索 (80% 向量, 20% 文本):")
    print("   " + "=" * 60)
    query = "神经网络深度学习"
    results = vector_store.hybrid_search(
        query,
        k=3,
        vector_weight=0.8,  # 更依赖语义相似度
    )

    print(f"   查询: '{query}'")
    print(f"   结果:\n")
    for i, doc in enumerate(results, 1):
        print(f"   [{i}] {doc.page_content}")
        print(f"       类别: {doc.metadata.get('category', 'N/A')}\n")

    # 8. Search with metadata filter
    print("8. 带元数据过滤的搜索:")
    print("   " + "=" * 60)
    query = "人工智能技术"
    results = vector_store.hybrid_search(
        query,
        k=5,
        vector_weight=0.6,
        filter={"category": "人工智能"},
    )

    print(f"   查询: '{query}' (过滤条件: category='人工智能')")
    print(f"   结果:\n")
    for i, doc in enumerate(results, 1):
        print(f"   [{i}] {doc.page_content}")
        print(f"       来源: {doc.metadata.get('source', 'N/A')}\n")

    # 9. Semantic vs keyword comparison
    print("9. 语义搜索 vs 关键词搜索对比:")
    print("   " + "=" * 60)

    # Query with no exact keyword match
    query = "计算机如何理解人说的话"

    print(f"\n   查询: '{query}'")
    print("   (注意: 查询中没有文档中的精确关键词)\n")

    # Pure vector search (semantic)
    print("   纯语义搜索结果:")
    results = vector_store.similarity_search(query, k=2)
    for i, doc in enumerate(results, 1):
        print(f"   [{i}] {doc.page_content[:50]}...")
        print(f"       类别: {doc.metadata.get('category', 'N/A')}\n")

    # Hybrid search
    print("   混合搜索结果 (70% 语义):")
    results = vector_store.hybrid_search(query, k=2, vector_weight=0.7)
    for i, doc in enumerate(results, 1):
        print(f"   [{i}] {doc.page_content[:50]}...")
        print(f"       类别: {doc.metadata.get('category', 'N/A')}\n")

    # 10. Complex query
    print("10. 复杂查询示例:")
    print("    " + "=" * 60)
    query = "如何使用深度学习模型进行中文自然语言处理？"
    results = vector_store.similarity_search_with_score(query, k=3)

    print(f"    查询: '{query}'")
    print(f"    结果:\n")
    for i, (doc, score) in enumerate(results, 1):
        print(f"    [{i}] 相似度: {score:.4f}")
        print(f"        {doc.page_content}")
        print(f"        类别: {doc.metadata.get('category', 'N/A')}\n")

    # 11. Cleanup
    print("11. 清理数据...")
    if ids:
        vector_store.delete(ids=ids)
        print("    ✓ 已删除文档\n")

    print("=== 示例运行完成! ===")
    print("\n💡 中文文本搜索的关键点:")
    print("   - DXMT 使用 Jieba 分词器进行中文文本索引")
    print("   - 混合搜索结合了语义理解和关键词匹配")
    print("   - 调整 vector_weight 可以控制语义vs关键词的权重")
    print("   - 语义搜索能理解同义词和相关概念")
    print("   - 对于专业术语，适当增加文本匹配权重\n")


def demonstrate_jieba_tokenization() -> None:
    """Demonstrate Jieba tokenization examples."""
    print("\n=== Jieba 分词示例 ===\n")

    examples = [
        "机器学习算法",
        "自然语言处理技术",
        "深度神经网络模型",
        "向量数据库系统",
        "语义相似度计算",
    ]

    print("中文短语分词示例:")
    print("(实际分词由 DXMT 服务器的 Jieba tokenizer 执行)\n")

    for phrase in examples:
        print(f"原文: {phrase}")
        # Note: Actual tokenization happens on DXMT server
        print(f"说明: 该短语会被 Jieba 分词器拆分为有意义的词语\n")


if __name__ == "__main__":
    main()

    # Uncomment to see Jieba tokenization explanation
    # demonstrate_jieba_tokenization()
