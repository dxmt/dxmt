#!/usr/bin/env python3
"""
端到端全文搜索测试案例
End-to-End Full-Text Search Test Case

这个测试演示了完整的工作流程：
1. 创建带全文搜索的表
2. 插入带文本内容的记录
3. 纯文本搜索
4. 混合搜索（向量 + 文本）
5. 带metadata过滤的搜索

Prerequisites:
- DXMT VectorDB server running on localhost:9999
- Compiled with --features text-bm25
- (Optional) Python packages: sentence-transformers for embeddings
"""

import requests
import json
import time
from typing import List, Dict, Any

BASE_URL = "http://localhost:9999/api/v1"


class Colors:
    """终端颜色"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_step(step_num: int, description: str):
    """打印测试步骤"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}步骤 {step_num}: {description}{Colors.END}")
    print("=" * 70)


def print_success(message: str):
    """打印成功消息"""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_info(message: str):
    """打印信息"""
    print(f"{Colors.YELLOW}ℹ {message}{Colors.END}")


def print_error(message: str):
    """打印错误"""
    print(f"{Colors.RED}✗ {message}{Colors.END}")


def get_simple_embeddings(texts: List[str], dimension: int = 128) -> List[List[float]]:
    """
    生成简单的embedding向量

    注意：在生产环境中，你应该使用真实的embedding模型，如：
    - sentence-transformers (SBERT)
    - OpenAI embeddings
    - Cohere embeddings

    这里我们使用简化的方法来演示
    """
    print_info(f"生成 {len(texts)} 个文本的embedding向量 (维度: {dimension})")

    # 简化的embedding：基于文本长度和内容生成伪向量
    embeddings = []
    for i, text in enumerate(texts):
        # 基于文本特征生成一个伪向量
        base_value = (len(text) % 10) / 10.0
        vector = [base_value + (i * 0.01) + (j * 0.001) for j in range(dimension)]
        embeddings.append(vector)

    return embeddings


# 如果你安装了 sentence-transformers，可以使用真实的embedding
def get_real_embeddings(texts: List[str]) -> List[List[float]]:
    """
    使用 sentence-transformers 生成真实的embedding

    安装: pip install sentence-transformers
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts).tolist()
        print_success(f"使用 sentence-transformers 生成了 {len(embeddings)} 个embedding")
        return embeddings
    except ImportError:
        print_info("sentence-transformers 未安装，使用简化的embedding方法")
        return get_simple_embeddings(texts, dimension=384)  # all-MiniLM-L6-v2 的维度


def step1_create_table():
    """步骤1: 创建带全文搜索的表"""
    print_step(1, "创建带全文搜索功能的表")

    table_config = {
        "name": "tech_articles",
        "dimension": 128,  # 向量维度
        "distance_metric": "cosine",
        "description": "技术文章数据库",
        "enable_full_text_search": True,
        "text_fields": [
            {
                "name": "title",
                "source": {"metadata_field": "title"},
                "analyzer": "standard",
                "boost": 2.0  # 标题权重是内容的2倍
            },
            {
                "name": "content",
                "source": {"metadata_field": "content"},
                "analyzer": "standard",
                "boost": 1.0
            },
            {
                "name": "tags",
                "source": {"metadata_field": "tags"},
                "analyzer": "standard",
                "boost": 1.5
            }
        ]
    }

    print(f"表配置:\n{json.dumps(table_config, indent=2, ensure_ascii=False)}")

    response = requests.post(f"{BASE_URL}/tables", json=table_config)

    if response.status_code == 200:
        print_success("表创建成功！")
        result = response.json()
        print(f"返回结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        return True
    else:
        print_error(f"表创建失败: {response.status_code}")
        print(response.text)
        return False


def step2_insert_documents():
    """步骤2: 插入带文本内容的记录"""
    print_step(2, "插入技术文章记录")

    # 准备文章数据
    articles = [
        {
            "id": "article_1",
            "title": "深度学习入门指南",
            "content": "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的层次化表示。本文介绍了深度学习的基础概念、常用框架和实践方法。",
            "tags": "深度学习 机器学习 神经网络 AI",
            "category": "教程",
            "difficulty": "入门",
            "views": 1200
        },
        {
            "id": "article_2",
            "title": "自然语言处理最新进展",
            "content": "自然语言处理（NLP）是人工智能的重要分支，近年来在大语言模型的推动下取得了巨大进展。Transformer架构彻底改变了NLP领域。",
            "tags": "NLP 自然语言处理 Transformer LLM",
            "category": "研究",
            "difficulty": "中级",
            "views": 2500
        },
        {
            "id": "article_3",
            "title": "计算机视觉应用实战",
            "content": "计算机视觉使用深度学习技术让计算机理解图像和视频。本文介绍了目标检测、图像分割、人脸识别等常见应用。",
            "tags": "计算机视觉 CV 图像识别 深度学习",
            "category": "实战",
            "difficulty": "中级",
            "views": 1800
        },
        {
            "id": "article_4",
            "title": "强化学习原理与实践",
            "content": "强化学习是机器学习的另一个重要分支，通过与环境交互来学习最优策略。本文从Q-learning到深度强化学习进行系统讲解。",
            "tags": "强化学习 RL Q-learning DQN",
            "category": "教程",
            "difficulty": "高级",
            "views": 980
        },
        {
            "id": "article_5",
            "title": "向量数据库技术详解",
            "content": "向量数据库是专门为存储和检索高维向量而设计的数据库系统。随着大语言模型的兴起，向量数据库在语义搜索、RAG等场景中发挥重要作用。",
            "tags": "向量数据库 VectorDB 语义搜索 Embedding",
            "category": "技术",
            "difficulty": "中级",
            "views": 3200
        }
    ]

    # 生成embedding向量
    texts_for_embedding = [f"{a['title']} {a['content']}" for a in articles]
    embeddings = get_simple_embeddings(texts_for_embedding, dimension=128)

    # 构建插入请求
    records = []
    for i, article in enumerate(articles):
        record = {
            "id": article["id"],
            "vector": embeddings[i],
            "metadata": {
                "title": article["title"],
                "content": article["content"],
                "tags": article["tags"],
                "category": article["category"],
                "difficulty": article["difficulty"],
                "views": article["views"]
            }
        }
        records.append(record)

    print(f"准备插入 {len(records)} 条记录...")

    response = requests.post(
        f"{BASE_URL}/tables/tech_articles/records",
        json={"records": records}
    )

    if response.status_code == 200:
        print_success(f"成功插入 {len(records)} 条记录！")
        print_info("文本内容已自动索引到全文搜索引擎")

        # 等待索引完成
        print_info("等待索引完成...")
        time.sleep(1)
        return True
    else:
        print_error(f"插入失败: {response.status_code}")
        print(response.text)
        return False


def step3_full_text_search():
    """步骤3: 纯文本搜索"""
    print_step(3, "纯文本搜索")

    # 测试查询
    queries = [
        {
            "description": "搜索'深度学习'相关文章",
            "query": "深度学习",
            "limit": 3
        },
        {
            "description": "搜索'自然语言处理'相关文章（带高亮）",
            "query": "自然语言处理",
            "limit": 3,
            "highlight": True
        }
    ]

    for i, q in enumerate(queries, 1):
        print(f"\n{Colors.YELLOW}查询 {i}: {q['description']}{Colors.END}")
        print(f"查询文本: \"{q['query']}\"")

        response = requests.post(
            f"{BASE_URL}/tables/tech_articles/search/text",
            json=q
        )

        if response.status_code == 200:
            result = response.json()["data"]
            print_success(f"找到 {result['total']} 个结果 (耗时: {result['took_ms']}ms)")

            for idx, hit in enumerate(result["hits"], 1):
                print(f"\n  {idx}. 文章ID: {hit['id']} (相关度分数: {hit['score']:.3f})")
                if hit.get("payload"):
                    print(f"     标题: {hit['payload'].get('title', 'N/A')}")
                    print(f"     分类: {hit['payload'].get('category', 'N/A')}")

                if hit.get("highlights"):
                    print(f"     高亮片段:")
                    for highlight in hit["highlights"]:
                        print(f"       {highlight}")
        else:
            print_error(f"搜索失败: {response.status_code}")
            print(response.text)


def step4_hybrid_search():
    """步骤4: 混合搜索（向量 + 文本）"""
    print_step(4, "混合搜索（向量相似度 + 文本相关性）")

    # 生成查询向量（模拟用户查询的embedding）
    query_text = "机器学习和神经网络"
    query_vector = get_simple_embeddings([query_text], dimension=128)[0]

    # 不同权重的混合搜索
    test_cases = [
        {
            "description": "均衡模式 (50% 向量 + 50% 文本)",
            "vector": query_vector,
            "text": "机器学习",
            "vector_weight": 0.5,
            "limit": 3
        },
        {
            "description": "向量优先模式 (80% 向量 + 20% 文本)",
            "vector": query_vector,
            "text": "机器学习",
            "vector_weight": 0.8,
            "limit": 3
        },
        {
            "description": "文本优先模式 (30% 向量 + 70% 文本)",
            "vector": query_vector,
            "text": "机器学习",
            "vector_weight": 0.3,
            "limit": 3
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{Colors.YELLOW}混合搜索 {i}: {case['description']}{Colors.END}")
        print(f"文本查询: \"{case.get('text', 'N/A')}\"")
        print(f"向量权重: {case['vector_weight']} (文本权重: {1 - case['vector_weight']})")

        response = requests.post(
            f"{BASE_URL}/tables/tech_articles/search/hybrid",
            json={
                "vector": case["vector"],
                "text": case.get("text"),
                "vector_weight": case["vector_weight"],
                "limit": case["limit"],
                "with_payload": True
            }
        )

        if response.status_code == 200:
            result = response.json()["data"]
            print_success(f"找到 {result['total']} 个结果 (耗时: {result['tookms']}ms)")

            for idx, hit in enumerate(result["hits"], 1):
                print(f"\n  {idx}. {hit['id']} (综合分数: {hit['score']:.3f})")
                if hit.get("payload"):
                    print(f"     {hit['payload'].get('title', 'N/A')}")
        else:
            print_error(f"混合搜索失败: {response.status_code}")
            print(response.text)


def step5_filtered_search():
    """步骤5: 带metadata过滤的搜索"""
    print_step(5, "带Metadata过滤的搜索")

    # 不同的过滤条件
    filter_cases = [
        {
            "description": "只搜索'教程'分类的文章",
            "query": "学习",
            "filter": {
                "must": [
                    {
                        "key": "category",
                        "operation": {
                            "type": "Equal",
                            "value": "教程"
                        }
                    }
                ]
            }
        },
        {
            "description": "搜索浏览量大于1000的中级文章",
            "query": "深度学习",
            "filter": {
                "must": [
                    {
                        "key": "views",
                        "operation": {
                            "type": "GreaterThan",
                            "value": 1000
                        }
                    },
                    {
                        "key": "difficulty",
                        "operation": {
                            "type": "Equal",
                            "value": "中级"
                        }
                    }
                ]
            }
        },
        {
            "description": "搜索非入门级别的技术文章",
            "query": "技术",
            "filter": {
                "must_not": [
                    {
                        "key": "difficulty",
                        "operation": {
                            "type": "Equal",
                            "value": "入门"
                        }
                    }
                ]
            }
        }
    ]

    for i, case in enumerate(filter_cases, 1):
        print(f"\n{Colors.YELLOW}过滤搜索 {i}: {case['description']}{Colors.END}")
        print(f"查询: \"{case['query']}\"")
        print(f"过滤条件: {json.dumps(case['filter'], ensure_ascii=False, indent=2)}")

        response = requests.post(
            f"{BASE_URL}/tables/tech_articles/search/text",
            json={
                "query": case["query"],
                "filter": case["filter"],
                "limit": 10
            }
        )

        if response.status_code == 200:
            result = response.json()["data"]
            print_success(f"找到 {result['total']} 个符合条件的结果")

            for idx, hit in enumerate(result["hits"], 1):
                payload = hit.get("payload", {})
                print(f"\n  {idx}. {hit['id']}")
                print(f"     标题: {payload.get('title', 'N/A')}")
                print(f"     分类: {payload.get('category', 'N/A')}")
                print(f"     难度: {payload.get('difficulty', 'N/A')}")
                print(f"     浏览: {payload.get('views', 'N/A')}")
        else:
            print_error(f"过滤搜索失败: {response.status_code}")
            print(response.text)


def step6_cleanup():
    """步骤6: 清理测试数据"""
    print_step(6, "清理测试数据")

    response = requests.delete(f"{BASE_URL}/tables/tech_articles")

    if response.status_code == 200:
        print_success("测试表已删除")
        return True
    else:
        print_error(f"删除失败: {response.status_code}")
        print(response.text)
        return False


def explain_embedding_workflow():
    """解释Embedding工作流程"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
    print("关于Embedding的重要说明")
    print(f"{'='*70}{Colors.END}")

    print("""
在DXMT VectorDB中，embedding的生成和使用流程如下：

1. 📝 Embedding生成（客户端负责）:
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   - Embedding是在插入记录之前生成的
   - 由客户端应用程序负责生成embedding向量
   - 可以使用各种embedding模型：
     • OpenAI Embeddings (text-embedding-ada-002)
     • Sentence Transformers (SBERT)
     • Cohere Embeddings
     • 自定义模型

   示例代码:
   ```python
   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer('all-MiniLM-L6-v2')
   text = "机器学习是人工智能的一个分支"
   embedding = model.encode(text)  # 生成向量
   ```

2. 💾 插入到VectorDB:
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   - 将生成的embedding向量和原始文本一起插入
   - VectorDB执行两个索引操作：
     a) 向量索引 - 用于向量相似度搜索
     b) 文本索引 - 用于全文搜索（如果启用）

   ```python
   record = {
       "id": "doc1",
       "vector": embedding.tolist(),  # 已生成的向量
       "metadata": {
           "text": "机器学习是...",  # 原始文本
           "title": "标题"
       }
   }
   ```

3. 🔍 搜索时的Embedding:
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   a) 向量搜索:
      - 查询文本 → Embedding模型 → 查询向量
      - 使用查询向量搜索相似向量

   b) 文本搜索:
      - 查询文本 → 直接搜索（BM25）
      - 不需要embedding

   c) 混合搜索:
      - 同时执行向量搜索和文本搜索
      - 合并结果并按权重排序

4. 🎯 自动Embedding（未来功能）:
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   - DXMT支持Multimodal功能（需要ONNX Runtime）
   - 可以自动生成embedding（待ONNX 2.0稳定版发布）
   - 路径: src/multimodal/

总结:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 当前: Embedding由客户端生成 ✓
• 优点: 灵活，可以使用任何embedding模型
• 未来: 服务端自动生成embedding（multimodal功能）
""")


def main():
    """主函数"""
    print(f"{Colors.BOLD}{Colors.GREEN}")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "DXMT VectorDB 全文搜索端到端测试" + " " * 16 + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"{Colors.END}\n")

    print("测试包括:")
    print("  1. 创建带全文搜索的表")
    print("  2. 插入带文本内容的记录")
    print("  3. 纯文本搜索")
    print("  4. 混合搜索（向量 + 文本）")
    print("  5. 带Metadata过滤的搜索")
    print("  6. 清理测试数据")

    try:
        # 执行测试步骤
        if not step1_create_table():
            return

        if not step2_insert_documents():
            return

        step3_full_text_search()
        step4_hybrid_search()
        step5_filtered_search()

        # 解释embedding工作流程
        explain_embedding_workflow()

        # 清理
        print("\n" + "─" * 70)
        cleanup_choice = input(f"\n{Colors.YELLOW}是否删除测试表? (y/n): {Colors.END}")
        if cleanup_choice.lower() == 'y':
            step6_cleanup()
        else:
            print_info("测试表保留，表名: tech_articles")

        print(f"\n{Colors.BOLD}{Colors.GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("所有测试成功完成！ ✓")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.END}\n")

    except requests.exceptions.ConnectionError:
        print_error("\n无法连接到DXMT VectorDB服务器")
        print_info("请确保服务器正在运行: cargo run -- serve")
    except Exception as e:
        print_error(f"\n测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
