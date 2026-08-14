#!/usr/bin/env python3
"""
查询 PDF 文档 - 使用已导入的向量数据库

用法:
    python3 query_pdf_docs.py "your question here"
    python3 query_pdf_docs.py  # 交互模式
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "dxmt-python-sdk"))

from dxmt import Client
from sentence_transformers import SentenceTransformer


def query_documents(client, embedder, question, table_name="pdf_docs", k=3):
    """查询文档"""
    print(f"\n{'='*80}")
    print(f"问题: {question}")
    print(f"{'='*80}\n")

    # 生成查询 embedding
    query_embedding = embedder.encode(question).tolist()

    # 检索
    results = client.search(
        table_name=table_name,
        vector=query_embedding,
        k=k
    )

    if not results:
        print("⚠️  未找到相关文档")
        return

    print(f"✓ 检索到 {len(results)} 个相关文档\n")

    # 显示结果
    for i, result in enumerate(results, 1):
        metadata = result.get('metadata', {})
        text = metadata.get('text', '')
        chunk_id = metadata.get('chunk_id', 0)
        score = result.get('score', 0.0)

        print(f"[{i}] 文档块 #{chunk_id} (相似度: {score:.3f})")
        print(f"{'─'*80}")

        # 显示完整文本（如果较短）或前 500 字符
        if len(text) <= 500:
            print(text)
        else:
            print(text[:500] + "...")

        print()

    print(f"{'='*80}\n")


def main():
    print("=" * 80)
    print("PDF 文档查询工具")
    print("=" * 80)
    print()

    # 连接 VectorDB
    print("连接 DXMT 服务器...")
    try:
        client = Client(host="localhost", port=9999)
        print("✓ 连接成功\n")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("   请确保 VectorDB 服务器正在运行:")
        print("   cargo run --release -- serve --host 0.0.0.0 --port 9999")
        return

    # 检查表是否存在
    table_name = "pdf_docs"
    try:
        # 简单检查：尝试搜索
        client.search(table_name=table_name, vector=[0.0]*384, k=1)
        print(f"✓ 找到表 '{table_name}'\n")
    except Exception as e:
        print(f"❌ 表 '{table_name}' 不存在或为空")
        print("   请先运行: python3 examples/pdf_rag_simple.py")
        return

    # 加载 embedding 模型
    print("加载 embedding 模型...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ 模型加载完成\n")

    # 预定义的示例问题
    example_questions = [
        "What is this document about?",
        "your question here",
        "How to run simulations?",
        "What types of analysis are supported?",
        "How is the output analyzed?",
        "What are the main workflows?",
        "What are the far field calculations?",
        "How to set up waveguide ports?",
    ]

    # 检查命令行参数
    if len(sys.argv) > 1:
        # 直接查询模式
        question = " ".join(sys.argv[1:])
        query_documents(client, embedder, question, table_name)
        return

    # 交互模式
    print("=" * 80)
    print("交互式问答")
    print("=" * 80)
    print()
    print("示例问题（输入编号快速提问）：")
    for i, q in enumerate(example_questions, 1):
        print(f"  {i}. {q}")
    print()
    print("或者直接输入您的问题")
    print("输入 'quit' 或 'exit' 退出")
    print()

    while True:
        try:
            user_input = input("您的问题 >>> ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n再见！")
                break

            if not user_input:
                continue

            # 检查是否是问题编号
            if user_input.isdigit():
                q_num = int(user_input)
                if 1 <= q_num <= len(example_questions):
                    question = example_questions[q_num - 1]
                else:
                    print(f"⚠️  请输入 1-{len(example_questions)} 之间的编号")
                    continue
            else:
                question = user_input

            # 查询
            query_documents(client, embedder, question, table_name, k=5)

        except (KeyboardInterrupt, EOFError):
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}\n")


if __name__ == "__main__":
    main()
