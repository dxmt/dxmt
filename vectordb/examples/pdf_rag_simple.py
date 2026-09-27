#!/usr/bin/env python3
"""
简化版 PDF RAG Demo - 自动运行问答示例
无需 OpenAI API，仅演示检索功能
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "clients" / "dxmt-python-sdk"))

from dxmt import Client
from sentence_transformers import SentenceTransformer
import PyPDF2
from typing import List, Dict, Any


def extract_pdf_text(pdf_path: str) -> str:
    """提取 PDF 文本"""
    print(f"正在读取 PDF: {pdf_path}")
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        print(f"PDF 总页数: {num_pages}")

        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
            if (page_num + 1) % 10 == 0:
                print(f"  已处理 {page_num + 1}/{num_pages} 页...")

    print(f"✓ PDF 文本提取完成，共 {len(text)} 字符")
    return text


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """分割文本"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if end < len(text):
            last_sentence = max(
                chunk.rfind('.'),
                chunk.rfind('?'),
                chunk.rfind('!'),
                chunk.rfind('\n')
            )
            if last_sentence > chunk_size * 0.5:
                chunk = chunk[:last_sentence + 1]
                end = start + last_sentence + 1

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap

    print(f"✓ 文本分割完成，共 {len(chunks)} 个块")
    return chunks


def main():
    print("=" * 80)
    print("PDF RAG 简化演示")
    print("=" * 80)
    print()

    # 连接服务器
    print("连接 DXMT 服务器...")
    client = Client(host="localhost", port=9999)
    print("✓ 连接成功\n")

    # 设置表
    table_name = "pdf_docs"
    print(f"设置表 '{table_name}'...")
    try:
        client.delete_table(table_name)
    except:
        pass

    client.create_table(name=table_name, dimension=384, distance_metric="cosine")
    print("✓ 表创建成功\n")

    # 加载 embedding 模型
    print("加载 embedding 模型...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ 模型加载完成\n")

    # 处理 PDF
    print("=" * 80)
    print("步骤 1: 处理 PDF 文件")
    print("=" * 80)
    print()

    # 自备任意 PDF，通过 DEMO_PDF_PATH 指定
    pdf_path = os.environ.get("DEMO_PDF_PATH", "pdf/sample.pdf")
    if not os.path.exists(pdf_path):
        print(f"❌ PDF 文件不存在: {pdf_path}")
        print("   本仓库不附带演示文档，请自备一份 PDF 后指定路径：")
        print("   DEMO_PDF_PATH=/path/to/your.pdf python examples/pdf_rag_simple.py")
        return

    text = extract_pdf_text(pdf_path)
    chunks = chunk_text(text, chunk_size=800, overlap=100)

    # 生成 embeddings 并插入
    print("\n生成 embeddings 并插入数据库...")
    records = []
    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        record = {
            "id": str(i),
            "vector": embedding,
            "metadata": {
                "text": chunk,
                "source": os.path.basename(pdf_path),
                "chunk_id": i
            }
        }
        records.append(record)

        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{len(chunks)} 个块...")

    print("批量插入向量数据库...")
    client.insert_records(table_name, records)
    print(f"✓ 成功插入 {len(records)} 个文档块\n")

    # 问答演示
    print("=" * 80)
    print("步骤 2: RAG 检索演示")
    print("=" * 80)
    print()

    questions = [
        "What is this document about?",
        "What are the key concepts covered?",
        "How to run simulations?",
        "What types of analysis are supported?",
        "How is the output analyzed?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"问题 {i}: {question}")
        print(f"{'='*80}\n")

        # 生成查询 embedding
        query_embedding = embedder.encode(question).tolist()

        # 搜索
        results = client.search(
            table_name=table_name,
            vector=query_embedding,
            k=3
        )

        print(f"✓ 检索到 {len(results)} 个相关文档\n")

        # 显示结果
        for j, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            text = metadata.get('text', '')
            chunk_id = metadata.get('chunk_id', 0)
            score = result.get('score', 0.0)

            # 显示前 300 字符
            preview = text[:300] + "..." if len(text) > 300 else text

            print(f"[{j}] 文档块 #{chunk_id} (相似度: {score:.3f})")
            print(f"    {preview}")
            print()

        print("-" * 80)
        print("如果配置了 OpenAI API，可以基于以上文档生成详细答案")
        print("-" * 80)

    # 完成
    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)
    print()
    print(f"✓ 已将 PDF 文档导入 vectordb 表 '{table_name}'")
    print(f"✓ 共 {len(records)} 个文档块，可随时进行语义搜索")
    print(f"✓ 表保留在数据库中，可继续使用")
    print()
    print("下一步:")
    print("  1. 设置 OPENAI_API_KEY 环境变量以启用 LLM 生成")
    print("  2. 运行完整的交互式版本:")
    print("     python3 examples/pdf_rag_demo.py")
    print()


if __name__ == "__main__":
    main()
