#!/usr/bin/env python3
"""
PDF RAG Demo
处理 PDF 文件并实现 RAG 问答系统

Requirements:
    pip install dxmt sentence-transformers pypdf2 openai
"""

import os
import sys
from typing import List, Dict, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "clients" / "dxmt-python-sdk"))

from dxmt import Client
from sentence_transformers import SentenceTransformer

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False
    print("⚠️  PyPDF2 not installed. Install with: pip install pypdf2")

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️  OpenAI not installed. Install with: pip install openai")


# The provider's models come from the environment so one `.env` switch moves every example.
# These are *deployment* names and do not carry between providers -- the Azure resource this
# was written against serves `gpt-4.1-mini` and answers DeploymentNotFound for `gpt-4o-mini`.
# The defaults are what this file used before, so nothing changes for a plain OPENAI_API_KEY.
CHAT_MODEL = os.environ.get("DXMT_CHAT_MODEL", "gpt-3.5-turbo")
EMBED_MODEL = os.environ.get("DXMT_EMBED_MODEL", "text-embedding-3-small")
EMBED_DIMS = int(os.environ.get("DXMT_EMBED_DIMS", "1536"))


class PDFRAGSystem:
    """PDF RAG 系统 - 处理 PDF 并实现问答"""

    def __init__(
        self,
        client: Client,
        table_name: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        初始化 PDF RAG 系统

        Args:
            client: DXMT 客户端
            table_name: 表名
            embedding_model: 嵌入模型
            chunk_size: 文本块大小（字符数）
            chunk_overlap: 文本块重叠大小
        """
        self.client = client
        self.table_name = table_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化 embedding 模型
        print(f"加载 embedding 模型: {embedding_model}...")
        self.embedder = SentenceTransformer(embedding_model)
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        print(f"✓ 模型加载完成，维度: {self.dimension}")

        # 初始化 OpenAI
        if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            print("✓ OpenAI 客户端初始化成功")
        else:
            self.openai_client = None
            print("⚠️  OpenAI 未配置，仅支持文档检索")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        从 PDF 文件中提取文本

        Args:
            pdf_path: PDF 文件路径

        Returns:
            提取的文本内容
        """
        if not HAS_PYPDF2:
            raise ImportError("PyPDF2 is required. Install with: pip install pypdf2")

        print(f"正在读取 PDF: {pdf_path}")
        text = ""

        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            print(f"PDF 总页数: {num_pages}")

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += page_text + "\n"

                if (page_num + 1) % 10 == 0:
                    print(f"  已处理 {page_num + 1}/{num_pages} 页...")

        print(f"✓ PDF 文本提取完成，共 {len(text)} 字符")
        return text

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        将长文本分割成块

        Args:
            text: 原始文本

        Returns:
            文本块列表
        """
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            # 提取文本块
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # 尝试在句子边界处分割
            if end < len(text):
                # 寻找最后一个句号、问号或感叹号
                last_sentence_end = max(
                    chunk_text.rfind('.'),
                    chunk_text.rfind('?'),
                    chunk_text.rfind('!'),
                    chunk_text.rfind('\n')
                )
                if last_sentence_end > self.chunk_size * 0.5:  # 至少保留一半
                    chunk_text = chunk_text[:last_sentence_end + 1]
                    end = start + last_sentence_end + 1

            # 跳过空白块
            if chunk_text.strip():
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text.strip(),
                    'start_pos': start,
                    'end_pos': end
                })
                chunk_id += 1

            # 移动到下一个块（带重叠）
            start = end - self.chunk_overlap

        print(f"✓ 文本分割完成，共 {len(chunks)} 个块")
        return chunks

    def add_pdf_to_knowledge_base(self, pdf_path: str) -> int:
        """
        将 PDF 添加到知识库

        Args:
            pdf_path: PDF 文件路径

        Returns:
            添加的文档块数量
        """
        # 提取文本
        text = self.extract_text_from_pdf(pdf_path)

        # 分块
        chunks = self.chunk_text(text)

        # 生成 embeddings 并插入
        print(f"正在生成 embeddings 并插入数据库...")
        records = []

        for i, chunk in enumerate(chunks):
            # 生成 embedding
            embedding = self.embedder.encode(chunk['text']).tolist()

            # 创建记录
            record = {
                "id": str(chunk['id']),
                "vector": embedding,
                "metadata": {
                    "text": chunk['text'],
                    "source": os.path.basename(pdf_path),
                    "chunk_id": chunk['id'],
                    "start_pos": chunk['start_pos'],
                    "end_pos": chunk['end_pos']
                }
            }
            records.append(record)

            if (i + 1) % 50 == 0:
                print(f"  已处理 {i + 1}/{len(chunks)} 个块...")

        # 批量插入
        print("正在批量插入向量数据库...")
        self.client.insert_records(self.table_name, records)
        print(f"✓ 成功插入 {len(records)} 个文档块到知识库")

        return len(records)

    def search_documents(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        搜索相关文档

        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 最低相似度阈值

        Returns:
            相关文档列表
        """
        # 生成查询 embedding
        query_embedding = self.embedder.encode(query).tolist()

        # 搜索
        results = self.client.search(
            table_name=self.table_name,
            vector=query_embedding,
            k=k
        )

        # 过滤
        filtered_results = [
            r for r in results
            if r.get('score', 0.0) >= score_threshold
        ]

        return filtered_results

    def generate_answer(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        max_tokens: int = 500
    ) -> str:
        """
        使用 LLM 生成答案

        Args:
            query: 用户问题
            context_docs: 检索到的文档
            max_tokens: 最大 token 数

        Returns:
            生成的答案
        """
        if not self.openai_client:
            return "OpenAI 未配置，无法生成答案。请设置 OPENAI_API_KEY 环境变量。"

        # 构建上下文
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            text = doc.get('metadata', {}).get('text', '')
            source = doc.get('metadata', {}).get('source', 'unknown')
            chunk_id = doc.get('metadata', {}).get('chunk_id', 0)
            score = doc.get('score', 0.0)

            context_parts.append(
                f"[{i}] (来源: {source}, 块#{chunk_id}, 相似度: {score:.3f})\n{text}"
            )

        context = "\n\n".join(context_parts)

        # 创建提示词
        system_prompt = """你是一个专业的技术助手，基于提供的文档内容回答问题。
如果文档中没有足够的信息来回答问题，请明确说明。
回答时请引用具体的文档来源 [编号]。"""

        user_prompt = f"""参考文档：
{context}

问题：{query}

请基于以上文档内容详细回答问题，并标注信息来源。"""

        try:
            # 调用 OpenAI API
            response = self.openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"生成答案时出错: {str(e)}"

    def ask(
        self,
        question: str,
        k: int = 5,
        score_threshold: float = 0.3,
        show_context: bool = True
    ) -> Dict[str, Any]:
        """
        完整的问答流程

        Args:
            question: 问题
            k: 检索文档数量
            score_threshold: 相似度阈值
            show_context: 是否显示上下文

        Returns:
            包含答案和上下文的字典
        """
        print(f"\n{'='*80}")
        print(f"问题: {question}")
        print(f"{'='*80}\n")

        # 检索文档
        print(f"正在检索相关文档 (top-{k})...")
        docs = self.search_documents(question, k, score_threshold)
        print(f"✓ 检索到 {len(docs)} 个相关文档\n")

        # 显示上下文
        if show_context and docs:
            print("检索到的文档：")
            print("-" * 80)
            for i, doc in enumerate(docs, 1):
                text = doc.get('metadata', {}).get('text', '')
                source = doc.get('metadata', {}).get('source', 'unknown')
                chunk_id = doc.get('metadata', {}).get('chunk_id', 0)
                score = doc.get('score', 0.0)

                # 显示文本片段（前200字符）
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"[{i}] {source} (块#{chunk_id}, 相似度: {score:.3f})")
                print(f"    {preview}")
                print()

        # 生成答案
        if self.openai_client:
            print("正在生成答案...")
            answer = self.generate_answer(question, docs)
            print(f"\n{'='*80}")
            print("答案:")
            print(f"{'='*80}")
            print(answer)
            print(f"{'='*80}\n")
        else:
            answer = "OpenAI 未配置，仅显示检索结果"
            print(f"⚠️  {answer}\n")

        return {
            'question': question,
            'answer': answer,
            'context_docs': docs,
            'context_count': len(docs)
        }


def main():
    """主函数"""
    print("=" * 80)
    print("PDF RAG Demo")
    print("=" * 80)
    print()

    # 检查依赖
    if not HAS_PYPDF2:
        print("❌ 缺少 PyPDF2 库")
        print("   安装命令: pip install pypdf2")
        return

    # PDF 路径：自备任意 PDF，通过 DEMO_PDF_PATH 指定
    pdf_path = os.environ.get("DEMO_PDF_PATH", "pdf/sample.pdf")
    if not os.path.exists(pdf_path):
        print(f"❌ PDF 文件不存在: {pdf_path}")
        print("   本仓库不附带演示文档，请自备一份 PDF 后指定路径：")
        print("   DEMO_PDF_PATH=/path/to/your.pdf python examples/pdf_rag_demo.py")
        return

    # 连接 DXMT
    print("连接 DXMT 服务器...")
    try:
        client = Client(host="localhost", port=9999)
        print("✓ 连接成功")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("   请确保 DXMT 服务器正在运行:")
        print("   cargo run --release -- serve --host 0.0.0.0 --port 9999")
        return

    # 创建表
    table_name = "altair_feko_docs"
    dimension = 384  # all-MiniLM-L6-v2

    print(f"\n准备表 '{table_name}'...")
    try:
        client.delete_table(table_name)
        print(f"  删除旧表")
    except:
        pass

    client.create_table(
        name=table_name,
        dimension=dimension,
        distance_metric="cosine"
    )
    print(f"✓ 表创建成功 (dimension={dimension})")

    # 初始化 RAG 系统
    print("\n初始化 PDF RAG 系统...")
    rag = PDFRAGSystem(
        client=client,
        table_name=table_name,
        chunk_size=800,  # 较大的块以保留更多上下文
        chunk_overlap=100
    )

    # 处理 PDF
    print(f"\n{'='*80}")
    print("步骤 1: 处理 PDF 并导入知识库")
    print(f"{'='*80}\n")

    num_chunks = rag.add_pdf_to_knowledge_base(pdf_path)
    print(f"\n✓ PDF 处理完成，共导入 {num_chunks} 个文档块")

    # 示例问答
    print(f"\n{'='*80}")
    print("步骤 2: RAG 问答测试")
    print(f"{'='*80}\n")

    # 预定义问题
    test_questions = [
        "What is this document about?",
        "What types of antennas are discussed in the guide?",
        "Can you explain how to simulate electromagnetic fields?",
        "What are the main features described?"
    ]

    print("示例问题：")
    for i, q in enumerate(test_questions, 1):
        print(f"  {i}. {q}")
    print()

    # 自动回答第一个问题
    rag.ask(test_questions[0], k=3, score_threshold=0.2)

    # 交互模式
    print(f"\n{'='*80}")
    print("步骤 3: 交互式问答")
    print(f"{'='*80}\n")
    print("您现在可以提问关于该文档的问题")
    print("输入 'quit' 或 'exit' 退出")
    print("输入问题编号 (1-4) 可以快速提问预定义问题")
    print()

    if not sys.stdin.isatty():
        print("非交互运行，跳过问答循环。加 -t 或直接运行本文件即可进入。")
        return

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
                if 1 <= q_num <= len(test_questions):
                    question = test_questions[q_num - 1]
                    rag.ask(question, k=3, score_threshold=0.2)
                else:
                    print(f"⚠️  问题编号必须在 1-{len(test_questions)} 之间")
                continue

            # 普通问题
            rag.ask(user_input, k=3, score_threshold=0.2)

        except (KeyboardInterrupt, EOFError):
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")

    # 清理（可选）
    print(f"\n{'='*80}")
    print("清理")
    print(f"{'='*80}\n")

    # Piped or run by a script, there is nobody to ask. Prompting anyway raised
    # `EOFError: EOF when reading a line` and took the example's exit code with it, which is
    # how an example that works by hand fails every time something tries to run it for you.
    if not sys.stdin.isatty():
        print("非交互运行，保留表 '%s'" % table_name)
        cleanup = "n"
    else:
        cleanup = input("是否删除表？(y/N) ").strip().lower()
    if cleanup == 'y':
        try:
            client.delete_table(table_name)
            print(f"✓ 已删除表 '{table_name}'")
        except Exception as e:
            print(f"❌ 删除表失败: {e}")
    else:
        print(f"保留表 '{table_name}'，您可以稍后继续使用")

    print("\n✓ 演示完成！")


if __name__ == "__main__":
    main()
