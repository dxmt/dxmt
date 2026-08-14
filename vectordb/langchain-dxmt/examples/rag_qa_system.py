"""RAG (Retrieval-Augmented Generation) Q&A System with DXMT.

This example demonstrates building a question-answering system using:
- DXMT vector store for document storage and retrieval
- OpenAI embeddings for semantic understanding
- OpenAI LLM for answer generation
- LangChain's LCEL (LangChain Expression Language) for RAG chains

Prerequisites:
- DXMT server running on localhost:9999
- OpenAI API key set in environment: export OPENAI_API_KEY='sk-...'

Install dependencies:
    pip install langchain-dxmt langchain-openai langchain-core

Use case:
    Build a Q&A system that answers questions about a company's documentation,
    product manuals, or knowledge base using semantic search + LLM generation.
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_dxmt import DXMTVectorDB


def create_sample_documents() -> List[Document]:
    """Create sample company documentation."""
    docs = [
        Document(
            page_content="""
            DXMT VectorDB is a lightweight, horizontally scalable vector database
            designed for AI applications. It supports multiple index types including
            HNSW, IVF, IVF-PQ, and LSH for different performance requirements.
            """,
            metadata={
                "source": "product_overview.md",
                "category": "product",
                "last_updated": "2026-01-15"
            }
        ),
        Document(
            page_content="""
            DXMT offers hybrid search capabilities that combine vector similarity
            search with full-text search (BM25). This allows finding documents that
            are both semantically similar and contain specific keywords. The
            vector_weight parameter controls the balance (0.0 = pure text, 1.0 = pure vector).
            """,
            metadata={
                "source": "hybrid_search_guide.md",
                "category": "features",
                "last_updated": "2026-01-10"
            }
        ),
        Document(
            page_content="""
            To achieve optimal performance, DXMT provides multiple insert modes:
            - Standard: Default balanced mode
            - LockFree: High-throughput parallel inserts
            - Ultimate: Maximum performance for batch operations
            Choose based on your workload: real-time vs batch processing.
            """,
            metadata={
                "source": "performance_tuning.md",
                "category": "optimization",
                "last_updated": "2026-01-12"
            }
        ),
        Document(
            page_content="""
            DXMT supports Chinese full-text search with Jieba tokenization.
            This enables accurate semantic search for Chinese documents with
            proper word segmentation. Simply enable full_text_search when
            creating the table and use hybrid search for best results.
            """,
            metadata={
                "source": "chinese_support.md",
                "category": "features",
                "last_updated": "2026-01-08"
            }
        ),
        Document(
            page_content="""
            The LangChain integration (langchain-dxmt) provides a standard
            VectorStore interface. It supports all standard operations including
            add_documents, similarity_search, as_retriever, and includes async
            variants for high-performance applications.
            """,
            metadata={
                "source": "langchain_integration.md",
                "category": "integration",
                "last_updated": "2026-01-20"
            }
        ),
        Document(
            page_content="""
            DXMT pricing is straightforward: self-hosted open-source version is free.
            Cloud version starts at $29/month for small workloads (up to 1M vectors).
            Enterprise plans include dedicated support, SLA guarantees, and custom
            deployments. Contact sales for enterprise pricing.
            """,
            metadata={
                "source": "pricing.md",
                "category": "business",
                "last_updated": "2026-01-05"
            }
        ),
        Document(
            page_content="""
            Installation is simple: pip install dxmt for the Python SDK, or
            use Docker for the server: docker run -p 9999:9999 dxmt/server.
            The server supports configuration via environment variables or
            a config file. Default settings work for most use cases.
            """,
            metadata={
                "source": "installation.md",
                "category": "getting_started",
                "last_updated": "2026-01-18"
            }
        ),
        Document(
            page_content="""
            DXMT provides comprehensive monitoring and observability. Built-in
            metrics include query latency, index size, throughput, and error rates.
            Metrics can be exported to Prometheus, Grafana, or other monitoring
            systems via the /metrics endpoint.
            """,
            metadata={
                "source": "monitoring.md",
                "category": "operations",
                "last_updated": "2026-01-14"
            }
        ),
    ]
    return docs


def main():
    """Run the RAG Q&A system example."""
    print("=" * 80)
    print("RAG (Retrieval-Augmented Generation) Q&A System")
    print("=" * 80)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        return

    # 1. Initialize embeddings and vector store
    print("\n1. Initializing embeddings and vector store...")
    embeddings = OpenAIEmbeddings()

    vectorstore = DXMTVectorDB(
        table_name="rag_qa_demo",
        embedding=embeddings,
        dimension=1536,  # OpenAI ada-002 dimension
        host="localhost",
        port=9999,
        distance_metric="cosine",
        index_type="hnsw",
    )
    print("   ✅ Vector store initialized")

    # 2. Load and index documents
    print("\n2. Loading company documentation...")
    docs = create_sample_documents()
    print(f"   Created {len(docs)} sample documents")

    print("\n3. Indexing documents in DXMT...")
    ids = vectorstore.add_documents(docs)
    print(f"   ✅ Indexed {len(ids)} documents")

    # 3. Initialize LLM
    print("\n4. Initializing ChatGPT (gpt-3.5-turbo)...")
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,  # Deterministic answers
    )
    print("   ✅ LLM initialized")

    # 4. Create RAG chain using LCEL (LangChain Expression Language)
    print("\n5. Creating RAG chain with LCEL...")

    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:

{context}

Question: {question}

Provide a clear, concise answer based on the information in the context.""")

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Helper function to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create the LCEL chain
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("   ✅ RAG chain created")

    # 5. Ask questions
    print("\n" + "=" * 80)
    print("Asking Questions")
    print("=" * 80)

    questions = [
        "What is DXMT VectorDB and what index types does it support?",
        "How does hybrid search work in DXMT?",
        "What are the different insert modes and when should I use them?",
        "Does DXMT support Chinese text? How?",
        "How much does DXMT cost?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'─' * 80}")
        print(f"Question {i}: {question}")
        print(f"{'─' * 80}")

        # Get answer from RAG chain
        answer = qa_chain.invoke(question)

        # Get source documents separately
        source_docs = retriever.invoke(question)

        print(f"\n📝 Answer:\n{answer}")

        print(f"\n📚 Sources ({len(source_docs)} documents):")
        for j, doc in enumerate(source_docs, 1):
            source = doc.metadata.get("source", "unknown")
            category = doc.metadata.get("category", "N/A")
            snippet = doc.page_content.strip()[:100] + "..."
            print(f"   {j}. {source} ({category})")
            print(f"      {snippet}")

    # 6. Interactive Q&A (optional)
    print("\n" + "=" * 80)
    print("Interactive Q&A Mode")
    print("=" * 80)
    print("Type your questions (or 'quit' to exit):\n")

    try:
        while True:
            question = input("❓ You: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break

            if not question:
                continue

            answer = qa_chain.invoke(question)

            print(f"\n🤖 Assistant: {answer}\n")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")

    # 7. Cleanup
    print("\n" + "=" * 80)
    print("Cleanup")
    print("=" * 80)

    print("\nDeleting documents...")
    vectorstore.delete(ids)
    print("✅ Documents deleted")

    print("\n" + "=" * 80)
    print("💡 Key Insights")
    print("=" * 80)
    print("""
    1. RAG combines retrieval + generation for accurate, grounded answers
    2. DXMT's fast semantic search retrieves relevant context
    3. LLM generates natural language answers based on retrieved documents
    4. Source documents provide transparency and allow verification
    5. The system can answer questions beyond the LLM's training data

    💡 Next Steps:
    - Add more documents to expand knowledge base
    - Use hybrid_search() for better keyword matching
    - Add metadata filters for domain-specific searches
    - Implement conversation memory for follow-up questions
    - Use streaming for real-time response generation
    """)


if __name__ == "__main__":
    main()
