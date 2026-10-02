"""
Contextual Compression Retriever Example

This example demonstrates how to use ContextualCompressionRetriever to extract
only the most relevant parts of retrieved documents, reducing noise and improving
LLM context efficiency.

Problem: Retrieved documents often contain irrelevant information
Solution: Use LLM to extract only query-relevant passages

How it works:
1. Base retriever fetches candidate documents
2. Compressor LLM extracts relevant passages
3. Returns compressed, focused content

Benefits:
- Reduces token usage (saves cost)
- Improves answer quality (less noise)
- Better LLM focus on relevant content
- Faster processing (less text to analyze)

Use cases:
- Long documents with mixed content
- Reducing LLM context window usage
- Improving RAG answer quality
- Cost optimization for production systems
"""

import os
from langchain_core.documents import Document
from embedding import default_embeddings, embed_dimension, needs_an_llm

# `langchain_openai` is imported where it is used, not here: without the
# package this example died at import time, before it could say what it needed.
from langchain_dxmt import DXMTVectorDB

# Try multiple import paths for different langchain versions
try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
except ImportError:
    try:
        from langchain_community.retrievers import ContextualCompressionRetriever
        from langchain_community.retrievers.document_compressors import LLMChainExtractor
    except ImportError:
        try:
            from langchain_classic.retrievers import ContextualCompressionRetriever
            from langchain_classic.retrievers.document_compressors import LLMChainExtractor
        except ImportError:
            print("⚠️ ContextualCompressionRetriever not available.")
            print("Tried import paths:")
            print("  - langchain.retrievers")
            print("  - langchain_community.retrievers")
            print("  - langchain_classic.retrievers")
            print("")
            print("💡 Recommendation: Use the simplified version instead:")
            print("   python3 contextual_compression_simple.py")
            print("")
            print("Or try: pip install langchain langchain-community")
            exit(1)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def count_tokens(text: str) -> int:
    """Rough token count estimation."""
    return len(text.split())


def main():
    # Check for OpenAI API key
    if not needs_an_llm("this example"):
        return

    from langchain_openai import ChatOpenAI

    print_section("Contextual Compression: Reducing Noise in Retrieval")

    # Initialize components
    embeddings = default_embeddings()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Create vector store
    print("Initializing DXMT vector store...")
    vectorstore = DXMTVectorDB(
        table_name="compression_demo",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
        host="localhost",
        port=9999,
    )

    # Clean up
    try:
        from dxmt import Client
        client = Client(host="localhost", port=9999)
        client.delete_table("compression_demo")
        print("   ✓ Cleaned existing table")
    except:
        print("   ℹ No existing table to clean")

    # Recreate vector store
    vectorstore = DXMTVectorDB(
        table_name="compression_demo",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
        host="localhost",
        port=9999,
    )

    # Sample documents with mixed relevant/irrelevant content
    # (Simulating real-world documents with noise)
    docs = [
        Document(
            page_content="""
DXMT VectorDB is a lightweight single-node vector database.
It supports multiple index types including HNSW, IVF, and BruteForce.
The company was founded in 2023 by a team of database engineers.
Our office is located in San Francisco, California.
DXMT offers both cloud and on-premise deployments.
The pricing starts at $99/month for the starter plan.
For index selection, HNSW is recommended for datasets over 10K vectors
due to its superior query performance. IVF is better for very large datasets.
Contact us at support@dxmt.online for more information.
            """.strip(),
            metadata={"doc_id": "1", "type": "product_info"}
        ),
        Document(
            page_content="""
Hybrid search combines vector similarity with full-text search.
The main benefit is improved recall for keyword-specific queries.
Our company headquarters features a modern open office design.
We offer free lunch and snacks to all employees.
In hybrid search, you can adjust the vector_weight parameter between 0 and 1.
A weight of 0.7 means 70% vector search and 30% text search.
The annual company retreat is held in Hawaii every summer.
For best results with hybrid search, enable full-text indexing when creating tables.
Benefits package includes health insurance and 401k matching.
            """.strip(),
            metadata={"doc_id": "2", "type": "feature_docs"}
        ),
        Document(
            page_content="""
To install DXMT, use pip install dxmt or download from our website.
System requirements include Python 3.8 or higher.
The installation wizard has a user-friendly interface with progress bars.
Our customer support team is available 24/7 via email and chat.
After installation, start the server with the command: dxmt serve --port 9999
Default configuration uses localhost with port 9999.
We have a community forum with over 5000 active members.
For production deployments, we recommend using Docker containers.
The company was named "Best Startup 2024" by TechCrunch.
            """.strip(),
            metadata={"doc_id": "3", "type": "installation"}
        ),
        Document(
            page_content="""
Performance benchmarks show DXMT achieves 3.5M QPS on standard hardware.
Our testing lab is equipped with the latest NVIDIA GPUs.
Memory usage scales linearly with dataset size.
The break room has a PlayStation 5 and ping pong table.
For optimal performance, allocate at least 8GB RAM per million vectors.
The tiered storage architecture reduces costs by 70% compared to in-memory solutions.
CEO John Smith has 20 years of experience in database systems.
Latency typically ranges from 5-10ms for 95th percentile queries.
The company logo was designed by a famous artist from New York.
            """.strip(),
            metadata={"doc_id": "4", "type": "performance"}
        ),
    ]

    print("Adding documents with mixed content (relevant + noise)...")
    vectorstore.add_documents(docs)
    print(f"   ✓ Added {len(docs)} documents\n")

    # Example 1: Standard retrieval (with noise)
    print_section("Example 1: Standard Retrieval (Baseline with Noise)")

    query = "How do I configure hybrid search in DXMT?"
    print(f"Query: \"{query}\"\n")

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    standard_results = base_retriever.invoke(query)

    print("Retrieved documents (full content, including noise):\n")
    total_tokens_before = 0
    for i, doc in enumerate(standard_results, 1):
        tokens = count_tokens(doc.page_content)
        total_tokens_before += tokens
        print(f"Document {i} ({tokens} tokens):")
        print(f"Type: {doc.metadata['type']}")
        print(f"Content:\n{doc.page_content}\n")
        print("-" * 80)

    print(f"\n📊 Total tokens (before compression): {total_tokens_before}")

    # Example 2: Contextual compression
    print_section("Example 2: Contextual Compression (Noise Removed)")

    print(f"Query: \"{query}\"\n")

    # Create compressor
    compressor = LLMChainExtractor.from_llm(llm)

    # Create compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # Retrieve and compress
    compressed_results = compression_retriever.invoke(query)

    print("Compressed documents (only relevant passages):\n")
    total_tokens_after = 0
    for i, doc in enumerate(compressed_results, 1):
        tokens = count_tokens(doc.page_content)
        total_tokens_after += tokens
        print(f"Document {i} ({tokens} tokens):")
        print(f"Type: {doc.metadata['type']}")
        print(f"Content:\n{doc.page_content}\n")
        print("-" * 80)

    print(f"\n📊 Total tokens (after compression): {total_tokens_after}")

    if total_tokens_before > 0:
        reduction = ((total_tokens_before - total_tokens_after) / total_tokens_before) * 100
        print(f"✅ Token reduction: {reduction:.1f}%")
        print(f"💰 Cost savings: ~{reduction:.1f}% on LLM API calls")

    # Example 3: Using in RAG chain
    print_section("Example 3: Compression in RAG Q&A Chain")

    from langchain.chains import RetrievalQA

    # Without compression
    print("Building Q&A chain WITHOUT compression...\n")
    qa_standard = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=base_retriever,
        return_source_documents=True
    )

    result_standard = qa_standard.invoke({"query": query})

    print("Answer (without compression):")
    print(result_standard["result"])
    std_tokens = sum(count_tokens(doc.page_content)
                     for doc in result_standard["source_documents"])
    print(f"\nContext size: {std_tokens} tokens")

    # With compression
    print("\n" + "-" * 80 + "\n")
    print("Building Q&A chain WITH compression...\n")
    qa_compressed = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=compression_retriever,
        return_source_documents=True
    )

    result_compressed = qa_compressed.invoke({"query": query})

    print("Answer (with compression):")
    print(result_compressed["result"])
    comp_tokens = sum(count_tokens(doc.page_content)
                      for doc in result_compressed["source_documents"])
    print(f"\nContext size: {comp_tokens} tokens")

    if std_tokens > 0:
        reduction = ((std_tokens - comp_tokens) / std_tokens) * 100
        print(f"\n✅ Context reduction: {reduction:.1f}%")

    # Example 4: Different queries comparison
    print_section("Example 4: Compression Impact on Different Queries")

    test_queries = [
        "What is the pricing for DXMT?",
        "How do I install DXMT?",
        "What is DXMT's query performance?",
    ]

    for q in test_queries:
        print(f"\n📝 Query: \"{q}\"")
        print("-" * 80)

        # Standard
        std_docs = base_retriever.invoke(q)
        std_tokens = sum(count_tokens(doc.page_content) for doc in std_docs)

        # Compressed
        comp_docs = compression_retriever.invoke(q)
        comp_tokens = sum(count_tokens(doc.page_content) for doc in comp_docs)

        reduction = ((std_tokens - comp_tokens) / std_tokens * 100) if std_tokens > 0 else 0

        print(f"Standard: {std_tokens} tokens → Compressed: {comp_tokens} tokens")
        print(f"Reduction: {reduction:.1f}%")

        # Show one compressed example
        if comp_docs:
            print(f"\nCompressed excerpt:")
            print(f"   {comp_docs[0].page_content[:100]}...")

    # Best practices summary
    print_section("Summary: Contextual Compression Best Practices")

    print("✅ When to use Contextual Compression:")
    print("   • Documents contain mixed relevant/irrelevant content")
    print("   • Need to reduce LLM token usage and costs")
    print("   • Want to improve answer focus and quality")
    print("   • Working with limited context windows")
    print("")
    print("⚠️  Considerations:")
    print("   • Adds LLM call latency (~500-1000ms per document)")
    print("   • Uses tokens for compression (but saves more on generation)")
    print("   • May remove context that seems irrelevant but isn't")
    print("   • Best for documents > 200 words with mixed content")
    print("")
    print("🎯 Configuration tips:")
    print("   • Use with k=5-10 base documents, compress to 2-3")
    print("   • Consider compression only for long documents (>150 words)")
    print("   • Test with your specific document types")
    print("   • Monitor compression quality over time")
    print("")
    print("💡 Cost-benefit analysis:")
    print("   • Compression cost: 1 LLM call per document")
    print("   • Savings: 40-70% reduction in generation tokens")
    print("   • Break-even: Usually beneficial for k >= 3")
    print("   • Best ROI: Long documents with high noise")
    print("")
    print("🔧 Advanced options:")
    print("   • LLMChainExtractor: Simple extraction (used here)")
    print("   • LLMChainFilter: Filter entire documents")
    print("   • EmbeddingsFilter: Fast semantic filtering")
    print("   • DocumentCompressorPipeline: Chain multiple compressors")

    # Cleanup
    print_section("Cleanup")
    try:
        client.delete_table("compression_demo")
        print("   ✓ Cleaned up demo table")
    except Exception as e:
        print(f"   ℹ Cleanup: {e}")

    print("\n" + "=" * 80)
    print("  Key Takeaway")
    print("=" * 80)
    print("""
Contextual Compression is a powerful technique for:
1. Reducing costs (40-70% token savings)
2. Improving answer quality (less noise)
3. Better utilizing limited context windows
4. Scaling RAG systems more efficiently

Trade-off: Adds latency but saves tokens and improves quality.
Recommendation: Use for production RAG systems with cost concerns.
    """)


if __name__ == "__main__":
    main()
