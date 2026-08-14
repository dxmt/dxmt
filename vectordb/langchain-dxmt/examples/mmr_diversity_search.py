"""
MMR (Maximal Marginal Relevance) Diversity Search Example

This example demonstrates how to use MMR to retrieve diverse results,
avoiding redundancy in search results.

MMR balances relevance and diversity:
- High relevance: Returns results most similar to query
- High diversity: Returns results different from each other

Use cases:
- News article recommendations (avoid similar stories)
- Product search (show variety)
- Research paper exploration (diverse perspectives)
- Content discovery (avoid echo chambers)
"""

import os
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_dxmt import DXMTVectorDB


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='sk-...'")
        return

    print_section("MMR (Maximal Marginal Relevance) Diversity Search")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Create vector store
    print("Initializing DXMT vector store...")
    vectorstore = DXMTVectorDB(
        table_name="mmr_diversity_demo",
        embedding=embeddings,
        dimension=1536,
        host="localhost",
        port=9999,
        distance_metric="cosine",
    )

    # Clean up any existing data
    try:
        from dxmt import Client
        client = Client(host="localhost", port=9999)
        client.delete_table("mmr_diversity_demo")
        print("   ✓ Cleaned existing table")
    except:
        print("   ℹ No existing table to clean")

    # Recreate vector store after cleanup
    vectorstore = DXMTVectorDB(
        table_name="mmr_diversity_demo",
        embedding=embeddings,
        dimension=1536,
        host="localhost",
        port=9999,
        distance_metric="cosine",
    )

    # Sample documents about programming languages
    # Notice: Some are very similar, some are diverse
    docs = [
        # Python cluster (similar documents)
        Document(
            page_content="Python is a high-level programming language known for its simplicity and readability. It's widely used in data science and machine learning.",
            metadata={"language": "Python", "category": "overview", "id": "1"}
        ),
        Document(
            page_content="Python programming language features dynamic typing and automatic memory management. It supports multiple programming paradigms.",
            metadata={"language": "Python", "category": "features", "id": "2"}
        ),
        Document(
            page_content="Python has a vast ecosystem of libraries like NumPy, Pandas, and TensorFlow for scientific computing and AI.",
            metadata={"language": "Python", "category": "ecosystem", "id": "3"}
        ),
        # Other languages (diverse)
        Document(
            page_content="JavaScript is the language of the web, running in browsers to create interactive websites and web applications.",
            metadata={"language": "JavaScript", "category": "overview", "id": "4"}
        ),
        Document(
            page_content="Rust is a systems programming language focused on safety, speed, and concurrency without garbage collection.",
            metadata={"language": "Rust", "category": "overview", "id": "5"}
        ),
        Document(
            page_content="Go (Golang) is a statically typed language designed at Google for building scalable and concurrent server applications.",
            metadata={"language": "Go", "category": "overview", "id": "6"}
        ),
        Document(
            page_content="Java is an object-oriented language used for enterprise applications, Android development, and big data processing.",
            metadata={"language": "Java", "category": "overview", "id": "7"}
        ),
        Document(
            page_content="C++ is a powerful language for game development, high-performance applications, and system programming.",
            metadata={"language": "C++", "category": "overview", "id": "8"}
        ),
    ]

    print("Adding documents to vector store...")
    vectorstore.add_documents(docs)
    print(f"   ✓ Added {len(docs)} documents\n")

    # Example 1: Standard similarity search (may return redundant results)
    print_section("Example 1: Standard Similarity Search")
    print("Query: 'Tell me about Python programming language'")
    print("Method: similarity_search (k=4)\n")

    standard_results = vectorstore.similarity_search(
        "Tell me about Python programming language",
        k=4
    )

    print("Results (may contain redundant Python docs):")
    for i, doc in enumerate(standard_results, 1):
        lang = doc.metadata.get("language", "Unknown")
        category = doc.metadata.get("category", "Unknown")
        preview = doc.page_content[:60] + "..."
        print(f"   {i}. [{lang} - {category}] {preview}")

    # Example 2: MMR search with low diversity (lambda=1.0)
    print_section("Example 2: MMR with Low Diversity (lambda=1.0)")
    print("Query: 'Tell me about Python programming language'")
    print("Method: max_marginal_relevance_search (lambda_mult=1.0)")
    print("Note: lambda_mult=1.0 means 100% relevance, 0% diversity (similar to standard search)\n")

    mmr_low_diversity = vectorstore.max_marginal_relevance_search(
        "Tell me about Python programming language",
        k=4,
        lambda_mult=1.0  # 100% relevance, 0% diversity
    )

    print("Results:")
    for i, doc in enumerate(mmr_low_diversity, 1):
        lang = doc.metadata.get("language", "Unknown")
        category = doc.metadata.get("category", "Unknown")
        preview = doc.page_content[:60] + "..."
        print(f"   {i}. [{lang} - {category}] {preview}")

    # Example 3: MMR search with high diversity (lambda=0.5)
    print_section("Example 3: MMR with Balanced Diversity (lambda=0.5)")
    print("Query: 'Tell me about Python programming language'")
    print("Method: max_marginal_relevance_search (lambda_mult=0.5)")
    print("Note: lambda_mult=0.5 means 50% relevance, 50% diversity (balanced)\n")

    mmr_balanced = vectorstore.max_marginal_relevance_search(
        "Tell me about Python programming language",
        k=4,
        lambda_mult=0.5  # 50% relevance, 50% diversity
    )

    print("Results (more diverse languages):")
    for i, doc in enumerate(mmr_balanced, 1):
        lang = doc.metadata.get("language", "Unknown")
        category = doc.metadata.get("category", "Unknown")
        preview = doc.page_content[:60] + "..."
        print(f"   {i}. [{lang} - {category}] {preview}")

    # Example 4: MMR search with maximum diversity (lambda=0.0)
    print_section("Example 4: MMR with Maximum Diversity (lambda=0.0)")
    print("Query: 'Tell me about Python programming language'")
    print("Method: max_marginal_relevance_search (lambda_mult=0.0)")
    print("Note: lambda_mult=0.0 means 0% relevance, 100% diversity (most diverse)\n")

    mmr_high_diversity = vectorstore.max_marginal_relevance_search(
        "Tell me about Python programming language",
        k=4,
        lambda_mult=0.0  # 0% relevance, 100% diversity
    )

    print("Results (maximum diversity, may be less relevant):")
    for i, doc in enumerate(mmr_high_diversity, 1):
        lang = doc.metadata.get("language", "Unknown")
        category = doc.metadata.get("category", "Unknown")
        preview = doc.page_content[:60] + "..."
        print(f"   {i}. [{lang} - {category}] {preview}")

    # Example 5: Using MMR as a retriever
    print_section("Example 5: MMR Retriever for RAG")
    print("Using MMR retriever with LangChain chains for diverse context\n")

    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "lambda_mult": 0.6  # 60% relevance, 40% diversity
        }
    )

    # Retrieve documents
    retrieved_docs = mmr_retriever.invoke("programming languages for web development")

    print("Retrieved documents for RAG:")
    for i, doc in enumerate(retrieved_docs, 1):
        lang = doc.metadata.get("language", "Unknown")
        preview = doc.page_content[:60] + "..."
        print(f"   {i}. [{lang}] {preview}")

    # Comparison summary
    print_section("Summary: When to Use MMR")

    print("🎯 Use MMR when you want:")
    print("   • Diverse perspectives on a topic")
    print("   • Avoid redundant/similar results")
    print("   • Explore different categories")
    print("   • Content discovery and recommendations")
    print("")
    print("📊 Lambda parameter guide:")
    print("   • lambda=1.0: Pure relevance (like standard search)")
    print("   • lambda=0.7: Slight diversity (recommended for Q&A)")
    print("   • lambda=0.5: Balanced (good for exploration)")
    print("   • lambda=0.3: High diversity (for recommendations)")
    print("   • lambda=0.0: Maximum diversity (may sacrifice relevance)")
    print("")
    print("💡 Recommended values:")
    print("   • Q&A systems: 0.6-0.8 (favor relevance)")
    print("   • Content recommendations: 0.3-0.5 (favor diversity)")
    print("   • Research/exploration: 0.4-0.6 (balanced)")

    # Cleanup
    print_section("Cleanup")
    try:
        client.delete_table("mmr_diversity_demo")
        print("   ✓ Cleaned up demo table")
    except Exception as e:
        print(f"   ℹ Cleanup: {e}")


if __name__ == "__main__":
    main()
