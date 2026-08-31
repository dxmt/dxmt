"""
Multi-Query Retriever Example

This example demonstrates how to use MultiQueryRetriever to improve retrieval
quality by generating multiple query variations and combining results.

How it works:
1. User asks a question
2. LLM generates 3-5 alternative phrasings of the question
3. Each variation is used to retrieve documents
4. Results are de-duplicated and ranked

Benefits:
- Improves recall (finds more relevant documents)
- Handles ambiguous queries better
- More robust to query phrasing
- Better than single-query retrieval for complex questions

Use cases:
- Q&A systems where precision matters
- Research paper search
- Legal document retrieval
- Customer support knowledge bases
"""

import os
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_dxmt import DXMTVectorDB

# Try multiple import paths for different langchain versions
try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
except ImportError:
    try:
        from langchain_community.retrievers.multi_query import MultiQueryRetriever
    except ImportError:
        try:
            from langchain_classic.retrievers.multi_query import MultiQueryRetriever
        except ImportError:
            print("⚠️ MultiQueryRetriever not available.")
            print("Tried import paths:")
            print("  - langchain.retrievers.multi_query")
            print("  - langchain_community.retrievers.multi_query")
            print("  - langchain_classic.retrievers.multi_query")
            print("")
            print("💡 Recommendation: Use the simplified version instead:")
            print("   python3 multi_query_retriever_simple.py")
            print("")
            print("Or try: pip install langchain langchain-community")
            exit(1)


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

    print_section("Multi-Query Retriever: Improved Retrieval Quality")

    # Initialize components
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Create vector store
    print("Initializing DXMT vector store...")
    vectorstore = DXMTVectorDB(
        table_name="multi_query_demo",
        embedding=embeddings,
        dimension=1536,
        host="localhost",
        port=9999,
    )

    # Clean up any existing data
    try:
        from dxmt import Client
        client = Client(host="localhost", port=9999)
        client.delete_table("multi_query_demo")
        print("   ✓ Cleaned existing table")
    except:
        print("   ℹ No existing table to clean")

    # Recreate vector store
    vectorstore = DXMTVectorDB(
        table_name="multi_query_demo",
        embedding=embeddings,
        dimension=1536,
        host="localhost",
        port=9999,
    )

    # Sample documents about technology topics
    docs = [
        Document(
            page_content="Machine learning models require large datasets for training. The quality and quantity of training data directly impacts model performance. Data preprocessing and cleaning are crucial steps.",
            metadata={"topic": "ML", "aspect": "data", "id": "1"}
        ),
        Document(
            page_content="Deep learning architectures like transformers have revolutionized natural language processing. Attention mechanisms allow models to focus on relevant parts of input sequences.",
            metadata={"topic": "DL", "aspect": "architecture", "id": "2"}
        ),
        Document(
            page_content="Cloud computing provides scalable infrastructure for deploying machine learning models. Services like AWS SageMaker and Google Cloud AI Platform simplify ML operations.",
            metadata={"topic": "Cloud", "aspect": "deployment", "id": "3"}
        ),
        Document(
            page_content="Data augmentation techniques can artificially increase training dataset size. Methods include rotation, flipping, noise injection, and synthetic data generation.",
            metadata={"topic": "ML", "aspect": "data-augmentation", "id": "4"}
        ),
        Document(
            page_content="Model evaluation metrics vary by task. Classification uses accuracy and F1-score, while regression uses MSE and R-squared. Cross-validation helps assess generalization.",
            metadata={"topic": "ML", "aspect": "evaluation", "id": "5"}
        ),
        Document(
            page_content="Transfer learning allows reusing pre-trained models for new tasks. Fine-tuning on domain-specific data often achieves better results than training from scratch.",
            metadata={"topic": "DL", "aspect": "transfer-learning", "id": "6"}
        ),
        Document(
            page_content="Feature engineering transforms raw data into meaningful representations. Good features can dramatically improve model performance, especially for traditional ML algorithms.",
            metadata={"topic": "ML", "aspect": "features", "id": "7"}
        ),
        Document(
            page_content="Hyperparameter tuning optimizes model configuration. Grid search, random search, and Bayesian optimization are common approaches. Automated tools like Optuna simplify the process.",
            metadata={"topic": "ML", "aspect": "optimization", "id": "8"}
        ),
        Document(
            page_content="Production ML systems require monitoring and retraining. Model drift occurs when data distribution changes. A/B testing helps evaluate model updates safely.",
            metadata={"topic": "MLOps", "aspect": "production", "id": "9"}
        ),
        Document(
            page_content="Explainable AI (XAI) helps understand model decisions. SHAP values and LIME provide local explanations. Feature importance analysis shows global patterns.",
            metadata={"topic": "XAI", "aspect": "interpretability", "id": "10"}
        ),
    ]

    print("Adding documents to vector store...")
    vectorstore.add_documents(docs)
    print(f"   ✓ Added {len(docs)} documents\n")

    # Example 1: Standard retrieval (baseline)
    print_section("Example 1: Standard Single-Query Retrieval (Baseline)")

    query = "How do I improve my ML model's accuracy?"
    print(f"User query: \"{query}\"")
    print(f"Method: Standard similarity_search\n")

    standard_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    standard_results = standard_retriever.invoke(query)

    print("Retrieved documents:")
    for i, doc in enumerate(standard_results, 1):
        topic = doc.metadata.get("topic", "Unknown")
        aspect = doc.metadata.get("aspect", "Unknown")
        preview = doc.page_content[:70] + "..."
        print(f"   {i}. [{topic}/{aspect}] {preview}")

    # Example 2: Multi-query retrieval
    print_section("Example 2: Multi-Query Retrieval (Enhanced)")

    print(f"User query: \"{query}\"")
    print(f"Method: MultiQueryRetriever (generates multiple query variations)\n")

    # Create multi-query retriever
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        llm=llm
    )

    # Enable logging to see generated queries
    import logging
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    print("Generated query variations (by LLM):")
    print("   (Check the logs above to see the variations)\n")

    multi_query_results = multi_query_retriever.invoke(query)

    print("Retrieved documents (de-duplicated and ranked):")
    for i, doc in enumerate(multi_query_results, 1):
        topic = doc.metadata.get("topic", "Unknown")
        aspect = doc.metadata.get("aspect", "Unknown")
        preview = doc.page_content[:70] + "..."
        print(f"   {i}. [{topic}/{aspect}] {preview}")

    # Example 3: Compare different queries
    print_section("Example 3: Comparison with Different Queries")

    test_queries = [
        "What data do I need for training?",
        "How to deploy ML models to production?",
        "Best practices for model evaluation?",
    ]

    for query_text in test_queries:
        print(f"\n📝 Query: \"{query_text}\"")
        print("-" * 80)

        # Standard retrieval
        standard = standard_retriever.invoke(query_text)
        print(f"Standard (1 query): {len(standard)} docs")
        for doc in standard[:2]:
            print(f"   • [{doc.metadata['topic']}] {doc.page_content[:50]}...")

        # Multi-query retrieval
        multi = multi_query_retriever.invoke(query_text)
        print(f"\nMulti-query (3-5 variations): {len(multi)} docs")
        for doc in multi[:2]:
            print(f"   • [{doc.metadata['topic']}] {doc.page_content[:50]}...")

    # Example 4: Using in a RAG chain
    print_section("Example 4: Multi-Query Retriever in RAG Chain")

    from langchain.chains import RetrievalQA

    print("Building Q&A chain with multi-query retriever...")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=multi_query_retriever,
        return_source_documents=True,
        verbose=False
    )

    question = "What are the best practices for improving model performance?"
    print(f"\nQuestion: {question}")
    print("\nGenerating answer with multi-query retrieval...\n")

    result = qa_chain.invoke({"query": question})

    print("Answer:")
    print(result["result"])
    print(f"\nSources ({len(result['source_documents'])} documents):")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"   {i}. {doc.metadata['topic']}/{doc.metadata['aspect']}")

    # Best practices summary
    print_section("Summary: Multi-Query Retriever Best Practices")

    print("✅ When to use Multi-Query Retriever:")
    print("   • Complex or ambiguous user queries")
    print("   • When recall is more important than speed")
    print("   • Q&A systems requiring comprehensive answers")
    print("   • Research and exploration tasks")
    print("")
    print("⚠️  Considerations:")
    print("   • Requires LLM call (adds latency ~500ms)")
    print("   • Uses more tokens (generates 3-5 query variations)")
    print("   • May retrieve more documents than needed")
    print("   • Best combined with re-ranking for precision")
    print("")
    print("🎯 Configuration tips:")
    print("   • Adjust k per query (default: 3-5)")
    print("   • Use with MMR for diversity: search_type='mmr'")
    print("   • Combine with filters for scoped search")
    print("   • Consider caching for repeated queries")
    print("")
    print("💡 Recommended settings:")
    print("   • Q&A systems: k=3, return top 5 unique docs")
    print("   • Research: k=5, with MMR (lambda=0.6)")
    print("   • Customer support: k=2, with strict filters")

    # Performance comparison
    print_section("Performance Metrics")
    print("Approximate comparison (typical values):\n")
    print("                      Standard    Multi-Query")
    print("                      --------    -----------")
    print("Latency (ms)         50-100      500-800")
    print("Token usage          0           200-500")
    print("Recall               70-80%      85-95%")
    print("Precision            High        Medium-High")
    print("Best for             Speed       Accuracy")

    # Cleanup
    print_section("Cleanup")
    try:
        client.delete_table("multi_query_demo")
        print("   ✓ Cleaned up demo table")
    except Exception as e:
        print(f"   ℹ Cleanup: {e}")


if __name__ == "__main__":
    main()
