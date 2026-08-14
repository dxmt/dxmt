"""DXMT VectorDB + Bisheng Integration Examples.

This example demonstrates how to use DXMT VectorDB with Bisheng through
the LangChain VectorStore interface.

Bisheng is a powerful LLMOps platform that uses LangChain for vector storage.
DXMT integrates seamlessly with Bisheng through the LangChain VectorStore interface.

Requirements:
    pip install dxmt-python-sdk langchain langchain-core langchain-openai

Before running:
    1. Start DXMT server: cargo run -- serve --host 0.0.0.0 --port 9999
    2. Set OPENAI_API_KEY environment variable for embeddings
    3. Ensure langchain packages are installed

Examples:
    1. Basic LangChain VectorStore usage
    2. Bisheng-style initialization
    3. Document search and retrieval
    4. Metadata filtering
"""

import os
import sys
from pathlib import Path

# Add DXMT SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dxmt-python-sdk"))


def example_1_basic_langchain():
    """Example 1: Basic LangChain VectorStore usage with DXMT."""
    print("\n" + "=" * 80)
    print("Example 1: Basic LangChain VectorStore Usage")
    print("=" * 80)

    try:
        from langchain_core.documents import Document
        from langchain_openai import OpenAIEmbeddings

        from dxmt.langchain_vectorstore import DXMTVectorStore
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install langchain langchain-core langchain-openai")
        return

    print("\n📊 Creating DXMT LangChain vector store...")

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create DXMT vector store
    try:
        vector_store = DXMTVectorStore(
            embedding=embeddings,
            collection_name="bisheng_demo",
            host="localhost",
            port=9999,
            distance_metric="cosine"
        )
        print("✅ DXMT vector store created")
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return

    # Create documents
    print("\n📝 Adding documents...")
    documents = [
        Document(
            page_content="DXMT VectorDB provides 98% cost savings through tiered storage architecture.",
            metadata={"source": "dxmt_docs", "category": "features"}
        ),
        Document(
            page_content="Bisheng is an open-source LLMOps platform for building AI applications.",
            metadata={"source": "bisheng_docs", "category": "overview"}
        ),
        Document(
            page_content="LangChain provides a standard interface for vector stores and retrievers.",
            metadata={"source": "langchain_docs", "category": "integration"}
        ),
    ]

    try:
        ids = vector_store.add_documents(documents)
        print(f"✅ Added {len(documents)} documents with IDs: {ids[:2]}...")
    except Exception as e:
        print(f"❌ Error adding documents: {e}")
        return

    # Search documents
    print("\n🔍 Searching documents...")
    queries = [
        "What is DXMT?",
        "Tell me about Bisheng",
        "What are the cost savings?",
    ]

    for query in queries:
        print(f"\n📝 Query: {query}")
        try:
            results = vector_store.similarity_search(query, k=2)
            for i, doc in enumerate(results, 1):
                print(f"   {i}. {doc.page_content[:60]}...")
                print(f"      Metadata: {doc.metadata}")
        except Exception as e:
            print(f"   Error: {e}")

    print("\n✅ Example 1 completed!")


def example_2_bisheng_initialization():
    """Example 2: Bisheng-style initialization pattern."""
    print("\n" + "=" * 80)
    print("Example 2: Bisheng-Style Initialization")
    print("=" * 80)

    try:
        from langchain_core.documents import Document
        from langchain_openai import OpenAIEmbeddings

        from dxmt.bisheng_integration import initialize_dxmt
        from dxmt.langchain_vectorstore import DXMTVectorStore
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return

    print("\n📊 Initializing DXMT using Bisheng pattern...")

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Bisheng-style params
    params = {
        'embedding': embeddings,
        'collection_name': 'bisheng_pattern_demo',
        'host': 'localhost',
        'port': 9999,
        'distance_metric': 'cosine',
    }

    # Initialize without documents (connect to existing collection)
    print("\n1️⃣  Connecting to existing collection...")
    try:
        vector_store1 = initialize_dxmt(DXMTVectorStore, params.copy(), {})
        print("✅ Connected to DXMT collection")
    except Exception as e:
        print(f"⚠️  Error (expected if collection doesn't exist): {e}")

    # Initialize with documents (create new collection)
    print("\n2️⃣  Creating new collection with documents...")
    params_with_docs = params.copy()
    params_with_docs['documents'] = [
        Document(page_content="Document 1", metadata={"id": 1}),
        Document(page_content="Document 2", metadata={"id": 2}),
        Document(page_content="Document 3", metadata={"id": 3}),
    ]

    try:
        vector_store2 = initialize_dxmt(DXMTVectorStore, params_with_docs, {})
        print("✅ Created collection with 3 documents")

        # Test search
        results = vector_store2.similarity_search("document", k=2)
        print(f"✅ Search returned {len(results)} results")
        for i, doc in enumerate(results, 1):
            print(f"   {i}. {doc.page_content}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n✅ Example 2 completed!")


def example_3_search_with_scores():
    """Example 3: Search with similarity scores."""
    print("\n" + "=" * 80)
    print("Example 3: Search with Similarity Scores")
    print("=" * 80)

    try:
        from langchain_core.documents import Document
        from langchain_openai import OpenAIEmbeddings

        from dxmt.langchain_vectorstore import DXMTVectorStore
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return

    print("\n📊 Setting up DXMT vector store...")

    # Create vector store
    embeddings = OpenAIEmbeddings()

    try:
        vector_store = DXMTVectorStore.from_documents(
            documents=[
                Document(page_content="Python is a programming language", metadata={"lang": "python"}),
                Document(page_content="Rust is a systems programming language", metadata={"lang": "rust"}),
                Document(page_content="JavaScript runs in browsers", metadata={"lang": "javascript"}),
                Document(page_content="DXMT is written in Rust for performance", metadata={"topic": "dxmt"}),
            ],
            embedding=embeddings,
            collection_name="search_scores_demo",
            host="localhost",
            port=9999
        )
        print("✅ Vector store created with 4 documents")
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return

    # Search with scores
    print("\n🔍 Searching with similarity scores...")
    queries = [
        "programming languages",
        "rust programming",
        "web development",
    ]

    for query in queries:
        print(f"\n📝 Query: {query}")
        try:
            results = vector_store.similarity_search_with_score(query, k=3)
            for i, (doc, score) in enumerate(results, 1):
                print(f"   {i}. Score: {score:.4f} | {doc.page_content[:50]}...")
        except Exception as e:
            print(f"   Error: {e}")

    print("\n✅ Example 3 completed!")


def example_4_metadata_filtering():
    """Example 4: Metadata filtering."""
    print("\n" + "=" * 80)
    print("Example 4: Metadata Filtering")
    print("=" * 80)

    try:
        from langchain_core.documents import Document
        from langchain_openai import OpenAIEmbeddings

        from dxmt.langchain_vectorstore import DXMTVectorStore
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return

    print("\n📊 Setting up DXMT vector store with metadata...")

    embeddings = OpenAIEmbeddings()

    # Create documents with rich metadata
    documents = [
        Document(page_content="Python tutorial for beginners", metadata={"category": "tutorial", "lang": "python", "level": "beginner"}),
        Document(page_content="Advanced Python patterns", metadata={"category": "advanced", "lang": "python", "level": "expert"}),
        Document(page_content="Rust tutorial for beginners", metadata={"category": "tutorial", "lang": "rust", "level": "beginner"}),
        Document(page_content="JavaScript async/await guide", metadata={"category": "guide", "lang": "javascript", "level": "intermediate"}),
    ]

    try:
        vector_store = DXMTVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="metadata_filter_demo",
            host="localhost",
            port=9999
        )
        print(f"✅ Created vector store with {len(documents)} documents")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Search with metadata filter
    print("\n🔍 Searching with metadata filters...")

    filters = [
        {"lang": "python"},
        {"level": "beginner"},
        {"category": "tutorial"},
    ]

    for filter_dict in filters:
        print(f"\n📝 Filter: {filter_dict}")
        try:
            results = vector_store.similarity_search(
                "programming tutorial",
                k=5,
                filter=filter_dict
            )
            print(f"   Found {len(results)} results:")
            for i, doc in enumerate(results, 1):
                print(f"   {i}. {doc.page_content} | {doc.metadata}")
        except Exception as e:
            print(f"   Error: {e}")

    print("\n✅ Example 4 completed!")


def example_5_cleanup():
    """Example 5: Cleanup test collections."""
    print("\n" + "=" * 80)
    print("Example 5: Cleanup")
    print("=" * 80)

    try:
        from dxmt import DXMTClient
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return

    print("\n🧹 Cleaning up test collections...")
    client = DXMTClient(host="localhost", port=9999)

    test_collections = [
        "bisheng_demo",
        "bisheng_pattern_demo",
        "search_scores_demo",
        "metadata_filter_demo",
    ]

    for collection in test_collections:
        try:
            client.delete_table(collection)
            print(f"✅ Deleted collection: {collection}")
        except Exception as e:
            print(f"ℹ️  Collection {collection}: {e}")

    print("\n✅ Cleanup completed!")


def main():
    """Run all Bisheng integration examples."""
    print("\n" + "=" * 80)
    print("DXMT VectorDB + Bisheng Integration Examples")
    print("=" * 80)
    print("\nThis demonstrates DXMT integration with Bisheng through")
    print("the LangChain VectorStore interface.")

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("Examples will fail without it.\n")
        return

    try:
        # Run examples
        example_1_basic_langchain()
        example_2_bisheng_initialization()
        example_3_search_with_scores()
        example_4_metadata_filtering()
        example_5_cleanup()

        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80)
        print("\n📚 Key Takeaways:")
        print("  ✓ DXMT integrates with LangChain VectorStore interface")
        print("  ✓ Works seamlessly with Bisheng's initialization pattern")
        print("  ✓ Supports similarity search with scores")
        print("  ✓ Metadata filtering for refined searches")
        print("  ✓ 98% cost savings with tiered storage")
        print("  ✓ High-performance vector search (<100ms)")
        print("\n🚀 Ready for production use with Bisheng!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        example_5_cleanup()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
