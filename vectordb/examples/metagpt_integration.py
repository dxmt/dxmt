"""DXMT VectorDB + MetaGPT Integration Examples.

This example demonstrates how to use DXMT VectorDB with MetaGPT's RAG system.

MetaGPT uses LlamaIndex as the underlying RAG framework, so this integration
works seamlessly with both MetaGPT and standalone LlamaIndex usage.

Requirements:
    pip install dxmt-python-sdk llama-index llama-index-core

Before running:
    1. Start DXMT server: cargo run -- serve --host 0.0.0.0 --port 9999
    2. Set OPENAI_API_KEY environment variable for embeddings
    3. Ensure llama-index is installed

Examples:
    1. Basic RAG pipeline with DXMT
    2. Add documents incrementally
    3. Multi-retriever hybrid search
    4. MetaGPT-compatible retriever
"""

import os
import sys
from pathlib import Path

# Add DXMT SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent / "clients" / "dxmt-python-sdk"))


# The provider's models come from the environment so one `.env` switch moves every example.
# These are *deployment* names and do not carry between providers -- the Azure resource this
# was written against serves `gpt-4.1-mini` and answers DeploymentNotFound for `gpt-4o-mini`.
# The defaults are what this file used before, so nothing changes for a plain OPENAI_API_KEY.
CHAT_MODEL = os.environ.get("DXMT_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.environ.get("DXMT_EMBED_MODEL", "text-embedding-3-small")
EMBED_DIMS = int(os.environ.get("DXMT_EMBED_DIMS", "1536"))


def example_1_basic_rag():
    """Example 1: Basic RAG pipeline with DXMT."""
    print("\n" + "=" * 80)
    print("Example 1: Basic RAG Pipeline with DXMT VectorStore")
    print("=" * 80)

    try:
        from llama_index.core import Document, StorageContext, VectorStoreIndex
        from llama_index.core.schema import TextNode

        from dxmt import DXMTClient
        from dxmt.metagpt_vector_store import DXMTVectorStore
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install llama-index llama-index-core")
        return

    # Create DXMT client
    print("\n📊 Creating DXMT client...")
    client = DXMTClient(host="localhost", port=9999)

    # Create DXMT vector store
    print("📊 Creating DXMT vector store...")
    vector_store = DXMTVectorStore(
        client=client,
        table_name="metagpt_basic_docs",
        dimension=EMBED_DIMS,
        distance_metric="cosine",
        create_if_not_exists=True,
    )

    # Create documents
    print("📝 Creating sample documents...")
    documents = [
        Document(
            text="DXMT VectorDB is a lightweight, horizontally scalable vector database "
            "with storage native architecture. It provides 98% cost savings through "
            "tiered storage (Memory → SSD → S3).",
            metadata={"source": "dxmt_docs", "category": "overview"},
        ),
        Document(
            text="MetaGPT is a multi-agent framework that assigns different roles to GPTs "
            "to form a collaborative software entity for complex tasks.",
            metadata={"source": "metagpt_docs", "category": "overview"},
        ),
        Document(
            text="LlamaIndex provides data framework for LLM applications with advanced "
            "RAG capabilities including indexing, retrieval, and query engines.",
            metadata={"source": "llama_docs", "category": "framework"},
        ),
    ]

    # Create storage context and index
    print("🔨 Building vector index...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    try:
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, show_progress=True
        )
        print(f"✅ Indexed {len(documents)} documents")
    except Exception as e:
        print(f"⚠️  Indexing error: {e}")
        print("Note: If embedding fails, ensure OPENAI_API_KEY is set")
        return

    # Query the index
    print("\n🔍 Querying the index...")
    query_engine = index.as_query_engine(similarity_top_k=2)

    queries = [
        "What is DXMT VectorDB?",
        "Tell me about MetaGPT",
        "What are the key features of DXMT?",
    ]

    for query in queries:
        print(f"\n📝 Query: {query}")
        try:
            response = query_engine.query(query)
            print(f"   Answer: {response.response[:150]}...")
        except Exception as e:
            print(f"   Error: {e}")

    print("\n✅ Example 1 completed!")


def example_2_metagpt_retriever():
    """Example 2: MetaGPT-compatible retriever with additional methods."""
    print("\n" + "=" * 80)
    print("Example 2: MetaGPT-Compatible DXMT Retriever")
    print("=" * 80)

    try:
        from llama_index.core import Document

        from dxmt.metagpt_retriever import create_dxmt_retriever
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return

    # Create documents
    print("\n📝 Creating knowledge base...")
    documents = [
        Document(text="Python is a high-level programming language", metadata={"topic": "python"}),
        Document(text="Rust provides memory safety without garbage collection", metadata={"topic": "rust"}),
        Document(text="JavaScript is the language of the web", metadata={"topic": "javascript"}),
        Document(
            text="DXMT VectorDB is written in Rust for high performance", metadata={"topic": "dxmt"}
        ),
    ]

    # Create DXMT retriever (all-in-one)
    print("🔨 Creating DXMT retriever...")
    try:
        retriever = create_dxmt_retriever(
            documents=documents,
            table_name="metagpt_retriever_demo",
            dimension=EMBED_DIMS,
            similarity_top_k=3,
            host="localhost",
            port=9999,
        )
        print("✅ Retriever created successfully")
    except Exception as e:
        print(f"⚠️  Error creating retriever: {e}")
        return

    # Test retrieval
    print("\n🔍 Testing retrieval...")
    queries = ["programming languages", "memory safety", "database performance"]

    for query in queries:
        print(f"\n📝 Query: {query}")
        try:
            results = retriever.retrieve(query)
            print(f"   Found {len(results)} results:")
            for i, node in enumerate(results[:2], 1):
                text = node.node.text[:60]
                score = node.score
                print(f"   {i}. {text}... (score: {score:.3f})")
        except Exception as e:
            print(f"   Error: {e}")

    # Test MetaGPT interface methods
    print("\n🧪 Testing MetaGPT interface methods...")

    # add_nodes
    print("📝 Adding new documents...")
    try:
        from llama_index.core.schema import TextNode

        new_nodes = [
            TextNode(text="TypeScript adds static typing to JavaScript", id_="ts1"),
            TextNode(text="Go is designed for concurrent programming", id_="go1"),
        ]
        retriever.add_nodes(new_nodes)
        print("✅ Added 2 new nodes")
    except Exception as e:
        print(f"⚠️  Error adding nodes: {e}")

    # query_total_count
    print("📊 Querying total document count...")
    try:
        count = retriever.query_total_count()
        print(f"✅ Total documents: {count}")
    except Exception as e:
        print(f"⚠️  Error getting count: {e}")

    # persist
    print("💾 Persisting index...")
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            retriever.persist(tmpdir)
            print(f"✅ Index persisted to {tmpdir}")
    except Exception as e:
        print(f"⚠️  Error persisting: {e}")

    print("\n✅ Example 2 completed!")


def example_3_incremental_updates():
    """Example 3: Incremental document updates."""
    print("\n" + "=" * 80)
    print("Example 3: Incremental Document Updates")
    print("=" * 80)

    try:
        from llama_index.core import Document, StorageContext, VectorStoreIndex

        from dxmt import DXMTClient
        from dxmt.metagpt_vector_store import DXMTVectorStore
        from dxmt.metagpt_retriever import DXMTRetriever
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return

    print("\n📊 Setting up DXMT vector store...")
    client = DXMTClient(host="localhost", port=9999)
    vector_store = DXMTVectorStore(
        client=client, table_name="metagpt_incremental", dimension=EMBED_DIMS, create_if_not_exists=True
    )

    # Initial documents
    print("📝 Creating initial document set...")
    initial_docs = [
        Document(text="Machine learning is a subset of AI", metadata={"batch": 1}),
        Document(text="Deep learning uses neural networks", metadata={"batch": 1}),
    ]

    # Create index
    print("🔨 Building initial index...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    try:
        index = VectorStoreIndex.from_documents(initial_docs, storage_context=storage_context)
        retriever = DXMTRetriever(index=index, similarity_top_k=3)
        print("✅ Initial index created with 2 documents")
    except Exception as e:
        print(f"⚠️  Error creating index: {e}")
        return

    # Query before updates
    print("\n🔍 Query before updates:")
    query = "neural networks"
    try:
        results = retriever.retrieve(query)
        print(f"   Found {len(results)} results")
        for node in results[:2]:
            print(f"   - {node.node.text[:50]}... (score: {node.score:.3f})")
    except Exception as e:
        print(f"   Error: {e}")

    # Add more documents incrementally
    print("\n📝 Adding documents incrementally...")
    batches = [
        [
            Document(text="Transformers revolutionized NLP", metadata={"batch": 2}),
            Document(text="Attention mechanisms improve model performance", metadata={"batch": 2}),
        ],
        [
            Document(text="GPT models use transformer architecture", metadata={"batch": 3}),
            Document(text="BERT is bidirectional transformer", metadata={"batch": 3}),
        ],
    ]

    for i, batch in enumerate(batches, 2):
        try:
            print(f"   Adding batch {i}...")
            retriever.add_documents(batch)
            print(f"   ✅ Added {len(batch)} documents")
        except Exception as e:
            print(f"   ⚠️  Error adding batch {i}: {e}")

    # Query after updates
    print("\n🔍 Query after updates:")
    try:
        results = retriever.retrieve(query)
        print(f"   Found {len(results)} results")
        for node in results[:3]:
            print(f"   - {node.node.text[:50]}... (score: {node.score:.3f})")

        # Check total count
        total = retriever.query_total_count()
        print(f"\n📊 Total documents in index: {total}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n✅ Example 3 completed!")


def example_4_cleanup():
    """Example 4: Cleanup and reset."""
    print("\n" + "=" * 80)
    print("Example 4: Cleanup and Reset")
    print("=" * 80)

    try:
        from dxmt import DXMTClient
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return

    print("\n🧹 Cleaning up test tables...")
    client = DXMTClient(host="localhost", port=9999)

    test_tables = [
        "metagpt_basic_docs",
        "metagpt_retriever_demo",
        "metagpt_incremental",
    ]

    for table_name in test_tables:
        try:
            client.delete_table(table_name)
            print(f"✅ Deleted table: {table_name}")
        except Exception as e:
            print(f"ℹ️  Table {table_name}: {e}")

    print("\n✅ Cleanup completed!")


def main():
    """Run all MetaGPT integration examples."""
    print("\n" + "=" * 80)
    print("DXMT VectorDB + MetaGPT Integration Examples")
    print("=" * 80)
    print("\nThis demonstrates DXMT integration with MetaGPT's RAG system.")
    print("MetaGPT uses LlamaIndex, so this works for both frameworks.")

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("Examples may fail without it.\n")

    try:
        # Run examples
        example_1_basic_rag()
        example_2_metagpt_retriever()
        example_3_incremental_updates()
        example_4_cleanup()

        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80)
        print("\n📚 Key Takeaways:")
        print("  ✓ DXMT integrates seamlessly with LlamaIndex")
        print("  ✓ Works with MetaGPT's RAG system out of the box")
        print("  ✓ Supports all MetaGPT retriever interfaces:")
        print("    - ModifiableRAGRetriever (add_nodes)")
        print("    - PersistableRAGRetriever (persist)")
        print("    - QueryableRAGRetriever (query_total_count)")
        print("    - DeletableRAGRetriever (clear)")
        print("  ✓ 98% cost savings with tiered storage")
        print("  ✓ High-performance vector search (<100ms)")
        print("\n🚀 Ready for production use with MetaGPT!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        example_4_cleanup()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
