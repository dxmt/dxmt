"""Basic usage example for DXMT Vector Store with LlamaIndex.

This example demonstrates:
1. Creating a vector store and index
2. Adding documents
3. Performing basic queries
4. Using metadata
"""

from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.dxmt import DXMTVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding


def main():
    """Run basic usage example."""
    print("=" * 60)
    print("DXMT Vector Store - Basic Usage Example")
    print("=" * 60)

    # 1. Initialize vector store
    print("\n1. Initializing DXMT vector store...")
    vector_store = DXMTVectorStore(
        table_name="llamaindex_basic_example",
        dimension=1536,  # OpenAI ada-002 embedding dimension
        host="localhost",
        port=9999,
        index_type="hnsw",
        distance_metric="cosine",
    )
    print("✓ Vector store created")

    # 2. Create index with OpenAI embeddings
    print("\n2. Creating LlamaIndex index...")
    embed_model = OpenAIEmbedding()
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    print("✓ Index created")

    # 3. Prepare sample documents
    print("\n3. Adding documents...")
    documents = [
        Document(
            text="LlamaIndex is a data framework for LLM applications. It provides tools to ingest, structure, and access data.",
            metadata={"category": "documentation", "source": "docs", "topic": "overview"}
        ),
        Document(
            text="Vector databases store embeddings and enable semantic search. They are essential for RAG applications.",
            metadata={"category": "documentation", "source": "docs", "topic": "vector-db"}
        ),
        Document(
            text="DXMT VectorDB is a lightweight, horizontally scalable vector database with object storage native architecture.",
            metadata={"category": "product", "source": "website", "topic": "dxmt"}
        ),
        Document(
            text="Embeddings are numerical representations of text that capture semantic meaning. OpenAI provides embedding models.",
            metadata={"category": "documentation", "source": "docs", "topic": "embeddings"}
        ),
        Document(
            text="RAG (Retrieval Augmented Generation) combines information retrieval with language generation for better results.",
            metadata={"category": "documentation", "source": "docs", "topic": "rag"}
        ),
    ]

    # Insert documents
    for i, doc in enumerate(documents, 1):
        index.insert(doc)
        print(f"  ✓ Document {i}/{len(documents)} added")

    # 4. Query the index
    print("\n4. Querying the index...")
    query_engine = index.as_query_engine(similarity_top_k=3)

    # Query 1: What is LlamaIndex?
    print("\n  Query 1: 'What is LlamaIndex?'")
    response = query_engine.query("What is LlamaIndex?")
    print(f"  Response: {response}")
    print(f"  Sources: {len(response.source_nodes)} documents")

    # Query 2: What is DXMT?
    print("\n  Query 2: 'What is DXMT VectorDB?'")
    response = query_engine.query("What is DXMT VectorDB?")
    print(f"  Response: {response}")
    print(f"  Sources: {len(response.source_nodes)} documents")

    # Query 3: Explain RAG
    print("\n  Query 3: 'Explain RAG'")
    response = query_engine.query("Explain RAG")
    print(f"  Response: {response}")
    print(f"  Sources: {len(response.source_nodes)} documents")

    # 5. Retrieve specific documents
    print("\n5. Using retriever...")
    retriever = index.as_retriever(similarity_top_k=2)
    nodes = retriever.retrieve("vector database")

    print(f"  Retrieved {len(nodes)} nodes for 'vector database':")
    for i, node in enumerate(nodes, 1):
        print(f"    {i}. Score: {node.score:.4f}")
        print(f"       Text: {node.text[:100]}...")
        print(f"       Metadata: {node.metadata}")

    # 6. Clean up (optional)
    print("\n6. Cleaning up...")
    # Uncomment to delete the table after the example
    # vector_store.clear()
    # print("  ✓ Table deleted")
    print("  (Skipped - table preserved for inspection)")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
