"""Basic usage example for DXMT Vector Store with LlamaIndex.

This example demonstrates:
1. Creating a vector store and index
2. Adding documents
3. Performing basic queries
4. Using metadata
"""

from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.dxmt import DXMTVectorStore
from embedding import default_embed_model, embed_dimension, has_llm


def main():
    """Run basic usage example."""
    print("=" * 60)
    print("DXMT Vector Store - Basic Usage Example")
    print("=" * 60)

    # 1. Initialize vector store. The table's dimension is the embedding model's, so ask the
    # model rather than hardcode 1536 -- a wrong guess is a dimension error on the first insert.
    embed_model = default_embed_model()
    print("\n1. Initializing DXMT vector store...")
    vector_store = DXMTVectorStore(
        table_name=f"llamaindex_basic_example_{embed_dimension(embed_model)}",
        dimension=embed_dimension(embed_model),
        host="localhost",
        port=9999,
        index_type="hnsw",
        distance_metric="cosine",
    )
    print("✓ Vector store created")

    # Start from an empty table: this example inserts a fixed set of documents, and without
    # this a second run shows every one of them twice -- which reads as a deduplication bug in
    # the database rather than as the example adding them again.
    try:
        vector_store.clear()
    except Exception:
        pass

    # 2. Create the index over that store
    print("\n2. Creating LlamaIndex index...")
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
            text="DXMT VectorDB is a lightweight single-node vector database written in Rust.",
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

    # 4. Query the index. Writing the answer needs an LLM; what DXMT is being asked for is the
    # nodes underneath it, so without a key this prints those and skips the generation.
    print("\n4. Querying the index...")
    questions = ["What is LlamaIndex?", "What is DXMT VectorDB?", "Explain RAG"]
    if has_llm():
        query_engine = index.as_query_engine(similarity_top_k=3)
        for question in questions:
            print(f"\n  Query: {question!r}")
            response = query_engine.query(question)
            print(f"  Response: {response}")
            print(f"  Sources: {len(response.source_nodes)} documents")
    else:
        retriever = index.as_retriever(similarity_top_k=3)
        for question in questions:
            print(f"\n  Query: {question!r}")
            for node in retriever.retrieve(question):
                print(f"    {node.score:.3f}  {node.node.text[:70]}...")
        print("\n  (set OPENAI_API_KEY to see generated answers over these nodes)")

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
