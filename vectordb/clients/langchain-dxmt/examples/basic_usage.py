"""Basic usage example for langchain-dxmt.

This example demonstrates how to use DXMT VectorDB with LangChain
for basic vector similarity search.

Prerequisites:
- DXMT VectorDB server running on localhost:9999
- OpenAI API key set in environment

Run:
    python examples/basic_usage.py
"""

from langchain_core.documents import Document
from embedding import default_embeddings, embed_dimension

from langchain_dxmt import DXMTVectorDB


def main() -> None:
    """Run basic usage example."""
    print("=== LangChain DXMT Basic Usage Example ===\n")

    # 1. Initialize embeddings
    print("1. Initializing OpenAI embeddings...")
    embeddings = default_embeddings()

    # 2. Create vector store
    print("2. Creating DXMT vector store...")
    vector_store = DXMTVectorDB(
        table_name="langchain_demo",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
        host="localhost",
        port=9999,
        index_type="hnsw",
        distance_metric="cosine",
    )
    print(f"   ✓ Vector store created: {vector_store._table_name}\n")

    # 3. Add documents
    print("3. Adding documents...")
    documents = [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models.",
            metadata={"source": "langchain_docs", "topic": "introduction"},
        ),
        Document(
            page_content="DXMT VectorDB is a lightweight single-node vector database.",
            metadata={"source": "dxmt_docs", "topic": "introduction"},
        ),
        Document(
            page_content="Vector databases store embeddings for similarity search.",
            metadata={"source": "general", "topic": "concept"},
        ),
        Document(
            page_content="Python is a high-level programming language.",
            metadata={"source": "programming", "topic": "language"},
        ),
    ]

    ids = vector_store.add_documents(documents)
    print(f"   ✓ Added {len(ids)} documents\n")

    # 4. Similarity search
    print("4. Performing similarity search...")
    query = "What is LangChain?"
    results = vector_store.similarity_search(query, k=2)

    print(f"   Query: '{query}'")
    print(f"   Found {len(results)} results:\n")
    for i, doc in enumerate(results, 1):
        print(f"   Result {i}:")
        print(f"     Content: {doc.page_content}")
        print(f"     Metadata: {doc.metadata}\n")

    # 5. Similarity search with scores
    print("5. Similarity search with scores...")
    query = "vector database"
    results_with_scores = vector_store.similarity_search_with_score(query, k=3)

    print(f"   Query: '{query}'")
    print(f"   Found {len(results_with_scores)} results:\n")
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"   Result {i} (score: {score:.4f}):")
        print(f"     Content: {doc.page_content}")
        print(f"     Source: {doc.metadata.get('source', 'N/A')}\n")

    # 6. Search with metadata filter
    print("6. Search with metadata filter...")
    query = "programming"
    results = vector_store.similarity_search(
        query,
        k=5,
        filter={"topic": "language"},
    )

    print(f"   Query: '{query}' with filter {{topic: 'language'}}")
    print(f"   Found {len(results)} results:\n")
    for i, doc in enumerate(results, 1):
        print(f"   Result {i}:")
        print(f"     Content: {doc.page_content}\n")

    # 7. Get documents by ID
    print("7. Retrieving documents by ID...")
    if ids:
        retrieved_docs = vector_store.get_by_ids([ids[0], ids[1]])
        print(f"   Retrieved {len(retrieved_docs)} documents:")
        for doc in retrieved_docs:
            print(f"     - ID: {doc.id}")
            print(f"       Content: {doc.page_content[:50]}...\n")

    # 8. Use as retriever
    print("8. Using as LangChain retriever...")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    query = "database"
    docs = retriever.invoke(query)
    print(f"   Retriever query: '{query}'")
    print(f"   Retrieved {len(docs)} documents\n")

    # 9. Delete documents
    print("9. Cleaning up...")
    if ids:
        success = vector_store.delete(ids=ids[:2])
        print(f"   ✓ Deleted documents: {success}\n")

    print("=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
