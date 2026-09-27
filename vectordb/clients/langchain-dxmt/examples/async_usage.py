"""Async usage example for langchain-dxmt.

This example demonstrates how to use DXMT VectorDB with async/await
for better performance in async applications.

Prerequisites:
- DXMT VectorDB server running on localhost:9999
- OpenAI API key set in environment

Run:
    python examples/async_usage.py
"""

import asyncio

from langchain_core.documents import Document
from embedding import default_embeddings, embed_dimension

from langchain_dxmt import DXMTVectorDB


async def main() -> None:
    """Run async usage example."""
    print("=== LangChain DXMT Async Usage Example ===\n")

    # 1. Initialize embeddings
    print("1. Initializing OpenAI embeddings...")
    embeddings = default_embeddings()

    # 2. Create vector store
    print("2. Creating DXMT vector store...")
    vector_store = DXMTVectorDB(
        table_name="async_demo",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
        host="localhost",
        port=9999,
        index_type="hnsw",
        distance_metric="cosine",
    )
    print(f"   ✓ Vector store created: {vector_store._table_name}\n")

    # 3. Add documents asynchronously
    print("3. Adding documents asynchronously...")
    documents = [
        Document(
            page_content="Async programming enables concurrent execution of tasks.",
            metadata={"topic": "programming", "difficulty": "intermediate"},
        ),
        Document(
            page_content="Python's asyncio library provides infrastructure for writing async code.",
            metadata={"topic": "python", "difficulty": "intermediate"},
        ),
        Document(
            page_content="Vector databases can handle async operations for better performance.",
            metadata={"topic": "databases", "difficulty": "beginner"},
        ),
        Document(
            page_content="LangChain supports both sync and async interfaces.",
            metadata={"topic": "langchain", "difficulty": "beginner"},
        ),
        Document(
            page_content="Concurrent requests improve throughput in async systems.",
            metadata={"topic": "programming", "difficulty": "advanced"},
        ),
    ]

    # Add documents asynchronously
    ids = await vector_store.aadd_documents(documents)
    print(f"   ✓ Added {len(ids)} documents asynchronously\n")

    # 4. Async similarity search
    print("4. Performing async similarity search...")
    query = "How does async programming work?"
    results = await vector_store.asimilarity_search(query, k=3)

    print(f"   Query: '{query}'")
    print(f"   Found {len(results)} results:\n")
    for i, doc in enumerate(results, 1):
        print(f"   Result {i}:")
        print(f"     Content: {doc.page_content}")
        print(f"     Topic: {doc.metadata.get('topic', 'N/A')}")
        print(f"     Difficulty: {doc.metadata.get('difficulty', 'N/A')}\n")

    # 5. Async search with scores
    print("5. Async similarity search with scores...")
    query = "vector database performance"
    results_with_scores = await vector_store.asimilarity_search_with_score(
        query,
        k=3,
    )

    print(f"   Query: '{query}'")
    print(f"   Found {len(results_with_scores)} results:\n")
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"   Result {i} (score: {score:.4f}):")
        print(f"     Content: {doc.page_content[:60]}...")
        print(f"     Topic: {doc.metadata.get('topic', 'N/A')}\n")

    # 6. Async batch operations
    print("6. Demonstrating concurrent async operations...")

    # Create multiple queries to run concurrently
    queries = [
        "What is Python asyncio?",
        "How do vector databases work?",
        "LangChain framework features",
    ]

    # Run searches concurrently
    print(f"   Running {len(queries)} searches concurrently...")
    search_tasks = [
        vector_store.asimilarity_search(query, k=2)
        for query in queries
    ]

    # Wait for all searches to complete
    all_results = await asyncio.gather(*search_tasks)

    print(f"   ✓ Completed {len(all_results)} concurrent searches\n")
    for i, (query, results) in enumerate(zip(queries, all_results), 1):
        print(f"   Query {i}: '{query}'")
        print(f"   Top result: {results[0].page_content[:50]}...\n")

    # 7. Async retrieval by IDs
    print("7. Async retrieval by IDs...")
    if ids:
        retrieved_docs = await vector_store.aget_by_ids([ids[0], ids[1]])
        print(f"   Retrieved {len(retrieved_docs)} documents:")
        for doc in retrieved_docs:
            print(f"     - ID: {doc.id}")
            print(f"       Content: {doc.page_content[:50]}...\n")

    # 8. Async deletion
    print("8. Cleaning up asynchronously...")
    if ids:
        success = await vector_store.adelete(ids=ids)
        print(f"   ✓ Deleted documents: {success}\n")

    print("=== Example completed successfully! ===")
    print("\n💡 Key Benefits of Async:")
    print("   - Non-blocking I/O operations")
    print("   - Concurrent request execution")
    print("   - Better resource utilization")
    print("   - Improved throughput for batch operations")
    print("   - Ideal for web applications and APIs\n")


async def benchmark_comparison() -> None:
    """Compare sync vs async performance."""
    import time

    print("=== Sync vs Async Performance Comparison ===\n")

    embeddings = default_embeddings()

    # Create two stores
    sync_store = DXMTVectorDB(
        table_name="sync_benchmark",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
    )

    async_store = DXMTVectorDB(
        table_name="async_benchmark",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
    )

    # Add test documents
    test_docs = [
        f"Test document {i} with some sample content"
        for i in range(20)
    ]

    await async_store.aadd_texts(test_docs)
    sync_store.add_texts(test_docs)

    # Benchmark sync searches
    print("1. Running 10 sync searches...")
    start_time = time.time()
    for i in range(10):
        sync_store.similarity_search(f"query {i}", k=3)
    sync_duration = time.time() - start_time
    print(f"   Sync time: {sync_duration:.3f}s\n")

    # Benchmark async searches (sequential)
    print("2. Running 10 async searches (sequential)...")
    start_time = time.time()
    for i in range(10):
        await async_store.asimilarity_search(f"query {i}", k=3)
    async_sequential_duration = time.time() - start_time
    print(f"   Async sequential time: {async_sequential_duration:.3f}s\n")

    # Benchmark async searches (concurrent)
    print("3. Running 10 async searches (concurrent)...")
    start_time = time.time()
    tasks = [
        async_store.asimilarity_search(f"query {i}", k=3)
        for i in range(10)
    ]
    await asyncio.gather(*tasks)
    async_concurrent_duration = time.time() - start_time
    print(f"   Async concurrent time: {async_concurrent_duration:.3f}s\n")

    # Calculate speedup
    speedup = sync_duration / async_concurrent_duration
    print("=== Results ===")
    print(f"Speedup (sync vs async concurrent): {speedup:.2f}x\n")

    # Cleanup
    await async_store.adelete(ids=list(range(20)))
    sync_store.delete(ids=list(range(20)))


if __name__ == "__main__":
    # Run main example
    asyncio.run(main())

    # Uncomment to run benchmark comparison
    # asyncio.run(benchmark_comparison())
