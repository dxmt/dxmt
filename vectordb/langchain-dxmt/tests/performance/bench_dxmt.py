"""Performance benchmarks for DXMT VectorStore.

Run benchmarks with:
    python tests/performance/bench_dxmt.py

Prerequisites:
- DXMT server running on localhost:9999
- OpenAI API key set in environment
"""

import os
import time
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from langchain_dxmt import DXMTVectorDB


# Configuration
DXMT_HOST = os.getenv("DXMT_HOST", "localhost")
DXMT_PORT = int(os.getenv("DXMT_PORT", "9999"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set - using dummy embeddings")


class BenchmarkResults:
    """Store and display benchmark results."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def add(
        self,
        name: str,
        duration: float,
        operations: int,
        **kwargs: Any,
    ) -> None:
        """Add benchmark result."""
        throughput = operations / duration if duration > 0 else 0
        latency = (duration * 1000) / operations if operations > 0 else 0

        self.results.append({
            "name": name,
            "duration": duration,
            "operations": operations,
            "throughput": throughput,
            "latency_ms": latency,
            **kwargs,
        })

    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)

        for result in self.results:
            print(f"\n{result['name']}")
            print("-" * 80)
            print(f"  Duration:    {result['duration']:.3f}s")
            print(f"  Operations:  {result['operations']}")
            print(f"  Throughput:  {result['throughput']:.1f} ops/sec")
            print(f"  Latency:     {result['latency_ms']:.2f}ms per operation")

            # Print additional metrics
            for key, value in result.items():
                if key not in ["name", "duration", "operations", "throughput", "latency_ms"]:
                    print(f"  {key.replace('_', ' ').title()}: {value}")

        print("\n" + "=" * 80)


def benchmark_insert_throughput(
    results: BenchmarkResults,
    num_docs: int = 1000,
) -> None:
    """Benchmark document insertion throughput."""
    print(f"\n[1/7] Benchmarking insert throughput ({num_docs} docs)...")

    embeddings = OpenAIEmbeddings()

    # Create vector store
    store = DXMTVectorDB(
        table_name="bench_insert",
        embedding=embeddings,
        dimension=1536,
        host=DXMT_HOST,
        port=DXMT_PORT,
        insert_mode="ultimate",
    )

    # Generate test documents
    documents = [
        Document(
            page_content=f"Test document {i} with some sample content for benchmarking",
            metadata={"index": i, "category": f"cat_{i % 10}"},
        )
        for i in range(num_docs)
    ]

    # Benchmark insertion
    start_time = time.time()
    ids = store.add_documents(documents)
    duration = time.time() - start_time

    results.add(
        name="Insert Throughput",
        duration=duration,
        operations=num_docs,
        mode="ultimate",
        dimension=1536,
    )

    # Cleanup
    store.delete(ids=ids)


def benchmark_search_latency(
    results: BenchmarkResults,
    num_docs: int = 1000,
    num_queries: int = 100,
    k: int = 10,
) -> None:
    """Benchmark search latency."""
    print(f"\n[2/7] Benchmarking search latency ({num_queries} queries, k={k})...")

    embeddings = OpenAIEmbeddings()

    # Create and populate vector store
    store = DXMTVectorDB(
        table_name="bench_search",
        embedding=embeddings,
        dimension=1536,
        host=DXMT_HOST,
        port=DXMT_PORT,
    )

    documents = [
        Document(page_content=f"Document {i}")
        for i in range(num_docs)
    ]
    ids = store.add_documents(documents)

    # Benchmark search
    latencies = []
    for i in range(num_queries):
        start_time = time.time()
        results_docs = store.similarity_search(f"query {i}", k=k)
        latency = time.time() - start_time
        latencies.append(latency)

    # Calculate statistics
    latencies_ms = [l * 1000 for l in latencies]
    total_duration = sum(latencies)

    results.add(
        name="Search Latency",
        duration=total_duration,
        operations=num_queries,
        k=k,
        p50_ms=np.percentile(latencies_ms, 50),
        p95_ms=np.percentile(latencies_ms, 95),
        p99_ms=np.percentile(latencies_ms, 99),
    )

    # Cleanup
    store.delete(ids=ids)


def benchmark_batch_sizes(
    results: BenchmarkResults,
) -> None:
    """Benchmark different batch sizes."""
    print("\n[3/7] Benchmarking different batch sizes...")

    embeddings = OpenAIEmbeddings()
    batch_sizes = [10, 50, 100, 500, 1000]

    for batch_size in batch_sizes:
        store = DXMTVectorDB(
            table_name=f"bench_batch_{batch_size}",
            embedding=embeddings,
            dimension=1536,
            host=DXMT_HOST,
            port=DXMT_PORT,
        )

        documents = [
            Document(page_content=f"Doc {i}")
            for i in range(batch_size)
        ]

        start_time = time.time()
        ids = store.add_documents(documents)
        duration = time.time() - start_time

        results.add(
            name=f"Batch Insert (size={batch_size})",
            duration=duration,
            operations=batch_size,
        )

        # Cleanup
        store.delete(ids=ids)


def benchmark_index_types(
    results: BenchmarkResults,
    num_docs: int = 500,
) -> None:
    """Benchmark different index types."""
    print(f"\n[4/7] Benchmarking different index types ({num_docs} docs)...")

    embeddings = OpenAIEmbeddings()
    index_types = ["hnsw", "ivf", "brute_force"]

    for index_type in index_types:
        # Create store
        store = DXMTVectorDB(
            table_name=f"bench_index_{index_type}",
            embedding=embeddings,
            dimension=1536,
            host=DXMT_HOST,
            port=DXMT_PORT,
            index_type=index_type,
        )

        # Add documents
        documents = [
            Document(page_content=f"Document {i}")
            for i in range(num_docs)
        ]

        insert_start = time.time()
        ids = store.add_documents(documents)
        insert_duration = time.time() - insert_start

        # Search benchmark
        num_queries = 50
        search_start = time.time()
        for i in range(num_queries):
            store.similarity_search(f"query {i}", k=10)
        search_duration = time.time() - search_start

        results.add(
            name=f"Index Type: {index_type}",
            duration=insert_duration + search_duration,
            operations=num_docs + num_queries,
            index_type=index_type,
            insert_throughput=num_docs / insert_duration,
            search_latency_ms=(search_duration * 1000) / num_queries,
        )

        # Cleanup
        store.delete(ids=ids)


def benchmark_distance_metrics(
    results: BenchmarkResults,
    num_docs: int = 500,
) -> None:
    """Benchmark different distance metrics."""
    print(f"\n[5/7] Benchmarking different distance metrics ({num_docs} docs)...")

    embeddings = OpenAIEmbeddings()
    metrics = ["cosine", "euclidean", "dot_product"]

    for metric in metrics:
        store = DXMTVectorDB(
            table_name=f"bench_metric_{metric}",
            embedding=embeddings,
            dimension=1536,
            host=DXMT_HOST,
            port=DXMT_PORT,
            distance_metric=metric,
        )

        # Add documents
        documents = [
            Document(page_content=f"Document {i}")
            for i in range(num_docs)
        ]
        ids = store.add_documents(documents)

        # Search benchmark
        num_queries = 50
        start_time = time.time()
        for i in range(num_queries):
            store.similarity_search(f"query {i}", k=10)
        duration = time.time() - start_time

        results.add(
            name=f"Distance Metric: {metric}",
            duration=duration,
            operations=num_queries,
            metric=metric,
        )

        # Cleanup
        store.delete(ids=ids)


def benchmark_hybrid_search(
    results: BenchmarkResults,
    num_docs: int = 500,
) -> None:
    """Benchmark hybrid search vs pure vector search."""
    print(f"\n[6/7] Benchmarking hybrid search ({num_docs} docs)...")

    embeddings = OpenAIEmbeddings()

    # Create store with full-text search
    store = DXMTVectorDB(
        table_name="bench_hybrid",
        embedding=embeddings,
        dimension=1536,
        host=DXMT_HOST,
        port=DXMT_PORT,
        enable_full_text_search=True,
    )

    # Add documents
    documents = [
        Document(
            page_content=f"Document {i} about machine learning and artificial intelligence",
            metadata={"index": i},
        )
        for i in range(num_docs)
    ]
    ids = store.add_documents(documents)

    # Benchmark pure vector search
    num_queries = 50
    vector_start = time.time()
    for i in range(num_queries):
        store.similarity_search("machine learning", k=10)
    vector_duration = time.time() - vector_start

    results.add(
        name="Pure Vector Search",
        duration=vector_duration,
        operations=num_queries,
        search_type="vector",
    )

    # Benchmark hybrid search
    hybrid_start = time.time()
    for i in range(num_queries):
        store.hybrid_search("machine learning", k=10, vector_weight=0.7)
    hybrid_duration = time.time() - hybrid_start

    results.add(
        name="Hybrid Search (70/30)",
        duration=hybrid_duration,
        operations=num_queries,
        search_type="hybrid",
        vector_weight=0.7,
    )

    # Cleanup
    store.delete(ids=ids)


def benchmark_concurrent_operations(
    results: BenchmarkResults,
) -> None:
    """Benchmark concurrent operations."""
    print("\n[7/7] Benchmarking concurrent operations...")

    import asyncio

    embeddings = OpenAIEmbeddings()

    async def concurrent_benchmark():
        store = DXMTVectorDB(
            table_name="bench_concurrent",
            embedding=embeddings,
            dimension=1536,
            host=DXMT_HOST,
            port=DXMT_PORT,
        )

        # Add initial documents
        documents = [
            Document(page_content=f"Document {i}")
            for i in range(100)
        ]
        await store.aadd_documents(documents)

        # Run concurrent searches
        num_concurrent = 20
        start_time = time.time()

        tasks = [
            store.asimilarity_search(f"query {i}", k=10)
            for i in range(num_concurrent)
        ]

        await asyncio.gather(*tasks)
        duration = time.time() - start_time

        return duration, num_concurrent

    # Run async benchmark
    duration, num_concurrent = asyncio.run(concurrent_benchmark())

    results.add(
        name="Concurrent Searches",
        duration=duration,
        operations=num_concurrent,
        concurrency=num_concurrent,
    )


def run_benchmarks() -> None:
    """Run all benchmarks."""
    print("=" * 80)
    print("DXMT VectorStore Performance Benchmarks")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Host: {DXMT_HOST}")
    print(f"  Port: {DXMT_PORT}")
    print(f"  Embedding: OpenAI (1536-dim)")

    results = BenchmarkResults()

    try:
        # Run all benchmarks
        benchmark_insert_throughput(results, num_docs=1000)
        benchmark_search_latency(results, num_docs=1000, num_queries=100, k=10)
        benchmark_batch_sizes(results)
        benchmark_index_types(results, num_docs=500)
        benchmark_distance_metrics(results, num_docs=500)
        benchmark_hybrid_search(results, num_docs=500)
        benchmark_concurrent_operations(results)

        # Print summary
        results.print_summary()

        print("\n✅ All benchmarks completed successfully!")

    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_benchmarks()
