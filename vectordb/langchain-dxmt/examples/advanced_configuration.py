"""Advanced configuration example for langchain-dxmt.

This example demonstrates advanced DXMT configuration options including:
- Different index types (HNSW, IVF, IVF-PQ, LSH, Brute Force)
- Different distance metrics (Cosine, Euclidean, Manhattan, Dot Product)
- Different insert modes (Standard, LockFree, Ultimate, etc.)
- Performance tuning options

Prerequisites:
- DXMT VectorDB server running on localhost:9999
- OpenAI API key set in environment

Run:
    python examples/advanced_configuration.py
"""

import time
from typing import Any

from langchain_core.documents import Document
from embedding import default_embeddings, embed_dimension

from langchain_dxmt import DXMTVectorDB


def example_index_types() -> None:
    """Demonstrate different index types."""
    print("=== Index Types Comparison ===\n")

    embeddings = default_embeddings()

    # Test documents
    documents = [
        Document(page_content=f"Test document {i}")
        for i in range(50)
    ]

    index_configs = [
        {
            "name": "HNSW",
            "type": "hnsw",
            "description": "Hierarchical Navigable Small World - Fast and accurate",
            "use_case": "General purpose, best for most applications",
        },
        {
            "name": "IVF",
            "type": "ivf",
            "description": "Inverted File Index - Memory efficient",
            "use_case": "Large datasets with memory constraints",
        },
        {
            "name": "IVF-PQ",
            "type": "ivf_pq",
            "description": "IVF with Product Quantization - Very memory efficient",
            "use_case": "Very large datasets, can trade accuracy for memory",
        },
        {
            "name": "LSH",
            "type": "lsh",
            "description": "Locality Sensitive Hashing - Fast approximate search",
            "use_case": "Extremely fast search, lower accuracy acceptable",
        },
        {
            "name": "Brute Force",
            "type": "brute_force",
            "description": "Exact search - 100% accurate but slow",
            "use_case": "Small datasets, maximum accuracy required",
        },
    ]

    for config in index_configs:
        print(f"Testing: {config['name']}")
        print(f"  Description: {config['description']}")
        print(f"  Use Case: {config['use_case']}")

        try:
            # Create vector store with specific index type
            store = DXMTVectorDB(
                table_name=f"index_test_{config['type']}",
                embedding=embeddings,
                dimension=embed_dimension(embeddings),
                index_type=config["type"],
                distance_metric="cosine",
            )

            # Measure insert performance
            start_time = time.time()
            ids = store.add_documents(documents)
            insert_time = time.time() - start_time

            # Measure search performance
            start_time = time.time()
            results = store.similarity_search("test query", k=5)
            search_time = time.time() - start_time

            print(f"  ✓ Insert time: {insert_time:.3f}s")
            print(f"  ✓ Search time: {search_time*1000:.1f}ms")
            print(f"  ✓ Results: {len(results)}\n")

            # Cleanup
            store.delete(ids=ids)

        except Exception as e:
            print(f"  ✗ Error: {e}\n")


def example_distance_metrics() -> None:
    """Demonstrate different distance metrics."""
    print("=== Distance Metrics Comparison ===\n")

    embeddings = default_embeddings()

    # Test documents
    documents = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"topic": "AI"},
        ),
        Document(
            page_content="Deep learning uses neural networks for pattern recognition.",
            metadata={"topic": "AI"},
        ),
        Document(
            page_content="Python is a popular programming language.",
            metadata={"topic": "Programming"},
        ),
    ]

    metrics = [
        {
            "name": "Cosine",
            "type": "cosine",
            "description": "Measures angle between vectors",
            "range": "[0, 1] (0=identical, 1=opposite)",
            "use_case": "Text embeddings, normalized vectors",
        },
        {
            "name": "Euclidean",
            "type": "euclidean",
            "description": "Straight-line distance between vectors",
            "range": "[0, ∞) (0=identical)",
            "use_case": "Image embeddings, dense vectors",
        },
        {
            "name": "Manhattan",
            "type": "manhattan",
            "description": "Sum of absolute differences",
            "range": "[0, ∞) (0=identical)",
            "use_case": "Sparse vectors, categorical features",
        },
        {
            "name": "Dot Product",
            # `dot`, not `dot_product`: the server's name for it, and the one it accepts. The
            # other spelling was in this list and answered INVALID_DISTANCE_METRIC every run.
            "type": "dot",
            "description": "Inner product of vectors",
            "range": "(-∞, ∞) (higher=more similar)",
            "use_case": "Normalized vectors, collaborative filtering",
        },
    ]

    query = "What is machine learning?"

    for metric in metrics:
        print(f"Metric: {metric['name']}")
        print(f"  Description: {metric['description']}")
        print(f"  Score Range: {metric['range']}")
        print(f"  Use Case: {metric['use_case']}")

        try:
            # Create vector store with specific distance metric
            store = DXMTVectorDB(
                table_name=f"metric_test_{metric['type']}",
                embedding=embeddings,
                dimension=embed_dimension(embeddings),
                index_type="hnsw",
                distance_metric=metric["type"],
            )

            # Add documents
            ids = store.add_documents(documents)

            # Search with scores
            results = store.similarity_search_with_score(query, k=3)

            print(f"  Query: '{query}'")
            print(f"  Results:")
            for i, (doc, score) in enumerate(results, 1):
                print(f"    [{i}] Score: {score:.4f} - {doc.page_content[:50]}...")

            print()

            # Cleanup
            store.delete(ids=ids)

        except Exception as e:
            print(f"  ✗ Error: {e}\n")


def example_production_config() -> None:
    """Demonstrate production-ready configuration."""
    print("=== Production Configuration Example ===\n")

    embeddings = default_embeddings()

    # High-performance configuration
    print("1. High-Performance Configuration:")
    print("   Use case: Maximum speed, sufficient memory available\n")

    store_high_perf = DXMTVectorDB(
        table_name="production_high_perf",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
        host="localhost",
        port=9999,
        # Performance optimizations
        index_type="hnsw",           # Fast and accurate
        distance_metric="cosine",     # Standard for text embeddings
        enable_full_text_search=True, # Hybrid search capability
    )

    print(f"   ✓ Created: {store_high_perf._table_name}")
    print(f"   - Index: HNSW (fast approximate search)")
    print(f"   - Metric: Cosine (text embeddings)")
    print(f"   - Features: Full-text search enabled\n")

    # Memory-efficient configuration
    print("2. Memory-Efficient Configuration:")
    print("   Use case: Large dataset, memory constraints\n")

    store_memory_eff = DXMTVectorDB(
        table_name="production_memory_eff",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
        host="localhost",
        port=9999,
        # Memory optimizations
        index_type="ivf_pq",          # Product quantization
        distance_metric="cosine",
        enable_full_text_search=False,# Disable if not needed
    )

    print(f"   ✓ Created: {store_memory_eff._table_name}")
    print(f"   - Index: IVF-PQ (memory efficient)")
    print(f"   - Metric: Cosine")
    print(f"   - Features: Vector-only search\n")

    # Accuracy-first configuration
    print("3. Accuracy-First Configuration:")
    print("   Use case: Small dataset, maximum precision required\n")

    store_accurate = DXMTVectorDB(
        table_name="production_accurate",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
        host="localhost",
        port=9999,
        # Accuracy optimizations
        index_type="brute_force",     # Exact search
        distance_metric="euclidean",  # Precise distance
        enable_full_text_search=True,
    )

    print(f"   ✓ Created: {store_accurate._table_name}")
    print(f"   - Index: Brute Force (100% recall)")
    print(f"   - Metric: Euclidean (precise)")
    print(f"   - Insert: Standard")
    print(f"   - Features: Full-text search enabled\n")

    # Cleanup
    print("Cleaning up...")
    for store in [store_high_perf, store_memory_eff, store_accurate]:
        try:
            store._client.delete_table(store._table_name)
        except Exception:
            pass
    print("✓ Done\n")


def example_connection_options() -> None:
    """Demonstrate connection configuration options."""
    print("=== Connection Configuration Options ===\n")

    embeddings = default_embeddings()

    # 1. Basic connection
    print("1. Basic Connection (default):")
    store1 = DXMTVectorDB(
        table_name="conn_test_1",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
    )
    print(f"   ✓ Connected to: localhost:9999")
    print(f"   - SSL: No\n")

    # 2. Custom host/port
    #
    # Shown as arguments rather than built: the constructor creates the table, so it connects,
    # and pointing it at an illustrative hostname made this example fail with a DNS error under
    # a line that said "✓ Would connect to".
    print("2. Custom Host/Port (arguments, not a live connection):")
    print("   DXMTVectorDB(table_name=..., embedding=..., dimension=...,")
    print("                host='my-dxmt-server.example.com', port=8080, ssl=True)")
    print("   - SSL: Yes\n")

    # 3. Custom headers (authentication, etc.)
    print("3. With Custom Headers:")
    store3 = DXMTVectorDB(
        table_name="conn_test_3",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
        headers={
            "Authorization": "Bearer YOUR_API_TOKEN",
            "X-Custom-Header": "custom-value",
        },
    )
    print(f"   ✓ Connection with custom headers")
    print(f"   - Authorization: Bearer token")
    print(f"   - Custom headers: Yes\n")

    # 4. Reuse existing client
    print("4. Reuse Existing Client:")
    # From the SDK, which is where the client lives: `langchain_dxmt.client` has never been a
    # module in this package, so this line raised ModuleNotFoundError every time anyone ran
    # the example this far.
    from dxmt import DXMTClient

    # Create client once
    client = DXMTClient(host="localhost", port=9999)

    # Reuse for multiple stores
    store4a = DXMTVectorDB(
        table_name="conn_test_4a",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
        client=client,  # Reuse client
    )

    store4b = DXMTVectorDB(
        table_name="conn_test_4b",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
        client=client,  # Same client
    )

    print(f"   ✓ Created 2 stores with shared client")
    print(f"   - Benefit: Connection pooling, resource efficiency\n")


def main() -> None:
    """Run all advanced configuration examples."""
    print("=" * 70)
    print("LangChain DXMT - Advanced Configuration Examples")
    print("=" * 70)
    print()

    try:
        # Run examples
        example_index_types()
        print()

        example_distance_metrics()
        print()

        print()

        example_production_config()
        print()

        example_connection_options()
        print()

        print("=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        print("\n💡 Configuration Tips:")
        print("   - Choose index type based on dataset size and memory")
        print("   - Use cosine distance for text embeddings")
        print("   - Use ultimate insert mode for maximum performance")
        print("   - Enable full-text search for hybrid queries")
        print("   - Use SSL in production environments")
        print("   - Reuse clients for connection pooling\n")

    except Exception as e:
        print(f"\n✗ Error: {e}\n")
        print("Make sure DXMT server is running on localhost:9999")


if __name__ == "__main__":
    main()
