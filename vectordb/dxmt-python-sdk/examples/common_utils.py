"""
Common Utilities for DXMT Examples

Shared functions used across multiple example scripts to reduce code duplication.
"""

import time
from typing import List, Dict, Any, Optional
from dxmt import Client


def create_test_client(host: str = "localhost", port: int = 9999) -> Client:
    """
    Create and return a DXMT client with standard configuration.

    Args:
        host: Server hostname (default: localhost)
        port: Server port (default: 9999)

    Returns:
        Configured Client instance
    """
    client = Client(host=host, port=port)
    print(f"✓ Connected to DXMT server at {host}:{port}")
    return client


def generate_dummy_data(
    num_records: int = 100,
    dimension: int = 128,
    include_metadata: bool = True
) -> List[Dict[str, Any]]:
    """
    Generate dummy vector data for testing.

    Args:
        num_records: Number of records to generate
        dimension: Vector dimensionality
        include_metadata: Whether to include metadata fields

    Returns:
        List of record dictionaries
    """
    import random

    records = []
    categories = ["technology", "science", "business", "health", "sports"]
    cities = ["New York", "San Francisco", "London", "Tokyo", "Berlin"]

    for i in range(num_records):
        record = {
            "id": str(i),
            "vector": [random.random() for _ in range(dimension)],
        }

        if include_metadata:
            record["metadata"] = {
                "category": random.choice(categories),
                "city": random.choice(cities),
                "score": random.randint(1, 100),
                "active": random.choice([True, False]),
                "title": f"Document {i}",
                "timestamp": time.time() - random.randint(0, 86400 * 30),  # Last 30 days
            }

        records.append(record)

    print(f"✓ Generated {num_records} dummy records (dim={dimension})")
    return records


def print_results(
    results: List[Dict[str, Any]],
    max_display: int = 5,
    show_vectors: bool = False
) -> None:
    """
    Pretty print search results.

    Args:
        results: List of search result dictionaries
        max_display: Maximum number of results to display
        show_vectors: Whether to display vectors
    """
    if not results:
        print("No results found")
        return

    print(f"\nFound {len(results)} results:")
    print("=" * 80)

    for i, result in enumerate(results[:max_display]):
        print(f"\nResult {i + 1}:")
        print(f"  ID: {result.get('id', 'N/A')}")

        if 'score' in result:
            print(f"  Score: {result['score']:.4f}")

        if 'distance' in result:
            print(f"  Distance: {result['distance']:.4f}")

        if 'metadata' in result and result['metadata']:
            print(f"  Metadata:")
            for key, value in result['metadata'].items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.2f}")
                else:
                    print(f"    {key}: {value}")

        if show_vectors and 'vector' in result:
            vector = result['vector']
            if len(vector) <= 10:
                print(f"  Vector: {vector}")
            else:
                print(f"  Vector: [{vector[0]:.3f}, {vector[1]:.3f}, ..., {vector[-1]:.3f}] (dim={len(vector)})")

    if len(results) > max_display:
        print(f"\n... and {len(results) - max_display} more results")

    print("=" * 80)


def benchmark(func, *args, **kwargs):
    """
    Benchmark a function's execution time.

    Args:
        func: Function to benchmark
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Tuple of (result, elapsed_time_ms)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_ms = (time.time() - start_time) * 1000

    return result, elapsed_ms


def print_benchmark(name: str, elapsed_ms: float) -> None:
    """
    Print benchmark results.

    Args:
        name: Operation name
        elapsed_ms: Elapsed time in milliseconds
    """
    print(f"⏱  {name}: {elapsed_ms:.2f}ms")


def setup_test_table(
    client: Client,
    table_name: str,
    dimension: int = 128,
    enable_full_text: bool = False,
    recreate: bool = True
) -> None:
    """
    Setup a test table with standard configuration.

    Args:
        client: DXMT client
        table_name: Name of the table to create
        dimension: Vector dimensionality
        enable_full_text: Whether to enable full-text search
        recreate: Whether to delete existing table first
    """
    # Delete existing table if recreate=True
    if recreate:
        try:
            client.delete_table(table_name)
            print(f"✓ Deleted existing table '{table_name}'")
        except Exception:
            pass  # Table doesn't exist

    # Create table
    config = {
        "name": table_name,
        "dimension": dimension,
        "index_type": "HNSW",
        "distance_metric": "cosine",
    }

    if enable_full_text:
        config["enable_full_text_search"] = True
        config["text_fields"] = [{
            "name": "content",
            "source": {"MetadataField": "text"},
            "analyzer": "Standard",
            "boost": 1.0,
        }]

    client.create_table(**config)
    print(f"✓ Created table '{table_name}' (dim={dimension}, full_text={enable_full_text})")


def compare_search_methods(
    client: Client,
    table_name: str,
    query_vector: List[float],
    k: int = 5
) -> Dict[str, Any]:
    """
    Compare different search methods (vector, hybrid, filtered).

    Args:
        client: DXMT client
        table_name: Table name
        query_vector: Query vector
        k: Number of results

    Returns:
        Dictionary with results and timings
    """
    results = {}

    # Vector search
    vector_results, vector_time = benchmark(
        client.search,
        table_name=table_name,
        vector=query_vector,
        k=k
    )
    results['vector'] = {
        'results': vector_results,
        'time_ms': vector_time,
        'count': len(vector_results)
    }

    # Try hybrid search if available
    try:
        hybrid_results, hybrid_time = benchmark(
            client.hybrid_search,
            table_name=table_name,
            vector=query_vector,
            text="test query",
            k=k,
            vector_weight=0.7
        )
        results['hybrid'] = {
            'results': hybrid_results,
            'time_ms': hybrid_time,
            'count': len(hybrid_results)
        }
    except Exception:
        results['hybrid'] = None

    # Filtered search
    try:
        filtered_results, filtered_time = benchmark(
            client.search,
            table_name=table_name,
            vector=query_vector,
            k=k,
            filter={"category": {"$eq": "technology"}}
        )
        results['filtered'] = {
            'results': filtered_results,
            'time_ms': filtered_time,
            'count': len(filtered_results)
        }
    except Exception:
        results['filtered'] = None

    return results


def print_section(title: str, width: int = 80) -> None:
    """
    Print a formatted section header.

    Args:
        title: Section title
        width: Width of the header
    """
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")


# ── Shared embedding utilities ────────────────────────────────────────────────
# Import these in any example: `from common_utils import embed, EMBED_DIM`

import hashlib
import math as _math

try:
    from sentence_transformers import SentenceTransformer as _ST
    _SMODEL = _ST("all-MiniLM-L6-v2")
    EMBED_DIM: int = _SMODEL.get_sentence_embedding_dimension()
    _HAS_ST = True
except Exception:
    _SMODEL = None
    EMBED_DIM = 384
    _HAS_ST = False


def synthetic_vector(seed: str, dim: int) -> list:
    """Deterministic unit vector from a string seed (no external dependency)."""
    raw = []
    for i in range(dim):
        h = int(hashlib.md5(f"{seed}:{i}".encode()).hexdigest(), 16)
        raw.append((h % 10000) / 10000.0 * 2 - 1)
    norm = _math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


def embed(texts: list) -> list:
    """Embed a list of strings. Uses sentence-transformers when available,
    falls back to deterministic synthetic vectors."""
    if _HAS_ST and _SMODEL is not None:
        return _SMODEL.encode(texts, convert_to_numpy=True).tolist()
    return [synthetic_vector(t, EMBED_DIM) for t in texts]


if __name__ == "__main__":
    # Demo usage
    print_section("Common Utilities Demo")

    # Create client
    client = create_test_client()

    # Generate dummy data
    data = generate_dummy_data(num_records=10, dimension=8)

    # Print results demo
    print_results(data[:3], show_vectors=True)

    # Benchmark demo
    import random
    result, elapsed = benchmark(
        lambda: sum([random.random() for _ in range(1000000)])
    )
    print_benchmark("Random sum calculation", elapsed)

    print("\n✓ Common utilities demo complete")
