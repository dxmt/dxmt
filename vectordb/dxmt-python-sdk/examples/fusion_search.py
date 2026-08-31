"""
Fusion Search with Reciprocal Rank Fusion (RRF)

Demonstrates combining multiple search strategies with RRF algorithm.
Useful for improving search quality by fusing results from different queries.

Use cases:
- Combining vector + text search
- Multi-query search (query variations)
- Multi-aspect search (different perspectives)
- Ensemble search for better accuracy

RRF Formula:
    RRF_score(d) = Σ 1 / (k + rank_i(d))
    where rank_i(d) is the rank of document d in query i
    and k is a constant (typically 60)
"""

from typing import List, Dict, Any
from dxmt import Client
from common_utils import (
    create_test_client,
    setup_test_table,
    generate_dummy_data,
    print_section,
    print_results
)


def reciprocal_rank_fusion(
    results_list: List[List[Dict[str, Any]]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Perform Reciprocal Rank Fusion on multiple result lists.

    Args:
        results_list: List of result lists from different queries
        k: Constant for RRF formula (default: 60)

    Returns:
        Fused results sorted by RRF score
    """
    # Collect all document IDs and their ranks in each query
    doc_ranks = {}  # doc_id -> list of ranks

    for query_idx, results in enumerate(results_list):
        for rank, result in enumerate(results, 1):  # Rank starts at 1
            doc_id = result.get('id')
            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = []
            doc_ranks[doc_id].append(rank)

    # Calculate RRF scores
    doc_scores = {}
    all_docs = {}  # Store full document data

    for query_idx, results in enumerate(results_list):
        for result in results:
            doc_id = result.get('id')
            if doc_id not in all_docs:
                all_docs[doc_id] = result

    for doc_id, ranks in doc_ranks.items():
        # RRF score = sum of 1/(k + rank) for all queries where doc appears
        rrf_score = sum(1.0 / (k + rank) for rank in ranks)
        doc_scores[doc_id] = rrf_score

    # Sort documents by RRF score (descending)
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    # Build final result list
    fused_results = []
    for doc_id, score in sorted_docs:
        doc = all_docs[doc_id].copy()
        doc['rrf_score'] = score
        doc['appeared_in'] = len(doc_ranks[doc_id])
        fused_results.append(doc)

    return fused_results


def multi_query_fusion(
    client: Client,
    table_name: str,
    query_vectors: List[List[float]],
    k: int = 10,
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """
    Perform fusion search with multiple query vectors.

    Args:
        client: DXMT client
        table_name: Table name
        query_vectors: List of query vectors
        k: Number of results per query
        rrf_k: RRF constant

    Returns:
        Fused results
    """
    # Execute multiple searches
    all_results = []
    for i, query_vector in enumerate(query_vectors):
        results = client.search(
            table_name=table_name,
            vector=query_vector,
            k=k
        )
        all_results.append(results)
        print(f"  Query {i+1}: Found {len(results)} results")

    # Fuse results
    fused = reciprocal_rank_fusion(all_results, k=rrf_k)
    return fused


def filtered_fusion(
    client: Client,
    table_name: str,
    query_vector: List[float],
    filters: List[Dict[str, Any]],
    k: int = 10,
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """
    Perform fusion search with multiple filters.

    Args:
        client: DXMT client
        table_name: Table name
        query_vector: Query vector
        filters: List of filter conditions
        k: Number of results per filter
        rrf_k: RRF constant

    Returns:
        Fused results
    """
    # Execute searches with different filters
    all_results = []
    for i, filter_cond in enumerate(filters):
        try:
            results = client.search(
                table_name=table_name,
                vector=query_vector,
                k=k,
                filter=filter_cond
            )
            all_results.append(results)
            print(f"  Filter {i+1}: Found {len(results)} results")
        except Exception as e:
            print(f"  Filter {i+1}: Error - {e}")
            all_results.append([])

    # Fuse results
    fused = reciprocal_rank_fusion(all_results, k=rrf_k)
    return fused


def hybrid_fusion(
    client: Client,
    table_name: str,
    query_vector: List[float],
    query_text: str,
    weights: List[float],
    k: int = 10,
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """
    Perform fusion with different hybrid search weights.

    Args:
        client: DXMT client
        table_name: Table name
        query_vector: Query vector
        query_text: Query text
        weights: List of vector weights for hybrid search
        k: Number of results per weight
        rrf_k: RRF constant

    Returns:
        Fused results
    """
    # Execute hybrid searches with different weights
    all_results = []
    for i, weight in enumerate(weights):
        try:
            results = client.hybrid_search(
                table_name=table_name,
                vector=query_vector,
                text=query_text,
                k=k,
                vector_weight=weight
            )
            all_results.append(results)
            print(f"  Weight {weight}: Found {len(results)} results")
        except Exception as e:
            print(f"  Weight {weight}: Error - {e}")
            # Fallback to vector search
            results = client.search(
                table_name=table_name,
                vector=query_vector,
                k=k
            )
            all_results.append(results)

    # Fuse results
    fused = reciprocal_rank_fusion(all_results, k=rrf_k)
    return fused


def main():
    print_section("Fusion Search with RRF Demo")

    # Create client
    client = create_test_client()

    # Setup table
    table_name = "fusion_demo"
    dimension = 128
    setup_test_table(client, table_name, dimension, enable_full_text=True)

    # Generate test data
    print("Generating test data...")
    records = generate_dummy_data(num_records=100, dimension=dimension)
    client.insert_records(table_name, records)
    print(f"✓ Inserted {len(records)} records\n")

    # Create query vectors
    import random
    query_vector1 = [random.random() for _ in range(dimension)]
    query_vector2 = [random.random() for _ in range(dimension)]
    query_vector3 = [random.random() for _ in range(dimension)]

    # ==========================================================================
    # Example 1: Multi-Query Fusion
    # ==========================================================================
    print_section("Example 1: Multi-Query Fusion")

    print("Combining results from 3 different query vectors...")
    query_vectors = [query_vector1, query_vector2, query_vector3]

    fused_results = multi_query_fusion(
        client, table_name, query_vectors, k=20, rrf_k=60
    )

    print(f"\n✓ Fused {len(fused_results)} unique documents")
    print("\nTop 5 results by RRF score:")
    for i, result in enumerate(fused_results[:5], 1):
        print(f"{i}. ID: {result['id']}, "
              f"RRF Score: {result['rrf_score']:.4f}, "
              f"Appeared in: {result['appeared_in']}/3 queries")

    # ==========================================================================
    # Example 2: Filtered Fusion
    # ==========================================================================
    print_section("Example 2: Filtered Fusion")

    print("Combining results from searches with different filters...")
    filters = [
        {"category": {"$eq": "technology"}},
        {"category": {"$eq": "science"}},
        {"score": {"$gte": 50}},
    ]

    fused_results = filtered_fusion(
        client, table_name, query_vector1, filters, k=10, rrf_k=60
    )

    print(f"\n✓ Fused {len(fused_results)} unique documents")
    print("\nTop 5 results by RRF score:")
    for i, result in enumerate(fused_results[:5], 1):
        category = result.get('metadata', {}).get('category', 'N/A')
        score = result.get('metadata', {}).get('score', 'N/A')
        print(f"{i}. ID: {result['id']}, "
              f"RRF Score: {result['rrf_score']:.4f}, "
              f"Category: {category}, Score: {score}")

    # ==========================================================================
    # Example 3: Hybrid Fusion (if supported)
    # ==========================================================================
    print_section("Example 3: Hybrid Fusion with Different Weights")

    print("Combining results from hybrid searches with different vector weights...")
    weights = [0.3, 0.5, 0.7, 0.9]  # Different vector/text balances

    try:
        fused_results = hybrid_fusion(
            client, table_name, query_vector1,
            query_text="technology science data",
            weights=weights, k=10, rrf_k=60
        )

        print(f"\n✓ Fused {len(fused_results)} unique documents")
        print("\nTop 5 results by RRF score:")
        for i, result in enumerate(fused_results[:5], 1):
            print(f"{i}. ID: {result['id']}, "
                  f"RRF Score: {result['rrf_score']:.4f}, "
                  f"Appeared in: {result['appeared_in']}/4 weights")
    except Exception as e:
        print(f"Hybrid fusion not available: {e}")

    # ==========================================================================
    # Example 4: Comparison with Single Query
    # ==========================================================================
    print_section("Example 4: Comparison with Single Query")

    print("Comparing fusion search vs. single query search...\n")

    # Single query
    single_results = client.search(
        table_name=table_name,
        vector=query_vector1,
        k=10
    )

    # Multi-query fusion
    fusion_results = multi_query_fusion(
        client, table_name,
        [query_vector1, query_vector2],
        k=10, rrf_k=60
    )

    print(f"Single query results: {len(single_results)}")
    print(f"Fusion results: {len(fusion_results)}")

    # Find unique documents in fusion results
    single_ids = {r['id'] for r in single_results}
    fusion_ids = {r['id'] for r in fusion_results}
    unique_in_fusion = fusion_ids - single_ids

    print(f"\nDocuments unique to fusion: {len(unique_in_fusion)}")
    print("Fusion search can discover relevant documents missed by single query!")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print_section("Summary: When to Use Fusion Search")

    print("✅ Use Fusion Search when:")
    print("   • Query intent is ambiguous or multi-faceted")
    print("   • You want to combine multiple search strategies")
    print("   • Improving recall is more important than speed")
    print("   • You have multiple relevant query formulations")
    print("   • Combining text + vector + filtered searches")
    print("")
    print("📊 Benefits:")
    print("   • Better coverage (finds more relevant results)")
    print("   • Reduced position bias (diverse ranking)")
    print("   • Ensemble effect (more robust results)")
    print("   • Flexibility (combine any search types)")
    print("")
    print("⚠️  Considerations:")
    print("   • Multiple queries = higher latency")
    print("   • More API calls = higher cost")
    print("   • Best for offline/batch scenarios")
    print("   • May need tuning (RRF k parameter)")
    print("")
    print("💡 Recommended RRF k values:")
    print("   • k=10-30:  Aggressive fusion (weights first ranks highly)")
    print("   • k=60:     Standard (balanced, recommended)")
    print("   • k=100+:   Conservative fusion (weights more ranks)")

    # Cleanup
    print_section("Cleanup")
    try:
        client.delete_table(table_name)
        print(f"✓ Deleted table '{table_name}'")
    except Exception as e:
        print(f"Error during cleanup: {e}")

    print("\n✓ Fusion search demo complete!")


if __name__ == "__main__":
    main()
