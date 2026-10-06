"""
Maximum Marginal Relevance (MMR) Search Example

Demonstrates MMR algorithm for balancing relevance and diversity in search results.
MMR is useful when you want to avoid returning too many similar results.

Use cases:
- Content recommendation (avoid showing similar items)
- Research paper discovery (diverse perspectives)
- Product search (show variety of options)
- News aggregation (variety of sources)
"""

import random
from typing import List, Dict, Any
from environment import skip

try:
    import numpy as np
except ImportError:
    skip("numpy is not installed: pip install numpy")

from dxmt import Client
from common_utils import (
    create_test_client,
    setup_test_table,
    print_section,
    print_results
)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0-1)
    """
    vec1_arr = np.array(vec1)
    vec2_arr = np.array(vec2)

    dot_product = np.dot(vec1_arr, vec2_arr)
    norm1 = np.linalg.norm(vec1_arr)
    norm2 = np.linalg.norm(vec2_arr)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def mmr_search(
    client: Client,
    table_name: str,
    query_vector: List[float],
    k: int = 10,
    fetch_k: int = 50,
    lambda_mult: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Perform Maximum Marginal Relevance (MMR) search.

    MMR balances relevance to query with diversity among results:
    MMR(D) = λ * sim(D, Q) - (1-λ) * max(sim(D, Di))

    Args:
        client: DXMT client
        table_name: Table name
        query_vector: Query vector
        k: Number of final results to return
        fetch_k: Number of candidate documents to fetch
        lambda_mult: Balance parameter (0-1)
                    0.0 = maximum diversity
                    0.5 = balanced
                    1.0 = maximum relevance (like standard search)

    Returns:
        List of MMR-selected documents
    """
    # Fetch more candidates than needed
    candidates = client.search(
        table_name=table_name,
        vector=query_vector,
        k=fetch_k
    )

    if not candidates or len(candidates) == 0:
        return []

    # Extract vectors and convert to numpy arrays
    # Filter out candidates without vectors
    valid_candidates = []
    candidate_vectors = []
    for candidate in candidates:
        if 'vector' in candidate and candidate['vector'] is not None:
            valid_candidates.append(candidate)
            candidate_vectors.append(candidate['vector'])

    # Update candidates list to only include valid ones
    candidates = valid_candidates

    if len(candidates) == 0:
        return []

    query_arr = np.array(query_vector)

    # Calculate similarities to query
    query_sims = np.array([
        cosine_similarity(query_vector, vec) for vec in candidate_vectors
    ])

    # Initialize selected indices
    selected_indices = []
    remaining_indices = list(range(len(candidates)))

    # Select first document (most similar to query)
    first_idx = int(np.argmax(query_sims))
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    # Select remaining k-1 documents using MMR
    while len(selected_indices) < k and remaining_indices:
        best_score = float('-inf')
        best_idx = None

        for idx in remaining_indices:
            # Similarity to query
            query_sim = query_sims[idx]

            # Maximum similarity to already selected documents
            selected_vectors = [candidate_vectors[i] for i in selected_indices]
            candidate_vec = candidate_vectors[idx]

            sims_to_selected = [
                cosine_similarity(candidate_vec, sel_vec)
                for sel_vec in selected_vectors
            ]
            max_sim_to_selected = max(sims_to_selected) if sims_to_selected else 0.0

            # MMR score
            mmr_score = lambda_mult * query_sim - (1 - lambda_mult) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        else:
            break

    # Return selected documents
    return [candidates[i] for i in selected_indices]


def create_clustered_data(dimension: int = 128, num_clusters: int = 3, points_per_cluster: int = 20):
    """
    Create clustered test data to demonstrate MMR diversity.

    Args:
        dimension: Vector dimensionality
        num_clusters: Number of clusters
        points_per_cluster: Points in each cluster

    Returns:
        List of records
    """
    records = []
    record_id = 0

    topics = ["Python Programming", "Machine Learning", "Data Science",
              "Web Development", "Mobile Apps", "Cloud Computing",
              "Cybersecurity", "Blockchain", "IoT", "AI Research"]

    for cluster_id in range(num_clusters):
        # Generate cluster center
        center = [random.random() * 2 - 1 for _ in range(dimension)]

        for i in range(points_per_cluster):
            # Add noise around cluster center
            vector = [c + random.gauss(0, 0.1) for c in center]

            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = [v / norm for v in vector]

            record = {
                "id": str(record_id),
                "vector": vector,
                "metadata": {
                    "cluster": f"cluster_{cluster_id}",
                    "topic": topics[cluster_id % len(topics)],
                    "title": f"Document {record_id} in cluster {cluster_id}",
                    "relevance_score": 100 - (i * 2),  # Decreasing relevance within cluster
                }
            }
            records.append(record)
            record_id += 1

    return records


def main():
    print_section("MMR (Maximum Marginal Relevance) Search Demo")

    # Create client
    client = create_test_client()

    # Setup table
    table_name = "mmr_demo"
    dimension = 128
    setup_test_table(client, table_name, dimension)

    # Create clustered test data
    print("Creating clustered test data...")
    records = create_clustered_data(dimension=dimension, num_clusters=5, points_per_cluster=10)
    print(f"✓ Generated {len(records)} records in 5 clusters")

    # Insert data
    print("Inserting data...")
    client.insert_records(table_name, records)
    print(f"✓ Inserted {len(records)} records\n")

    # Create query vector (similar to cluster 0)
    query_vector = records[0]['vector']

    # ==========================================================================
    # Example 1: Standard search (high redundancy)
    # ==========================================================================
    print_section("Example 1: Standard Search (No Diversity)")

    print("Query: Documents similar to cluster 0")
    standard_results = client.search(
        table_name=table_name,
        vector=query_vector,
        k=10
    )

    print("\nResults (likely many from same cluster):")
    clusters_found = {}
    for result in standard_results:
        cluster = result['metadata']['cluster']
        clusters_found[cluster] = clusters_found.get(cluster, 0) + 1

    print(f"Cluster distribution: {clusters_found}")
    print_results(standard_results[:5], show_vectors=False)

    # ==========================================================================
    # Example 2: MMR with high relevance (lambda=0.9)
    # ==========================================================================
    print_section("Example 2: MMR with High Relevance (lambda=0.9)")

    print("Query: Same query, but with 90% relevance, 10% diversity")
    mmr_high_relevance = mmr_search(
        client, table_name, query_vector,
        k=10, fetch_k=30, lambda_mult=0.9
    )

    clusters_found = {}
    for result in mmr_high_relevance:
        cluster = result['metadata']['cluster']
        clusters_found[cluster] = clusters_found.get(cluster, 0) + 1

    print(f"Cluster distribution: {clusters_found}")
    print("Note: Still mostly from one cluster, but slightly more diverse")
    print_results(mmr_high_relevance[:5], show_vectors=False)

    # ==========================================================================
    # Example 3: MMR with balanced (lambda=0.5)
    # ==========================================================================
    print_section("Example 3: MMR with Balanced Mode (lambda=0.5)")

    print("Query: Same query, but with 50% relevance, 50% diversity")
    mmr_balanced = mmr_search(
        client, table_name, query_vector,
        k=10, fetch_k=30, lambda_mult=0.5
    )

    clusters_found = {}
    for result in mmr_balanced:
        cluster = result['metadata']['cluster']
        clusters_found[cluster] = clusters_found.get(cluster, 0) + 1

    print(f"Cluster distribution: {clusters_found}")
    print("Note: Good balance of relevance and diversity")
    print_results(mmr_balanced[:5], show_vectors=False)

    # ==========================================================================
    # Example 4: MMR with high diversity (lambda=0.1)
    # ==========================================================================
    print_section("Example 4: MMR with High Diversity (lambda=0.1)")

    print("Query: Same query, but with 10% relevance, 90% diversity")
    mmr_high_diversity = mmr_search(
        client, table_name, query_vector,
        k=10, fetch_k=30, lambda_mult=0.1
    )

    clusters_found = {}
    for result in mmr_high_diversity:
        cluster = result['metadata']['cluster']
        clusters_found[cluster] = clusters_found.get(cluster, 0) + 1

    print(f"Cluster distribution: {clusters_found}")
    print("Note: Maximum diversity - results from many different clusters")
    print_results(mmr_high_diversity[:5], show_vectors=False)

    # ==========================================================================
    # Comparison Summary
    # ==========================================================================
    print_section("Summary: Lambda Parameter Effects")

    print("Lambda parameter controls the relevance-diversity tradeoff:\n")
    print("  λ = 1.0  →  Pure relevance (like standard search)")
    print("  λ = 0.9  →  Mostly relevant, slight diversity")
    print("  λ = 0.7  →  Favor relevance, some diversity")
    print("  λ = 0.5  →  Balanced (recommended for most cases)")
    print("  λ = 0.3  →  Favor diversity, some relevance")
    print("  λ = 0.1  →  Maximum diversity, minimal relevance")
    print("  λ = 0.0  →  Pure diversity (maximum difference)")

    print("\n" + "=" * 80)
    print("Use Case Recommendations:")
    print("=" * 80)
    print("  Content Recommendations  →  λ = 0.3-0.5 (favor diversity)")
    print("  Product Search          →  λ = 0.4-0.6 (balanced)")
    print("  News Aggregation        →  λ = 0.2-0.4 (high diversity)")
    print("  Research Papers         →  λ = 0.4-0.6 (diverse perspectives)")
    print("  Q&A Systems            →  λ = 0.6-0.8 (favor relevance)")

    # Cleanup
    print_section("Cleanup")
    try:
        client.delete_table(table_name)
        print(f"✓ Deleted table '{table_name}'")
    except Exception as e:
        print(f"Error during cleanup: {e}")

    print("\n✓ MMR search demo complete!")


if __name__ == "__main__":
    main()
