"""Hybrid search example for DXMT Python SDK (Fixed Version).

This example demonstrates combining vector similarity search with full-text search.

Improvements:
- Uses real sentence-transformers embeddings (not dummy vectors)
- Shows meaningful similarity scores
- Explains score meanings
- Demonstrates the effect of alpha parameter

Requirements:
    pip install dxmt sentence-transformers
"""

from dxmt import Client

# Optional: sentence-transformers for real embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️  sentence-transformers not installed")
    print("   For better demo, install with: pip install sentence-transformers")
    print("   Falling back to simple vectors...\n")


def create_embedder():
    """Create embedding model or return None."""
    if HAS_SENTENCE_TRANSFORMERS:
        print("📊 Loading sentence-transformers model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Model loaded (dimension: 384)\n")
        return model
    return None


def embed_text(text, model):
    """Generate embedding for text."""
    if model:
        return model.encode(text).tolist()
    else:
        # Fallback: use simple hash-based vector
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        # Generate pseudo-random but deterministic vector
        import random
        random.seed(hash_val)
        return [random.random() for _ in range(128)]


def main() -> None:
    """Run hybrid search example."""
    print("=" * 80)
    print("  Hybrid Search Demo - Fixed Version")
    print("=" * 80)
    print()

    # Create embedder
    model = create_embedder()
    dimension = 384 if model else 128

    # Create client
    print("Connecting to DXMT server...")
    client = Client(host="localhost", port=9999)

    # Create table with full-text search enabled
    print("Creating table with full-text search...")

    try:
        client.delete_table("articles_fixed")
    except:
        pass

    client.create_table(
        name="articles_fixed",
        dimension=dimension,
        distance_metric="cosine",
        index_type="hnsw",
        enable_full_text_search=True,
    )
    print("✓ Table created\n")

    # Prepare articles with REAL content
    articles_data = [
        {
            "id": "art1",
            "title": "Introduction to Machine Learning",
            "content": (
                "Machine learning is a subset of artificial intelligence that "
                "enables systems to learn and improve from experience. It focuses "
                "on developing algorithms that can access data and learn patterns."
            ),
            "category": "AI",
        },
        {
            "id": "art2",
            "title": "Deep Learning with Neural Networks",
            "content": (
                "Deep learning uses neural networks with multiple layers to "
                "process data. Neural networks are inspired by the human brain "
                "and can learn complex patterns through backpropagation."
            ),
            "category": "AI",
        },
        {
            "id": "art3",
            "title": "Natural Language Processing Basics",
            "content": (
                "NLP enables computers to understand and process human language. "
                "It combines linguistics, computer science, and machine learning "
                "to analyze text and speech data."
            ),
            "category": "NLP",
        },
        {
            "id": "art4",
            "title": "Computer Vision Applications",
            "content": (
                "Computer vision allows machines to interpret and understand "
                "visual information from the world. Applications include image "
                "recognition, object detection, and autonomous vehicles."
            ),
            "category": "CV",
        },
    ]

    # Insert articles with REAL embeddings
    print("Inserting articles with semantic embeddings...")
    records = []
    for article in articles_data:
        # Generate embedding from CONTENT (semantic representation)
        embedding = embed_text(article['content'], model)

        records.append({
            "id": article["id"],
            "vector": embedding,
            "metadata": {  # Using metadata (SDK auto-maps to payload)
                "title": article["title"],
                "content": article["content"],
                "category": article["category"],
            }
        })

    client.insert_records(table_name="articles_fixed", records=records)
    print(f"✓ Inserted {len(records)} articles with real embeddings\n")

    # Display article info
    print("=" * 80)
    print("Article Summary:")
    print("=" * 80)
    for article in articles_data:
        print(f"[{article['id']}] {article['title']}")
        print(f"      {article['content'][:60]}...")
        print()

    # ========================================================================
    # Test 1: Hybrid Search with different alpha values
    # ========================================================================
    print("=" * 80)
    print("HYBRID SEARCH: Combining vector similarity + text matching")
    print("=" * 80)
    print()

    # Query: Looking for machine learning content
    query_text = "machine learning algorithms"
    query_vector = embed_text(query_text, model)

    print(f"Query: '{query_text}'")
    print()

    # Try different alpha values
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]

    for alpha in alphas:
        mode = (
            'text only' if alpha == 0
            else 'vector only' if alpha == 1
            else f'{int(alpha*100)}% vector + {int((1-alpha)*100)}% text'
        )

        print(f"Alpha = {alpha} ({mode}):")
        print("-" * 80)

        results = client.hybrid_search(
            table_name="articles_fixed",
            vector=query_vector,
            text_query=query_text,
            alpha=alpha,
            limit=3,
        )

        for i, result in enumerate(results, 1):
            rid = result['id']
            score = result['score']

            # Get article title for display
            title = next((a['title'] for a in articles_data if a['id'] == rid), rid)

            print(f"{i}. [{rid}] {title}")
            print(f"   Score: {score:.4f}")

            # Explain what the score means
            if alpha == 0:
                print(f"   (Pure text match - BM25 relevance)")
            elif alpha == 1:
                print(f"   (Pure vector similarity - semantic match)")
            else:
                print(f"   (Hybrid: {int(alpha*100)}% semantic + {int((1-alpha)*100)}% text)")

        print()

    # ========================================================================
    # Test 2: Pure Text Search
    # ========================================================================
    print("=" * 80)
    print("FULL-TEXT SEARCH: Text matching only")
    print("=" * 80)
    print()

    text_queries = [
        "neural networks",
        "computer vision",
        "language processing",
    ]

    for query in text_queries:
        results = client.full_text_search(
            table_name="articles_fixed",
            text_query=query,
            limit=2
        )

        print(f"Query: '{query}'")
        print(f"Found {len(results)} result(s):")

        for i, result in enumerate(results, 1):
            rid = result['id']
            score = result.get('score', 0)
            title = next((a['title'] for a in articles_data if a['id'] == rid), rid)

            print(f"  {i}. [{rid}] {title}")
            print(f"     BM25 Score: {score:.4f}")

        print()

    # ========================================================================
    # Test 3: Semantic Search (Vector Only)
    # ========================================================================
    print("=" * 80)
    print("SEMANTIC SEARCH: Vector similarity only")
    print("=" * 80)
    print()

    # Query: Semantic search for "learning from data"
    semantic_query = "learning patterns from data and experience"
    semantic_vector = embed_text(semantic_query, model)

    print(f"Query: '{semantic_query}'")
    print("(Using pure vector search to find semantically similar content)")
    print()

    results = client.search(
        table_name="articles_fixed",
        vector=semantic_vector,
        k=4
    )

    print("Results (sorted by semantic similarity):")
    for i, result in enumerate(results, 1):
        rid = result['id']
        score = result['score']
        metadata = result.get('metadata', {})
        title = metadata.get('title', rid)

        print(f"{i}. [{rid}] {title}")
        print(f"   Cosine Similarity: {score:.4f}")

        # Explain similarity level
        if score > 0.8:
            level = "Very similar"
        elif score > 0.6:
            level = "Similar"
        elif score > 0.4:
            level = "Somewhat related"
        else:
            level = "Loosely related"

        print(f"   ({level})")
        print()

    # ========================================================================
    # Explanation
    # ========================================================================
    print("=" * 80)
    print("Score Explanations:")
    print("=" * 80)
    print()
    print("📊 Cosine Similarity (Vector Search):")
    print("   - Range: 0.0 to 1.0")
    print("   - 1.0 = Identical semantic meaning")
    print("   - 0.8-1.0 = Very similar")
    print("   - 0.6-0.8 = Similar")
    print("   - 0.4-0.6 = Somewhat related")
    print("   - 0.0-0.4 = Loosely related or unrelated")
    print()
    print("📄 BM25 Score (Text Search):")
    print("   - Range: 0.0 to 10.0+ (no upper limit)")
    print("   - Higher = More relevant based on term frequency")
    print("   - Considers: term frequency, document length, corpus statistics")
    print()
    print("🔀 Hybrid Score (Combined):")
    print("   - Formula: vector_score × alpha + text_score × (1-alpha)")
    print("   - Alpha controls the balance between semantic and text matching")
    print("   - 0.0 = Pure text search")
    print("   - 0.5 = Balanced")
    print("   - 1.0 = Pure semantic search")
    print()

    # Clean up
    print("=" * 80)
    print("Cleaning up...")
    client.delete_table("articles_fixed")
    client.close()
    print("✓ Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
