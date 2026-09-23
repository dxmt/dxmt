"""Advanced filtering example for DXMT Vector Store with LlamaIndex.

This example demonstrates:
1. Simple metadata filters
2. Complex filters with AND/OR conditions
3. Range filters (comparison operators)
4. IN/NOT IN filters
5. Combining filters with semantic search
"""

from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.dxmt import DXMTVectorStore
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    FilterCondition,
)
from embedding import default_embed_model, embed_dimension


def create_sample_documents():
    """Create sample documents with rich metadata."""
    return [
        Document(
            text="Python is a high-level programming language known for its simplicity and readability.",
            metadata={
                "language": "Python",
                "category": "Programming",
                "difficulty": "Beginner",
                "year": 1991,
                "popularity": 100,
                "tags": ["general-purpose", "interpreted", "dynamic"]
            }
        ),
        Document(
            text="JavaScript is essential for web development, running in browsers and on servers with Node.js.",
            metadata={
                "language": "JavaScript",
                "category": "Programming",
                "difficulty": "Beginner",
                "year": 1995,
                "popularity": 95,
                "tags": ["web", "interpreted", "dynamic"]
            }
        ),
        Document(
            text="Rust is a systems programming language focused on safety, speed, and concurrency.",
            metadata={
                "language": "Rust",
                "category": "Programming",
                "difficulty": "Advanced",
                "year": 2010,
                "popularity": 75,
                "tags": ["systems", "compiled", "static"]
            }
        ),
        Document(
            text="Go is a statically typed language designed for building simple, reliable, and efficient software.",
            metadata={
                "language": "Go",
                "category": "Programming",
                "difficulty": "Intermediate",
                "year": 2009,
                "popularity": 80,
                "tags": ["systems", "compiled", "static"]
            }
        ),
        Document(
            text="TypeScript adds static typing to JavaScript, making it more suitable for large-scale applications.",
            metadata={
                "language": "TypeScript",
                "category": "Programming",
                "difficulty": "Intermediate",
                "year": 2012,
                "popularity": 90,
                "tags": ["web", "compiled", "static"]
            }
        ),
        Document(
            text="Java is a class-based, object-oriented language designed for portability across platforms.",
            metadata={
                "language": "Java",
                "category": "Programming",
                "difficulty": "Intermediate",
                "year": 1995,
                "popularity": 85,
                "tags": ["general-purpose", "compiled", "static"]
            }
        ),
        Document(
            text="C++ is a powerful language used for system software, game engines, and performance-critical applications.",
            metadata={
                "language": "C++",
                "category": "Programming",
                "difficulty": "Advanced",
                "year": 1985,
                "popularity": 70,
                "tags": ["systems", "compiled", "static"]
            }
        ),
    ]


def main():
    """Run advanced filtering example."""
    print("=" * 60)
    print("DXMT Vector Store - Advanced Filtering Example")
    print("=" * 60)

    # 1. Initialize vector store. Its dimension is the embedding model's.
    embed_model = default_embed_model()
    print("\n1. Initializing DXMT vector store...")
    vector_store = DXMTVectorStore(
        table_name=f"llamaindex_filters_example_{embed_dimension(embed_model)}",
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

    # 2. Create index and add documents
    print("\n2. Creating index and adding documents...")
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    documents = create_sample_documents()
    for i, doc in enumerate(documents, 1):
        index.insert(doc)
        print(f"  ✓ Document {i}/{len(documents)}: {doc.metadata['language']}")

    # Get query embedding for searches
    query_text = "programming language"
    query_embedding = embed_model.get_text_embedding(query_text)

    # 3. Simple equality filter
    print("\n3. Simple Equality Filter")
    print("   Filter: difficulty == 'Beginner'")
    print("   " + "─" * 56)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="difficulty", value="Beginner", operator=FilterOperator.EQ)
        ]
    )

    query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=5,
        filters=filters,
    )
    results = vector_store.query(query)

    for i, (node, score) in enumerate(zip(results.nodes, results.similarities), 1):
        print(f"   {i}. {node.metadata['language']} (Score: {score:.4f}, Difficulty: {node.metadata['difficulty']})")

    # 4. Range filter (comparison operators)
    print("\n4. Range Filter")
    print("   Filter: year >= 2000 AND popularity >= 80")
    print("   " + "─" * 56)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="year", value=2000, operator=FilterOperator.GTE),
            MetadataFilter(key="popularity", value=80, operator=FilterOperator.GTE),
        ],
        condition=FilterCondition.AND,
    )

    query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=5,
        filters=filters,
    )
    results = vector_store.query(query)

    for i, (node, score) in enumerate(zip(results.nodes, results.similarities), 1):
        lang = node.metadata['language']
        year = node.metadata['year']
        pop = node.metadata['popularity']
        print(f"   {i}. {lang} (Score: {score:.4f}, Year: {year}, Popularity: {pop})")

    # 5. IN filter
    print("\n5. IN Filter")
    print("   Filter: difficulty IN ['Intermediate', 'Advanced']")
    print("   " + "─" * 56)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="difficulty",
                value=["Intermediate", "Advanced"],
                operator=FilterOperator.IN
            )
        ]
    )

    query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=5,
        filters=filters,
    )
    results = vector_store.query(query)

    for i, (node, score) in enumerate(zip(results.nodes, results.similarities), 1):
        lang = node.metadata['language']
        diff = node.metadata['difficulty']
        print(f"   {i}. {lang} (Score: {score:.4f}, Difficulty: {diff})")

    # 6. Complex AND filter
    print("\n6. Complex AND Filter")
    print("   Filter: category == 'Programming' AND year < 2000 AND popularity > 70")
    print("   " + "─" * 56)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="Programming", operator=FilterOperator.EQ),
            MetadataFilter(key="year", value=2000, operator=FilterOperator.LT),
            MetadataFilter(key="popularity", value=70, operator=FilterOperator.GT),
        ],
        condition=FilterCondition.AND,
    )

    query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=5,
        filters=filters,
    )
    results = vector_store.query(query)

    if results.nodes:
        for i, (node, score) in enumerate(zip(results.nodes, results.similarities), 1):
            lang = node.metadata['language']
            year = node.metadata['year']
            pop = node.metadata['popularity']
            print(f"   {i}. {lang} (Score: {score:.4f}, Year: {year}, Popularity: {pop})")
    else:
        print("   No results found matching all criteria")

    # 7. OR filter
    print("\n7. OR Filter")
    print("   Filter: difficulty == 'Beginner' OR popularity >= 90")
    print("   " + "─" * 56)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="difficulty", value="Beginner", operator=FilterOperator.EQ),
            MetadataFilter(key="popularity", value=90, operator=FilterOperator.GTE),
        ],
        condition=FilterCondition.OR,
    )

    query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=5,
        filters=filters,
    )
    results = vector_store.query(query)

    for i, (node, score) in enumerate(zip(results.nodes, results.similarities), 1):
        lang = node.metadata['language']
        diff = node.metadata['difficulty']
        pop = node.metadata['popularity']
        print(f"   {i}. {lang} (Score: {score:.4f}, Difficulty: {diff}, Popularity: {pop})")

    # 8. NOT EQUAL filter
    print("\n8. NOT EQUAL Filter")
    print("   Filter: difficulty != 'Beginner'")
    print("   " + "─" * 56)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="difficulty", value="Beginner", operator=FilterOperator.NE)
        ]
    )

    query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=5,
        filters=filters,
    )
    results = vector_store.query(query)

    for i, (node, score) in enumerate(zip(results.nodes, results.similarities), 1):
        lang = node.metadata['language']
        diff = node.metadata['difficulty']
        print(f"   {i}. {lang} (Score: {score:.4f}, Difficulty: {diff})")

    # 9. Multiple comparison operators
    print("\n9. Multiple Comparison Operators")
    print("   Filter: year > 1990 AND year < 2010 AND popularity >= 75")
    print("   " + "─" * 56)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="year", value=1990, operator=FilterOperator.GT),
            MetadataFilter(key="year", value=2010, operator=FilterOperator.LT),
            MetadataFilter(key="popularity", value=75, operator=FilterOperator.GTE),
        ],
        condition=FilterCondition.AND,
    )

    query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=5,
        filters=filters,
    )
    results = vector_store.query(query)

    for i, (node, score) in enumerate(zip(results.nodes, results.similarities), 1):
        lang = node.metadata['language']
        year = node.metadata['year']
        pop = node.metadata['popularity']
        print(f"   {i}. {lang} (Score: {score:.4f}, Year: {year}, Popularity: {pop})")

    # 10. Using retriever with filters
    print("\n10. Using Retriever with Filters")
    print("    Filter: difficulty == 'Advanced'")
    print("    " + "─" * 56)

    retriever = index.as_retriever(
        similarity_top_k=3,
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="difficulty", value="Advanced", operator=FilterOperator.EQ)
            ]
        ),
    )

    nodes = retriever.retrieve("fast systems programming language")

    for i, node in enumerate(nodes, 1):
        lang = node.metadata['language']
        diff = node.metadata['difficulty']
        print(f"    {i}. {lang} (Score: {node.score:.4f}, Difficulty: {diff})")
        print(f"       {node.text[:80]}...")

    # Summary
    print("\n" + "=" * 60)
    print("Filter Capabilities Summary")
    print("=" * 60)
    print("""
    Supported Filter Operators:
    - EQ (==)          : Exact match
    - NE (!=)          : Not equal
    - GT (>)           : Greater than
    - GTE (>=)         : Greater than or equal
    - LT (<)           : Less than
    - LTE (<=)         : Less than or equal
    - IN               : Value in list
    - NIN              : Value not in list
    - CONTAINS         : String contains
    - TEXT_MATCH       : Text matching

    Filter Conditions:
    - AND              : All filters must match
    - OR               : Any filter must match

    Filters are combined with semantic search to provide:
    - Precise results (filtered by metadata)
    - Semantic relevance (ranked by similarity)
    - Flexible queries (complex filter logic)
    """)

    # Clean up
    print("\n11. Cleaning up...")
    # vector_store.clear()
    print("   (Skipped - table preserved for inspection)")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
