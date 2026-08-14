"""Semantic Search Engine using DXMT.

This example demonstrates building a powerful search engine that:
- Understands meaning, not just keywords
- Finds semantically similar documents
- Ranks results by relevance
- Supports metadata filtering

Prerequisites:
- DXMT server running on localhost:9999
- OpenAI API key set in environment: export OPENAI_API_KEY='sk-...'

Install dependencies:
    pip install langchain-dxmt langchain-openai

Use case:
    Internal knowledge search, document discovery, content recommendation,
    or any scenario where keyword search is insufficient.
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from langchain_dxmt import DXMTVectorDB


def create_article_database() -> List[Document]:
    """Create a database of tech articles for demonstration."""
    articles = [
        # AI/ML Articles
        Document(
            page_content="""
            Machine learning models are revolutionizing how we process data. Recent
            advances in deep neural networks have enabled computers to recognize
            patterns in images, understand natural language, and make predictions
            with unprecedented accuracy. Transfer learning allows us to leverage
            pre-trained models for new tasks with minimal training data.
            """,
            metadata={
                "title": "Deep Learning Revolution",
                "author": "Dr. Sarah Chen",
                "category": "AI/ML",
                "published_date": "2026-01-10",
                "tags": ["machine learning", "deep learning", "neural networks"],
                "views": 15420,
                "difficulty": "intermediate"
            }
        ),
        Document(
            page_content="""
            Vector databases are becoming essential infrastructure for AI applications.
            They enable similarity search by storing high-dimensional embeddings and
            performing nearest neighbor queries. Use cases include semantic search,
            recommendation systems, and retrieval-augmented generation (RAG) for LLMs.
            HNSW and IVF are popular indexing algorithms.
            """,
            metadata={
                "title": "Vector Databases Explained",
                "author": "Michael Rodriguez",
                "category": "Databases",
                "published_date": "2026-01-15",
                "tags": ["vector database", "embeddings", "search"],
                "views": 8900,
                "difficulty": "intermediate"
            }
        ),
        Document(
            page_content="""
            Large language models like GPT-4 have transformed natural language processing.
            These models are trained on massive text corpora and can generate human-like
            text, answer questions, translate languages, and even write code. Fine-tuning
            and prompt engineering are key techniques for adapting LLMs to specific tasks.
            """,
            metadata={
                "title": "LLMs: The New Interface",
                "author": "Dr. Sarah Chen",
                "category": "AI/ML",
                "published_date": "2025-12-20",
                "tags": ["LLM", "GPT", "NLP"],
                "views": 23450,
                "difficulty": "beginner"
            }
        ),

        # Web Development
        Document(
            page_content="""
            React 19 introduces several groundbreaking features including Server
            Components, improved Suspense, and automatic batching. These improvements
            make React apps faster and easier to build. The new compiler optimizes
            re-renders automatically, reducing the need for manual memoization.
            Developers can now build more interactive UIs with less code.
            """,
            metadata={
                "title": "What's New in React 19",
                "author": "Emma Thompson",
                "category": "Web Development",
                "published_date": "2026-01-18",
                "tags": ["react", "javascript", "frontend"],
                "views": 12300,
                "difficulty": "intermediate"
            }
        ),
        Document(
            page_content="""
            TypeScript continues to gain popularity for large-scale applications.
            Static typing catches bugs at compile time, improving code quality and
            developer productivity. Recent versions add features like template literal
            types and improved inference. Many major frameworks now provide first-class
            TypeScript support.
            """,
            metadata={
                "title": "TypeScript Best Practices",
                "author": "James Wilson",
                "category": "Web Development",
                "published_date": "2026-01-05",
                "tags": ["typescript", "javascript", "programming"],
                "views": 9800,
                "difficulty": "intermediate"
            }
        ),

        # Cloud & DevOps
        Document(
            page_content="""
            Kubernetes has become the de facto standard for container orchestration.
            It automates deployment, scaling, and management of containerized applications.
            Key concepts include pods, services, deployments, and StatefulSets. Cloud
            providers offer managed Kubernetes services like EKS, GKE, and AKS for
            production workloads.
            """,
            metadata={
                "title": "Kubernetes for Beginners",
                "author": "Alex Kumar",
                "category": "DevOps",
                "published_date": "2025-12-28",
                "tags": ["kubernetes", "containers", "cloud"],
                "views": 17600,
                "difficulty": "beginner"
            }
        ),
        Document(
            page_content="""
            Observability is crucial for modern distributed systems. The three pillars—
            logs, metrics, and traces—provide visibility into system behavior. Tools
            like Prometheus, Grafana, and Jaeger enable monitoring and debugging.
            OpenTelemetry standardizes instrumentation across different platforms.
            """,
            metadata={
                "title": "Observability Best Practices",
                "author": "Michael Rodriguez",
                "category": "DevOps",
                "published_date": "2026-01-12",
                "tags": ["observability", "monitoring", "devops"],
                "views": 7200,
                "difficulty": "advanced"
            }
        ),

        # Data Engineering
        Document(
            page_content="""
            Data pipelines are the backbone of modern analytics. Tools like Apache
            Airflow and dbt enable data engineers to build, schedule, and monitor
            complex workflows. Best practices include idempotent operations, data
            validation, and incremental processing. Real-time streaming with Kafka
            is essential for event-driven architectures.
            """,
            metadata={
                "title": "Building Data Pipelines",
                "author": "Priya Sharma",
                "category": "Data Engineering",
                "published_date": "2026-01-08",
                "tags": ["data engineering", "pipelines", "ETL"],
                "views": 11400,
                "difficulty": "intermediate"
            }
        ),
        Document(
            page_content="""
            Data lakes and warehouses serve different purposes. Lakes store raw,
            unstructured data in formats like Parquet. Warehouses like Snowflake and
            BigQuery provide SQL interfaces for analytics. The modern data stack combines
            both: ingest into lakes, transform with dbt, analyze in warehouses. Lakehouse
            architectures like Delta Lake merge the benefits of both.
            """,
            metadata={
                "title": "Data Lake vs Warehouse",
                "author": "Priya Sharma",
                "category": "Data Engineering",
                "published_date": "2025-12-15",
                "tags": ["data lake", "data warehouse", "analytics"],
                "views": 14200,
                "difficulty": "intermediate"
            }
        ),

        # Cybersecurity
        Document(
            page_content="""
            Zero-trust security assumes no user or system is trustworthy by default.
            Every access request must be verified, regardless of origin. Key principles
            include least privilege access, micro-segmentation, and continuous verification.
            Multi-factor authentication and encryption are essential components.
            """,
            metadata={
                "title": "Zero-Trust Security Model",
                "author": "David Martinez",
                "category": "Security",
                "published_date": "2026-01-03",
                "tags": ["security", "zero trust", "cybersecurity"],
                "views": 10300,
                "difficulty": "advanced"
            }
        ),
    ]
    return articles


def print_search_results(query: str, results: List[tuple], show_scores: bool = True):
    """Pretty print search results."""
    print(f"\n🔍 Query: \"{query}\"")
    print(f"📊 Found {len(results)} results\n")

    for i, item in enumerate(results, 1):
        if show_scores:
            doc, score = item
            print(f"{'─' * 80}")
            print(f"Result #{i} (Relevance: {score:.4f})")
        else:
            doc = item
            print(f"{'─' * 80}")
            print(f"Result #{i}")

        print(f"Title: {doc.metadata.get('title', 'N/A')}")
        print(f"Author: {doc.metadata.get('author', 'N/A')}")
        print(f"Category: {doc.metadata.get('category', 'N/A')}")
        print(f"Published: {doc.metadata.get('published_date', 'N/A')}")
        print(f"Tags: {', '.join(doc.metadata.get('tags', []))}")
        print(f"Views: {doc.metadata.get('views', 0):,}")
        print(f"Difficulty: {doc.metadata.get('difficulty', 'N/A')}")

        # Show content preview
        content = doc.page_content.strip()
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"\nPreview:\n{preview}")
        print()


def main():
    """Run the semantic search engine example."""
    print("=" * 80)
    print("Semantic Search Engine with DXMT")
    print("=" * 80)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        return

    # 1. Initialize
    print("\n1. Initializing search engine...")
    embeddings = OpenAIEmbeddings()

    vectorstore = DXMTVectorDB(
        table_name="semantic_search_demo",
        embedding=embeddings,
        dimension=1536,
        host="localhost",
        port=9999,
        distance_metric="cosine",
        index_type="hnsw",
        enable_full_text_search=True,
    )
    print("   ✅ Search engine initialized")

    # 2. Index documents
    print("\n2. Indexing articles...")
    articles = create_article_database()
    ids = vectorstore.add_documents(articles)
    print(f"   ✅ Indexed {len(ids)} articles")

    # 3. Demonstrate semantic search capabilities
    print("\n" + "=" * 80)
    print("Demo 1: Understanding Intent, Not Just Keywords")
    print("=" * 80)
    print("\n💡 Semantic search understands meaning, not just exact matches\n")

    # Query uses different words but same meaning
    query1 = "How do I deploy containerized apps at scale?"
    results1 = vectorstore.similarity_search_with_score(query1, k=3)
    print_search_results(query1, results1)

    print("\n💬 Notice:")
    print("   - Query mentions 'containerized apps' → Found Kubernetes article")
    print("   - No exact keyword match needed!")

    # 4. Compare with different phrasing
    print("\n" + "=" * 80)
    print("Demo 2: Different Phrasing, Same Results")
    print("=" * 80)

    query2a = "neural network advancements"
    query2b = "recent progress in deep learning"

    print(f"\n🔍 Query A: \"{query2a}\"")
    results2a = vectorstore.similarity_search(query2a, k=2)
    for i, doc in enumerate(results2a, 1):
        print(f"   {i}. {doc.metadata['title']}")

    print(f"\n🔍 Query B: \"{query2b}\"")
    results2b = vectorstore.similarity_search(query2b, k=2)
    for i, doc in enumerate(results2b, 1):
        print(f"   {i}. {doc.metadata['title']}")

    print("\n💬 Both queries return similar results because they're semantically equivalent!")

    # 5. Metadata filtering
    print("\n" + "=" * 80)
    print("Demo 3: Filtered Search (Category + Difficulty)")
    print("=" * 80)

    query3 = "best practices for coding"
    filter3 = {
        "category": "Web Development",
        "difficulty": "intermediate"
    }

    results3 = vectorstore.similarity_search(query3, k=5, filter=filter3)
    print_search_results(query3, results3, show_scores=False)

    print("\n💬 Filtered to 'Web Development' category with 'intermediate' difficulty")

    # 6. Search by author
    print("\n" + "=" * 80)
    print("Demo 4: Find All Articles by Specific Author")
    print("=" * 80)

    query4 = "data engineering concepts"
    filter4 = {"author": "Priya Sharma"}

    results4 = vectorstore.similarity_search(query4, k=10, filter=filter4)
    print_search_results(query4, results4, show_scores=False)

    # 7. Show hybrid search advantage (if supported)
    print("\n" + "=" * 80)
    print("Demo 5: Hybrid Search (Semantic + Keyword)")
    print("=" * 80)
    print("\n💡 Combines vector similarity with keyword matching\n")

    query5 = "Kubernetes deployment"

    # Pure semantic search
    print("📍 Pure Semantic Search:")
    semantic_results = vectorstore.similarity_search(query5, k=3)
    for i, doc in enumerate(semantic_results, 1):
        print(f"   {i}. {doc.metadata['title']} ({doc.metadata['category']})")

    # Hybrid search (if supported)
    try:
        print("\n📍 Hybrid Search (70% semantic + 30% keyword):")
        hybrid_results = vectorstore.hybrid_search(
            query=query5,
            k=3,
            vector_weight=0.7  # 70% semantic, 30% keyword
        )
        for i, doc in enumerate(hybrid_results, 1):
            print(f"   {i}. {doc.metadata['title']} ({doc.metadata['category']})")

        print("\n💬 Hybrid search often provides better results for specific terms")
    except (AttributeError, ValueError) as e:
        print(f"\n⚠️  Hybrid search not available: {e}")

    # 8. Interactive search
    print("\n" + "=" * 80)
    print("Interactive Search Mode")
    print("=" * 80)
    print("\n💡 Try searches like:")
    print("   - 'how to monitor microservices'")
    print("   - 'frontend framework comparison'")
    print("   - 'machine learning for beginners'")
    print("   - 'data storage solutions'")
    print("\nCommands: 'stats' for database stats, 'quit' to exit\n")

    try:
        while True:
            query = input("🔍 Search: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break

            if query.lower() == "stats":
                print(f"\n📊 Database Statistics:")
                print(f"   Total articles: {len(ids)}")
                categories = {}
                difficulties = {}
                for article in articles:
                    cat = article.metadata.get("category", "N/A")
                    diff = article.metadata.get("difficulty", "N/A")
                    categories[cat] = categories.get(cat, 0) + 1
                    difficulties[diff] = difficulties.get(diff, 0) + 1

                print(f"\n   By Category:")
                for cat, count in sorted(categories.items()):
                    print(f"      {cat}: {count}")

                print(f"\n   By Difficulty:")
                for diff, count in sorted(difficulties.items()):
                    print(f"      {diff}: {count}")
                print()
                continue

            if not query:
                continue

            # Perform search
            results = vectorstore.similarity_search_with_score(query, k=3)
            print_search_results(query, results)

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")

    # 9. Cleanup
    print("\n" + "=" * 80)
    print("Cleanup")
    print("=" * 80)

    print("\nDeleting articles...")
    vectorstore.delete(ids)
    print("✅ Articles deleted")

    print("\n" + "=" * 80)
    print("💡 Key Advantages of Semantic Search")
    print("=" * 80)
    print("""
    ✅ Understands Meaning:
       - Finds relevant docs even without keyword matches
       - Handles synonyms and paraphrasing naturally
       - Better than traditional full-text search

    ✅ Ranking by Relevance:
       - Returns most semantically similar results first
       - Cosine similarity scores show relevance
       - Filter by metadata for precise targeting

    ✅ Cross-Lingual Potential:
       - Embeddings can capture meaning across languages
       - Search in English, find results in other languages
       - (Requires multilingual embedding model)

    💡 Use Cases:
    - Internal knowledge bases / documentation search
    - E-commerce product discovery
    - Content recommendation engines
    - Research paper exploration
    - Customer support ticket routing
    - Code search and discovery

    💡 Production Tips:
    1. Use hybrid search for best of both worlds
    2. Add metadata filters for domain-specific search
    3. Cache popular queries for performance
    4. Implement query expansion for better coverage
    5. Track search analytics to improve relevance
    6. Update embeddings as content changes
    """)


if __name__ == "__main__":
    main()
