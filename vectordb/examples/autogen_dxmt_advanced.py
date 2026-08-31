"""Advanced features example for DXMT VectorDB + AutoGen.

This example demonstrates:
1. Hybrid search (vector + text)
2. Score threshold filtering
3. Custom metadata and complex filtering
4. Performance tuning and optimization
5. Error handling and recovery

Requirements:
    pip install autogen-ext autogen-agentchat dxmt-python-sdk

Before running:
    Start DXMT server: cargo run -- serve --host 0.0.0.0 --port 9999
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add DXMT SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dxmt-python-sdk"))

try:
    from autogen_core.memory import MemoryContent
    from autogen_ext.memory.dxmt import DXMTVectorMemory, HttpDXMTVectorMemoryConfig
except ImportError:
    print("Error: AutoGen not installed. Install with:")
    print("  pip install autogen-ext")
    sys.exit(1)


async def example_hybrid_search():
    """Example 1: Hybrid search configuration."""
    print("=" * 80)
    print("Example 1: Hybrid Search (Vector + Text)")
    print("=" * 80)

    # Configure with hybrid search enabled
    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="hybrid_search_demo",
        k=5,
        enable_full_text_search=True,
        enable_hybrid_search=False,  # Will use full-text for now
        hybrid_vector_weight=0.7  # 70% vector, 30% text
    )

    memory = DXMTVectorMemory(config=config)

    print("\n📝 Adding technical documents...\n")

    documents = [
        {
            "content": "Python is an interpreted, high-level programming language with dynamic semantics",
            "metadata": {"category": "programming", "language": "python", "difficulty": "beginner"}
        },
        {
            "content": "Machine learning algorithms learn patterns from data without explicit programming",
            "metadata": {"category": "AI", "language": "python", "difficulty": "advanced"}
        },
        {
            "content": "FastAPI is a modern web framework for building APIs with Python based on type hints",
            "metadata": {"category": "web", "language": "python", "difficulty": "intermediate"}
        },
        {
            "content": "Docker containers package applications with dependencies for consistent deployment",
            "metadata": {"category": "devops", "language": "any", "difficulty": "intermediate"}
        },
        {
            "content": "PostgreSQL is a powerful open-source relational database system",
            "metadata": {"category": "database", "language": "SQL", "difficulty": "intermediate"}
        },
    ]

    for doc in documents:
        await memory.add(MemoryContent(
            content=doc["content"],
            mime_type="text/plain",
            metadata=doc["metadata"]
        ))
        category = doc["metadata"]["category"]
        print(f"  ✓ [{category}] {doc['content'][:60]}...")

    # Test different search approaches
    print("\n🔍 Testing search approaches...\n")

    query = "Python web development frameworks"

    # Full-text search (currently the default)
    print(f"Query: {query}\n")
    print("Approach: Full-text search (BM25)")

    results = await memory.query(query, limit=3)

    print(f"Found {len(results.results)} results:\n")
    for i, result in enumerate(results.results, 1):
        score = result.metadata.get("score", 0.0)
        category = result.metadata.get("category", "unknown")
        print(f"  {i}. [{category}] {result.content[:70]}...")
        print(f"     Score: {score:.3f}")
        print()

    print("💡 Note: Hybrid search (vector + text) requires embedding model configuration.")
    print("   Current implementation uses full-text search with Tantivy + BM25.")

    # Clean up
    await memory.clear()
    await memory.close()
    print("✅ Example 1 completed!")


async def example_score_threshold():
    """Example 2: Score threshold filtering."""
    print("\n" + "=" * 80)
    print("Example 2: Score Threshold Filtering")
    print("=" * 80)

    # Configure with different score thresholds
    thresholds = [None, 0.3, 0.5, 0.7]

    for threshold in thresholds:
        print(f"\n--- Score Threshold: {threshold if threshold else 'None (all results)'} ---\n")

        config = HttpDXMTVectorMemoryConfig(
            host="localhost",
            port=9999,
            table_name=f"threshold_test_{str(threshold).replace('.', '_') if threshold else 'none'}",
            k=10,
            score_threshold=threshold
        )

        memory = DXMTVectorMemory(config=config)

        # Add varied content
        if threshold == thresholds[0]:  # Only add once
            print("📝 Adding test content...\n")
            contents = [
                "Python is great for data science and machine learning",
                "JavaScript is used for web development",
                "Rust is a systems programming language",
                "Go is designed for cloud infrastructure",
                "Java is popular in enterprise applications",
            ]

            for content in contents:
                await memory.add(MemoryContent(
                    content=content,
                    mime_type="text/plain"
                ))

        # Query
        query = "programming languages for data analysis"
        print(f"Query: {query}")

        results = await memory.query(query)

        print(f"Results: {len(results.results)} memories returned\n")

        for i, result in enumerate(results.results, 1):
            score = result.metadata.get("score", 0.0)
            print(f"  {i}. {result.content[:50]}... (score: {score:.3f})")

        await memory.clear()
        await memory.close()

    print("\n✅ Example 2 completed!")


async def example_complex_metadata():
    """Example 3: Complex metadata and filtering."""
    print("\n" + "=" * 80)
    print("Example 3: Complex Metadata and Filtering")
    print("=" * 80)

    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="complex_metadata",
        k=10
    )

    memory = DXMTVectorMemory(config=config)

    print("\n📝 Adding memories with rich metadata...\n")

    # Simulate a project management system
    now = datetime.utcnow()

    tasks = [
        {
            "content": "Implement user authentication with OAuth2",
            "metadata": {
                "user_id": "alice",
                "project": "web_app",
                "priority": "high",
                "status": "in_progress",
                "assigned_to": "bob",
                "created_at": (now - timedelta(days=5)).isoformat(),
            }
        },
        {
            "content": "Write unit tests for payment processing",
            "metadata": {
                "user_id": "alice",
                "project": "web_app",
                "priority": "high",
                "status": "pending",
                "assigned_to": "charlie",
                "created_at": (now - timedelta(days=3)).isoformat(),
            }
        },
        {
            "content": "Update documentation for REST API endpoints",
            "metadata": {
                "user_id": "bob",
                "project": "web_app",
                "priority": "medium",
                "status": "completed",
                "assigned_to": "alice",
                "created_at": (now - timedelta(days=7)).isoformat(),
            }
        },
        {
            "content": "Optimize database queries for dashboard",
            "metadata": {
                "user_id": "charlie",
                "project": "analytics",
                "priority": "high",
                "status": "in_progress",
                "assigned_to": "alice",
                "created_at": (now - timedelta(days=2)).isoformat(),
            }
        },
        {
            "content": "Design new landing page mockups",
            "metadata": {
                "user_id": "alice",
                "project": "marketing",
                "priority": "low",
                "status": "pending",
                "assigned_to": "bob",
                "created_at": (now - timedelta(days=1)).isoformat(),
            }
        },
    ]

    for task in tasks:
        await memory.add(MemoryContent(
            content=task["content"],
            mime_type="text/plain",
            metadata=task["metadata"]
        ))
        priority = task["metadata"]["priority"]
        status = task["metadata"]["status"]
        print(f"  ✓ [{priority}] [{status}] {task['content']}")

    # Test various queries with metadata filtering
    print("\n🔍 Testing queries with metadata filtering...\n")

    queries = [
        {
            "desc": "All tasks for user 'alice'",
            "query": "tasks",
            "filter": {"user_id": "alice"}
        },
        {
            "desc": "High priority tasks",
            "query": "high priority",
            "filter": {}
        },
        {
            "desc": "Tasks in 'web_app' project",
            "query": "web application tasks",
            "filter": {}  # Note: project not in supported filter fields yet
        },
    ]

    for q in queries:
        print(f"Query: {q['desc']}")
        print(f"  Search: '{q['query']}'")
        print(f"  Filter: {q['filter']}")

        results = await memory.query(q["query"], limit=5, **q["filter"])

        print(f"  Results: {len(results.results)}\n")

        for i, result in enumerate(results.results, 1):
            meta = result.metadata
            priority = meta.get("priority", "?")
            status = meta.get("status", "?")
            project = meta.get("project", "?")
            print(f"    {i}. [{project}] [{priority}] [{status}]")
            print(f"       {result.content}")
        print()

    # Clean up
    await memory.clear()
    await memory.close()
    print("✅ Example 3 completed!")


async def example_performance_tuning():
    """Example 4: Performance tuning and optimization."""
    print("\n" + "=" * 80)
    print("Example 4: Performance Tuning")
    print("=" * 80)

    print("\n📊 Testing different configuration options...\n")

    # Test 1: Default configuration
    print("Test 1: Default configuration")
    config1 = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="perf_test_default",
        k=5
    )

    memory1 = DXMTVectorMemory(config=config1)

    import time

    # Add some data
    start = time.time()
    for i in range(50):
        await memory1.add(MemoryContent(
            content=f"Test document number {i} about various topics",
            mime_type="text/plain",
            metadata={"index": i}
        ))
    insert_time = time.time() - start

    # Query
    start = time.time()
    results = await memory1.query("test document topics", limit=5)
    query_time = time.time() - start

    print(f"  Insert 50 docs: {insert_time:.2f}s ({50/insert_time:.1f} docs/s)")
    print(f"  Query latency: {query_time*1000:.1f}ms")
    print(f"  Results: {len(results.results)}")

    await memory1.clear()
    await memory1.close()

    # Test 2: Optimized configuration
    print("\nTest 2: Optimized configuration (higher timeout, lower k)")
    config2 = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="perf_test_optimized",
        k=3,  # Fewer results
        timeout=60,  # Higher timeout
        score_threshold=0.3  # Filter low scores
    )

    memory2 = DXMTVectorMemory(config=config2)

    # Add same data
    start = time.time()
    for i in range(50):
        await memory2.add(MemoryContent(
            content=f"Test document number {i} about various topics",
            mime_type="text/plain",
            metadata={"index": i}
        ))
    insert_time = time.time() - start

    # Query
    start = time.time()
    results = await memory2.query("test document topics", limit=3)
    query_time = time.time() - start

    print(f"  Insert 50 docs: {insert_time:.2f}s ({50/insert_time:.1f} docs/s)")
    print(f"  Query latency: {query_time*1000:.1f}ms")
    print(f"  Results: {len(results.results)}")

    await memory2.clear()
    await memory2.close()

    # Performance tips
    print("\n💡 Performance Optimization Tips:\n")
    print("1. Configuration:")
    print("   - Lower 'k' value returns fewer results (faster)")
    print("   - Use 'score_threshold' to filter low-quality results")
    print("   - Increase 'timeout' for large datasets")
    print()
    print("2. DXMT Server:")
    print("   - Enable tiered storage (DB_STORAGE_TYPE=tiered)")
    print("   - Configure memory cache size (DB_MEMORY_CACHE_SIZE)")
    print("   - Use S3 backend for cost-efficient persistence")
    print()
    print("3. Queries:")
    print("   - Use metadata filters to narrow search scope")
    print("   - Enable full-text search for text-heavy queries")
    print("   - Batch operations when possible")

    print("\n✅ Example 4 completed!")


async def example_error_handling():
    """Example 5: Error handling and recovery."""
    print("\n" + "=" * 80)
    print("Example 5: Error Handling and Recovery")
    print("=" * 80)

    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="error_test",
        k=5
    )

    memory = DXMTVectorMemory(config=config)

    print("\n🛡️ Testing error handling scenarios...\n")

    # Test 1: Empty query
    print("Test 1: Empty query")
    try:
        results = await memory.query("")
        print(f"  ✓ Handled gracefully, returned {len(results.results)} results")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Test 2: Query with no results
    print("\nTest 2: Query with very specific term (likely no results)")
    try:
        await memory.add(MemoryContent(
            content="Python programming tutorial",
            mime_type="text/plain"
        ))

        results = await memory.query("xyzabc123nonexistent")
        print(f"  ✓ Handled gracefully, returned {len(results.results)} results")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Test 3: Clear empty table
    print("\nTest 3: Clear empty table")
    try:
        await memory.clear()
        await memory.clear()  # Clear again
        print("  ✓ Handled gracefully")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Test 4: Multiple close calls
    print("\nTest 4: Multiple close calls")
    try:
        await memory.close()
        await memory.close()  # Close again
        print("  ✓ Handled gracefully")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Best practices
    print("\n💡 Error Handling Best Practices:\n")
    print("1. Always wrap memory operations in try-except blocks")
    print("2. Check if results.results is empty before processing")
    print("3. Use score_threshold to filter unreliable results")
    print("4. Log errors for debugging")
    print("5. Implement retry logic for transient failures")

    print("\n✅ Example 5 completed!")


async def example_memory_patterns():
    """Example 6: Common memory usage patterns."""
    print("\n" + "=" * 80)
    print("Example 6: Common Memory Usage Patterns")
    print("=" * 80)

    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="patterns_demo",
        k=5
    )

    memory = DXMTVectorMemory(config=config)

    # Pattern 1: Conversation History
    print("\n📝 Pattern 1: Conversation History\n")

    conversation = [
        ("user", "What is machine learning?"),
        ("assistant", "Machine learning is a subset of AI that enables systems to learn from data"),
        ("user", "Can you give me an example?"),
        ("assistant", "Sure! Spam email detection is a common ML application"),
    ]

    for role, message in conversation:
        await memory.add(MemoryContent(
            content=f"{role}: {message}",
            mime_type="text/plain",
            metadata={"role": role, "type": "conversation"}
        ))

    print("Added 4 conversation turns")

    # Query conversation
    results = await memory.query("machine learning examples", limit=2)
    print(f"\nRecall: {len(results.results)} relevant messages")
    for result in results.results:
        print(f"  → {result.content}")

    # Pattern 2: Document Chunks
    print("\n\n📄 Pattern 2: Document Chunks\n")

    doc_chunks = [
        "Introduction: This guide covers Python best practices for beginners",
        "Chapter 1: Variables and data types in Python are dynamically typed",
        "Chapter 2: Functions should follow the single responsibility principle",
        "Chapter 3: Error handling using try-except blocks is essential",
        "Conclusion: Following these practices leads to maintainable code",
    ]

    for i, chunk in enumerate(doc_chunks):
        await memory.add(MemoryContent(
            content=chunk,
            mime_type="text/plain",
            metadata={"doc_id": "python_guide", "chunk_index": i, "type": "document"}
        ))

    print("Added 5 document chunks")

    # Query document
    results = await memory.query("error handling in Python", limit=2)
    print(f"\nRelevant chunks: {len(results.results)}")
    for result in results.results:
        print(f"  → {result.content}")

    # Pattern 3: User Preferences
    print("\n\n⚙️ Pattern 3: User Preferences\n")

    preferences = [
        "User prefers dark mode for all applications",
        "User's preferred programming language is Python",
        "User wants email notifications disabled",
        "User's timezone is UTC-8 (Pacific)",
    ]

    for pref in preferences:
        await memory.add(MemoryContent(
            content=pref,
            mime_type="text/plain",
            metadata={"user_id": "user_123", "type": "preference"}
        ))

    print("Added 4 user preferences")

    # Query preferences
    results = await memory.query("user settings and preferences", limit=4, user_id="user_123")
    print(f"\nUser preferences: {len(results.results)}")
    for result in results.results:
        print(f"  → {result.content}")

    # Clean up
    await memory.clear()
    await memory.close()

    print("\n✅ Example 6 completed!")


async def main():
    """Run all advanced examples."""
    print("\n" + "=" * 80)
    print("DXMT VectorDB Advanced Features Examples")
    print("=" * 80)

    try:
        await example_hybrid_search()
        await example_score_threshold()
        await example_complex_metadata()
        await example_performance_tuning()
        await example_error_handling()
        await example_memory_patterns()

        print("\n" + "=" * 80)
        print("✅ All advanced examples completed successfully!")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
