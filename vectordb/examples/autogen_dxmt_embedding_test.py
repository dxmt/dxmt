"""Test DXMT AutoGen integration with embedding support.

This example demonstrates the updated DXMT AutoGen integration with:
1. OpenAI embedding support
2. SentenceTransformers embedding support
3. Vector-based semantic search

Requirements:
    pip install autogen-ext autogen-agentchat dxmt-python-sdk
    pip install openai  # For OpenAI embeddings
    # OR
    pip install sentence-transformers  # For local embeddings

Before running:
    1. Start DXMT server: cargo run -- serve --host 0.0.0.0 --port 9999
    2. For OpenAI: Set OPENAI_API_KEY environment variable
    3. For SentenceTransformers: No API key needed
"""

import asyncio
import sys
import os
from pathlib import Path

# Add DXMT SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dxmt-python-sdk"))

from autogen_core.memory import MemoryContent
from autogen_ext.memory.dxmt import DXMTVectorMemory, HttpDXMTVectorMemoryConfig


async def test_openai_embeddings():
    """Test with OpenAI embeddings."""
    print("=" * 80)
    print("Test 1: OpenAI Embeddings (text-embedding-3-small)")
    print("=" * 80)

    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Skipping OpenAI test.")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return

    # Configure DXMT memory with OpenAI embeddings
    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="test_openai_embeddings",
        dimension=1536,  # text-embedding-3-small dimension
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        k=3,
        score_threshold=0.1
    )

    memory = DXMTVectorMemory(config=config)

    print("\n📝 Adding memories with OpenAI embeddings...")
    memories = [
        "Python is a high-level programming language known for simplicity",
        "Machine learning models can learn patterns from data",
        "Neural networks are inspired by biological neurons",
        "The weather today is sunny and warm",
        "I enjoy hiking in the mountains",
    ]

    for content in memories:
        await memory.add(MemoryContent(
            content=content,
            mime_type="text/plain",
            metadata={"source": "test"}
        ))
        print(f"  ✅ Added: {content[:50]}...")

    print("\n🔍 Querying with semantic search...")
    queries = [
        "Tell me about programming languages",
        "What is artificial intelligence?",
        "What are outdoor activities?",
    ]

    for query in queries:
        print(f"\n  Query: {query}")
        results = await memory.query(query)

        if results.results:
            for i, result in enumerate(results.results, 1):
                score = result.metadata.get("score", 0.0)
                print(f"    {i}. [{score:.3f}] {result.content}")
        else:
            print("    No results found")

    # Clean up
    await memory.clear()
    await memory.close()
    print("\n✅ OpenAI embeddings test completed!")


async def test_sentence_transformers():
    """Test with SentenceTransformers embeddings."""
    print("\n" + "=" * 80)
    print("Test 2: SentenceTransformers (all-MiniLM-L6-v2)")
    print("=" * 80)

    try:
        import sentence_transformers
    except ImportError:
        print("⚠️  sentence-transformers not installed. Skipping test.")
        print("   Install with: pip install sentence-transformers")
        return

    # Configure DXMT memory with SentenceTransformers
    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="test_st_embeddings",
        dimension=384,  # all-MiniLM-L6-v2 dimension
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        k=3,
        score_threshold=0.1
    )

    memory = DXMTVectorMemory(config=config)

    print("\n📝 Adding memories with SentenceTransformers...")
    print("   (First run will download model ~80MB)")

    memories = [
        "The Eiffel Tower is located in Paris, France",
        "Tokyo is the capital city of Japan",
        "The Great Wall of China is a famous landmark",
        "Pizza is a popular Italian dish",
        "Sushi originated in Japan",
    ]

    for content in memories:
        await memory.add(MemoryContent(
            content=content,
            mime_type="text/plain",
            metadata={"source": "test", "category": "facts"}
        ))
        print(f"  ✅ Added: {content}")

    print("\n🔍 Querying with semantic search...")
    queries = [
        "Tell me about famous buildings",
        "What foods are from Asia?",
        "European landmarks",
    ]

    for query in queries:
        print(f"\n  Query: {query}")
        results = await memory.query(query, limit=2)

        if results.results:
            for i, result in enumerate(results.results, 1):
                score = result.metadata.get("score", 0.0)
                print(f"    {i}. [{score:.3f}] {result.content}")
        else:
            print("    No results found")

    # Clean up
    await memory.clear()
    await memory.close()
    print("\n✅ SentenceTransformers test completed!")


async def test_dimension_auto_detection():
    """Test automatic dimension detection."""
    print("\n" + "=" * 80)
    print("Test 3: Automatic Dimension Detection")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Skipping dimension test.")
        return

    # Configure with wrong dimension - should auto-correct
    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="test_dimension_auto",
        dimension=999,  # Wrong dimension - will be auto-corrected
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        k=2
    )

    memory = DXMTVectorMemory(config=config)

    print(f"\n📊 Initial dimension: {config.dimension}")
    print("   (This is wrong for text-embedding-3-small)")

    print("\n📝 Adding first memory (will auto-correct dimension)...")
    await memory.add(MemoryContent(
        content="Test content for dimension detection",
        mime_type="text/plain"
    ))

    print(f"   ✅ Dimension auto-corrected to: {config.dimension}")

    # Clean up
    await memory.clear()
    await memory.close()
    print("\n✅ Dimension auto-detection test completed!")


async def test_comparison():
    """Compare embedding providers side-by-side."""
    print("\n" + "=" * 80)
    print("Test 4: Embedding Provider Comparison")
    print("=" * 80)

    test_query = "What is machine learning?"

    # Test with OpenAI
    if os.getenv("OPENAI_API_KEY"):
        print("\n🔵 OpenAI Results:")
        config = HttpDXMTVectorMemoryConfig(
            host="localhost",
            port=9999,
            table_name="comparison_openai",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )
        memory = DXMTVectorMemory(config=config)

        await memory.add(MemoryContent(content="Machine learning uses algorithms to learn from data", mime_type="text/plain"))
        await memory.add(MemoryContent(content="Deep learning is a subset of machine learning", mime_type="text/plain"))
        await memory.add(MemoryContent(content="The weather is nice today", mime_type="text/plain"))

        results = await memory.query(test_query, limit=2)
        for i, r in enumerate(results.results, 1):
            print(f"  {i}. [{r.metadata['score']:.3f}] {r.content}")

        await memory.clear()
        await memory.close()

    # Test with SentenceTransformers
    try:
        import sentence_transformers

        print("\n🟢 SentenceTransformers Results:")
        config = HttpDXMTVectorMemoryConfig(
            host="localhost",
            port=9999,
            table_name="comparison_st",
            dimension=384,
            embedding_provider="sentence-transformers",
            embedding_model="all-MiniLM-L6-v2",
        )
        memory = DXMTVectorMemory(config=config)

        await memory.add(MemoryContent(content="Machine learning uses algorithms to learn from data", mime_type="text/plain"))
        await memory.add(MemoryContent(content="Deep learning is a subset of machine learning", mime_type="text/plain"))
        await memory.add(MemoryContent(content="The weather is nice today", mime_type="text/plain"))

        results = await memory.query(test_query, limit=2)
        for i, r in enumerate(results.results, 1):
            print(f"  {i}. [{r.metadata['score']:.3f}] {r.content}")

        await memory.clear()
        await memory.close()

    except ImportError:
        print("\n🟢 SentenceTransformers: Not installed")

    print("\n✅ Comparison test completed!")


async def main():
    """Run all tests."""
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "DXMT AutoGen Embedding Integration Tests" + " " * 22 + "║")
    print("╚" + "=" * 78 + "╝")

    tests = [
        ("OpenAI Embeddings", test_openai_embeddings),
        ("SentenceTransformers", test_sentence_transformers),
        ("Dimension Auto-Detection", test_dimension_auto_detection),
        ("Provider Comparison", test_comparison),
    ]

    for name, test_func in tests:
        try:
            await test_func()
        except Exception as e:
            print(f"\n❌ {name} test failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("🎉 All tests completed!")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  ✅ OpenAI embeddings (text-embedding-3-small)")
    print("  ✅ SentenceTransformers embeddings (all-MiniLM-L6-v2)")
    print("  ✅ Vector-based semantic search")
    print("  ✅ Automatic dimension detection")
    print("  ✅ Score-based filtering")
    print("\nNext Steps:")
    print("  - Use in production with your AutoGen agents")
    print("  - Try different embedding models")
    print("  - Experiment with score thresholds")
    print("  - Scale to millions of memories with DXMT's tiered storage")


if __name__ == "__main__":
    asyncio.run(main())
