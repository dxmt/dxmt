"""Basic example of DXMT VectorDB integration with AutoGen.

This example demonstrates:
1. Configuring DXMT memory backend
2. Adding memories
3. Querying memories with similarity search
4. Using memory with an AutoGen agent

Requirements:
    pip install autogen-ext dxmt-python-sdk openai

Before running:
    1. Start DXMT server: cargo run -- serve --host 0.0.0.0 --port 9999
    2. Set OPENAI_API_KEY environment variable (if using OpenAI models)
"""

import asyncio
import sys
from pathlib import Path

# Add DXMT SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dxmt-python-sdk"))

# Import AutoGen components
try:
    from autogen_core.memory import MemoryContent
    from autogen_ext.memory.dxmt import DXMTVectorMemory, HttpDXMTVectorMemoryConfig
except ImportError:
    print("Error: AutoGen not installed. Install with:")
    print("  pip install autogen-ext autogen-agentchat")
    sys.exit(1)


async def example_basic_memory():
    """Example 1: Basic memory operations."""
    print("=" * 80)
    print("Example 1: Basic Memory Operations")
    print("=" * 80)

    # Configure DXMT memory with SentenceTransformers (no API key needed)
    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="basic_example_memory",
        dimension=384,  # all-MiniLM-L6-v2 dimension
        k=3,  # Return top 3 results
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        score_threshold=0.1  # Filter out low-relevance results
    )

    # Alternative: Use OpenAI embeddings (requires OPENAI_API_KEY)
    # config = HttpDXMTVectorMemoryConfig(
    #     host="localhost",
    #     port=9999,
    #     table_name="basic_example_memory",
    #     dimension=1536,
    #     embedding_provider="openai",
    #     embedding_model="text-embedding-3-small",
    #     k=3,
    #     score_threshold=0.1
    # )

    memory = DXMTVectorMemory(config=config)

    # Add some memories
    print("\n📝 Adding memories...")

    memories_to_add = [
        "User's name is Alice and she lives in San Francisco",
        "Alice is a software engineer working on AI systems",
        "Alice's favorite programming language is Python",
        "Alice enjoys hiking in the mountains on weekends",
        "Alice is learning about large language models"
    ]

    for content in memories_to_add:
        await memory.add(MemoryContent(
            content=content,
            mime_type="text/plain",
            metadata={"user_id": "alice_123", "source": "conversation"}
        ))
        print(f"  ✓ Added: {content}")

    # Query memories
    print("\n🔍 Querying memories...")

    queries = [
        "What is the user's name?",
        "What does Alice do for work?",
        "What are Alice's hobbies?",
    ]

    for query in queries:
        print(f"\n  Query: {query}")
        results = await memory.query(query, user_id="alice_123")

        if results.results:
            for i, result in enumerate(results.results, 1):
                score = result.metadata.get("score", 0.0)
                print(f"    {i}. {result.content}")
                print(f"       (relevance: {score:.3f})")
        else:
            print("    No results found")

    # Clean up
    await memory.clear()
    await memory.close()
    print("\n✅ Example 1 completed!")


async def example_multi_user_memory():
    """Example 2: Multi-user memory with filtering."""
    print("\n" + "=" * 80)
    print("Example 2: Multi-User Memory with Filtering")
    print("=" * 80)

    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="multi_user_memory",
        k=5
    )

    memory = DXMTVectorMemory(config=config)

    # Add memories for different users
    print("\n📝 Adding memories for multiple users...")

    user_memories = [
        ("alice", "Alice prefers Python over Java"),
        ("alice", "Alice's favorite framework is FastAPI"),
        ("bob", "Bob is an expert in Rust programming"),
        ("bob", "Bob enjoys working with systems programming"),
        ("charlie", "Charlie specializes in frontend development"),
        ("charlie", "Charlie uses React and TypeScript daily"),
    ]

    for user_id, content in user_memories:
        await memory.add(MemoryContent(
            content=content,
            mime_type="text/plain",
            metadata={"user_id": user_id}
        ))
        print(f"  ✓ [{user_id}] {content}")

    # Query with user filtering
    print("\n🔍 Querying with user filtering...")

    test_queries = [
        ("alice", "What programming language?"),
        ("bob", "What does the user know?"),
        ("charlie", "What technologies?"),
    ]

    for user_id, query in test_queries:
        print(f"\n  User: {user_id}, Query: {query}")
        results = await memory.query(query, user_id=user_id, limit=2)

        for i, result in enumerate(results.results, 1):
            score = result.metadata.get("score", 0.0)
            print(f"    {i}. {result.content} (score: {score:.3f})")

    # Clean up
    await memory.clear()
    await memory.close()
    print("\n✅ Example 2 completed!")


async def example_with_agent():
    """Example 3: Using DXMT memory with an AutoGen agent."""
    print("\n" + "=" * 80)
    print("Example 3: DXMT Memory with AutoGen Agent")
    print("=" * 80)

    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_ext.models.openai import OpenAIChatCompletionClient
    except ImportError:
        print("\n⚠️  Skipping Example 3: autogen-agentchat not installed")
        print("   Install with: pip install autogen-agentchat")
        return

    # Configure memory
    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="agent_memory",
        k=3,
        enable_full_text_search=True
    )

    memory = DXMTVectorMemory(config=config)

    # Add some background knowledge
    print("\n📝 Adding background knowledge to memory...")

    knowledge = [
        "The user's company uses Python and Django for backend services",
        "The team is migrating from PostgreSQL to a distributed database",
        "The user's main project is a real-time analytics platform",
        "The team follows agile methodology with 2-week sprints",
    ]

    for content in knowledge:
        await memory.add(MemoryContent(
            content=content,
            mime_type="text/plain",
            metadata={"source": "company_info"}
        ))
        print(f"  ✓ {content}")

    # Create agent with memory
    print("\n🤖 Creating agent with DXMT memory...")

    try:
        model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini"
        )

        agent = AssistantAgent(
            name="helpful_assistant",
            model_client=model_client,
            memory=[memory],
            system_message=(
                "You are a helpful assistant with access to memory. "
                "Use the provided memories to give informed responses."
            )
        )

        # Test queries
        print("\n💬 Testing agent with memory-augmented responses...")

        test_tasks = [
            "What technology stack does my company use?",
            "Tell me about our database migration",
        ]

        for task in test_tasks:
            print(f"\n  User: {task}")
            result = await agent.run(task=task)
            print(f"  Agent: {result.messages[-1].content}")

    except Exception as e:
        print(f"\n⚠️  Could not create agent: {e}")
        print("   Make sure OPENAI_API_KEY is set")

    # Clean up
    await memory.clear()
    await memory.close()
    print("\n✅ Example 3 completed!")


async def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("DXMT VectorDB + AutoGen Integration Examples")
    print("=" * 80)

    try:
        await example_basic_memory()
        await example_multi_user_memory()
        await example_with_agent()

        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
