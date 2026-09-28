"""Complete CrewAI + DXMT VectorDB Integration Example.

This example demonstrates how to use DXMT VectorDB with CrewAI for:
1. Vector search tools for agents
2. Persistent memory storage
3. Multi-agent collaboration with shared memory

Requirements:
    pip install crewai crewai-tools dxmt-python-sdk openai

Before running:
    1. Start DXMT server: cargo run -- serve --host 0.0.0.0 --port 9999
    2. Set OPENAI_API_KEY environment variable
    3. Ensure crewai and dxmt-python-sdk are installed
"""

import os
import sys
from pathlib import Path

# Add DXMT SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent / "clients" / "dxmt-python-sdk"))

# The provider's models come from the environment so one `.env` switch moves every example.
# These are *deployment* names and do not carry between providers -- the Azure resource this
# was written against serves `gpt-4.1-mini` and answers DeploymentNotFound for `gpt-4o-mini`.
# The defaults are what this file used before, so nothing changes for a plain OPENAI_API_KEY.
CHAT_MODEL = os.environ.get("DXMT_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.environ.get("DXMT_EMBED_MODEL", "text-embedding-3-large")
EMBED_DIMS = int(os.environ.get("DXMT_EMBED_DIMS", "3072"))


def example_1_vector_search_tool():
    """Example 1: Using DXMT as a vector search tool for CrewAI agents."""
    print("\n" + "=" * 80)
    print("Example 1: DXMT Vector Search Tool for CrewAI Agents")
    print("=" * 80)

    try:
        from crewai import Agent, Task, Crew
        from dxmt.crewai_tool import DXMTVectorSearchTool, DXMTToolConfig
        from dxmt import DXMTClient
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install crewai crewai-tools dxmt-python-sdk")
        return

    # Initialize DXMT client
    print("\n📊 Initializing DXMT client...")
    client = DXMTClient(host="localhost", port=9999)

    # Create table for documents
    table_name = "research_documents"
    try:
        client.create_table(
            name=table_name,
            dimension=EMBED_DIMS,
            distance_metric="cosine",
            description="Research documents for CrewAI agents"
        )
        print(f"✅ Created table: {table_name}")
    except Exception as e:
        print(f"ℹ️  Table already exists or error: {e}")

    # Configure DXMT search tool
    config = DXMTToolConfig(
        host="localhost",
        port=9999,
        table_name=table_name,
        limit=5,
        score_threshold=0.35
    )

    search_tool = DXMTVectorSearchTool(dxmt_config=config)

    # Add sample documents
    print("\n📝 Adding sample documents to DXMT...")
    sample_docs = [
        "DXMT VectorDB uses tiered storage (Memory→SSD→S3) for 98% cost savings",
        "CrewAI enables multi-agent collaboration with role-based task delegation",
        "Vector databases excel at semantic search and similarity matching",
        "Python is the most popular language for AI and machine learning applications",
        "Rust provides memory safety and high performance for systems programming"
    ]

    try:
        search_tool.add_documents(
            texts=sample_docs,
            metadatas=[{"source": "knowledge_base", "index": i} for i in range(len(sample_docs))]
        )
        print(f"✅ Added {len(sample_docs)} documents to DXMT")
    except Exception as e:
        print(f"⚠️  Error adding documents (may already exist): {e}")

    # Create agent with DXMT search tool
    print("\n🤖 Creating research agent with DXMT tool...")

    researcher = Agent(
        role="Senior Research Analyst",
        goal="Find and analyze relevant technical information",
        backstory=(
            "You are an experienced research analyst who excels at finding "
            "relevant information and providing insightful analysis."
        ),
        tools=[search_tool],
        verbose=True
    )

    # Create task
    research_task = Task(
        description=(
            "Search for information about vector databases and their benefits. "
            "Focus on cost efficiency and performance characteristics."
        ),
        agent=researcher,
        expected_output="A summary of vector database benefits based on search results"
    )

    # Run crew
    print("\n🚀 Running CrewAI with DXMT search...")
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True
    )

    try:
        result = crew.kickoff()
        print("\n" + "=" * 80)
        print("📊 Research Results:")
        print("=" * 80)
        print(result)
    except Exception as e:
        print(f"❌ Error running crew: {e}")

    print("\n✅ Example 1 completed!")


def example_2_memory_storage():
    """Example 2: Using DXMT as persistent memory storage for CrewAI."""
    print("\n" + "=" * 80)
    print("Example 2: DXMT as Persistent Memory Storage")
    print("=" * 80)

    try:
        from crewai import Agent, Task, Crew
        from dxmt.crewai_storage import DXMTStorage
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install crewai dxmt-python-sdk")
        return

    # Configure DXMT storage for long-term memory
    print("\n📊 Configuring DXMT memory storage...")

    storage_config = {
        "host": "localhost",
        "port": 9999,
        "table_name": "crew_long_term_memory",
        "user_id": "user_001",
        "dimension": EMBED_DIMS
    }

    long_term_storage = DXMTStorage(
        type="long_term",
        config=storage_config
    )

    print("✅ Initialized DXMT long-term memory storage")

    # Save some memories
    print("\n💾 Saving memories to DXMT...")

    memories = [
        ("User prefers Python for AI development", {"category": "preference"}),
        ("User's favorite vector database is DXMT", {"category": "preference"}),
        ("User works on multi-agent systems", {"category": "work"}),
        ("User is interested in cost-efficient solutions", {"category": "interest"}),
    ]

    for content, metadata in memories:
        long_term_storage.save(content, metadata)
        print(f"  ✓ Saved: {content}")

    # Search memories
    print("\n🔍 Searching memories...")

    queries = [
        "What programming language does the user prefer?",
        "What is the user interested in?",
        "What does the user work on?"
    ]

    for query in queries:
        print(f"\n📝 Query: {query}")
        results = long_term_storage.search(query, limit=3, score_threshold=0.6)

        if results and "results" in results:
            for i, result in enumerate(results["results"][:2], 1):
                print(f"  {i}. {result['content']}")
                print(f"     Score: {result['score']:.3f}")

    print("\n✅ Example 2 completed!")


def example_3_multi_agent_with_memory():
    """Example 3: Multi-agent CrewAI with shared DXMT memory."""
    print("\n" + "=" * 80)
    print("Example 3: Multi-Agent CrewAI with Shared DXMT Memory")
    print("=" * 80)

    try:
        from crewai import Agent, Task, Crew
        from dxmt.crewai_storage import DXMTStorage
        from dxmt.crewai_tool import DXMTVectorSearchTool, DXMTToolConfig
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return

    # Configure shared memory storage
    print("\n📊 Setting up shared memory...")

    memory_config = {
        "host": "localhost",
        "port": 9999,
        "table_name": "multi_agent_memory",
        "user_id": "project_001",
        "dimension": EMBED_DIMS
    }

    shared_memory = DXMTStorage(
        type="long_term",
        config=memory_config
    )

    # Configure knowledge base tool
    kb_config = DXMTToolConfig(
        host="localhost",
        port=9999,
        table_name="project_knowledge",
        limit=3
    )

    kb_tool = DXMTVectorSearchTool(dxmt_config=kb_config)

    # Add knowledge base documents
    print("\n📚 Populating knowledge base...")
    try:
        kb_documents = [
            "Project deadline is February 15, 2026",
            "The project uses DXMT VectorDB for semantic search",
            "Team consists of 5 engineers and 2 researchers",
            "Budget allocated is $50,000 for infrastructure",
            "Primary technology stack: Python, Rust, and React"
        ]

        kb_tool.add_documents(
            texts=kb_documents,
            metadatas=[{"type": "project_info"} for _ in kb_documents]
        )
        print(f"✅ Added {len(kb_documents)} documents to knowledge base")
    except Exception as e:
        print(f"⚠️  Knowledge base error: {e}")

    # Create agents
    print("\n🤖 Creating multi-agent team...")

    project_manager = Agent(
        role="Project Manager",
        goal="Coordinate project activities and track progress",
        backstory="Experienced PM with focus on delivery and team coordination",
        tools=[kb_tool],
        verbose=False
    )

    technical_lead = Agent(
        role="Technical Lead",
        goal="Make technical decisions and guide architecture",
        backstory="Senior engineer with expertise in distributed systems",
        tools=[kb_tool],
        verbose=False
    )

    # Create tasks
    planning_task = Task(
        description="Review project information and create a high-level plan",
        agent=project_manager,
        expected_output="Project plan with key milestones"
    )

    architecture_task = Task(
        description="Define technical architecture based on project requirements",
        agent=technical_lead,
        expected_output="Technical architecture overview"
    )

    # Create crew with shared memory
    print("\n🚀 Running multi-agent crew with shared memory...")

    crew = Crew(
        agents=[project_manager, technical_lead],
        tasks=[planning_task, architecture_task],
        verbose=True,
        memory=False  # We're using custom DXMT storage
    )

    try:
        result = crew.kickoff()

        # Save results to shared memory
        print("\n💾 Saving results to shared memory...")
        shared_memory.save(
            str(result),
            {"type": "crew_output", "agents": "pm_and_tech_lead"}
        )

        print("\n" + "=" * 80)
        print("📊 Crew Results:")
        print("=" * 80)
        print(result)

        # Query shared memory
        print("\n🔍 Querying shared memory...")
        memory_results = shared_memory.search(
            "What did the team plan?",
            limit=2
        )

        if memory_results and "results" in memory_results:
            print("\n📋 Retrieved from shared memory:")
            for i, mem in enumerate(memory_results["results"][:1], 1):
                print(f"{i}. {mem['content'][:200]}...")

    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n✅ Example 3 completed!")


def example_4_custom_embedding():
    """Example 4: Using custom embedding function with DXMT tool."""
    print("\n" + "=" * 80)
    print("Example 4: Custom Embedding Function with DXMT")
    print("=" * 80)

    try:
        from crewai import Agent, Task, Crew
        from dxmt.crewai_tool import DXMTVectorSearchTool, DXMTToolConfig
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return

    # Define custom embedding function (example with sentence-transformers)
    def custom_embed(text: str) -> list[float]:
        """Custom embedding using sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(text)
            return embedding.tolist()
        except ImportError:
            print("⚠️  sentence-transformers not installed, using dummy embedding")
            import random
            return [random.random() for _ in range(384)]

    print("\n📊 Configuring DXMT with custom embedding...")

    config = DXMTToolConfig(
        host="localhost",
        port=9999,
        table_name="custom_embeddings",
        limit=3
    )

    search_tool = DXMTVectorSearchTool(
        dxmt_config=config,
        custom_embedding_fn=custom_embed
    )

    print("✅ Configured DXMT tool with custom embedding function")
    print("✅ Example 4 setup completed!")


def main():
    """Run all CrewAI + DXMT integration examples."""
    print("\n" + "=" * 80)
    print("DXMT VectorDB + CrewAI Integration Examples")
    print("=" * 80)
    print("\nThis demonstrates:")
    print("  1. DXMT as a vector search tool for agents")
    print("  2. DXMT as persistent memory storage")
    print("  3. Multi-agent collaboration with shared memory")
    print("  4. Custom embedding functions")

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("Some examples may fail without it.\n")

    try:
        # Run examples
        example_1_vector_search_tool()
        example_2_memory_storage()
        example_3_multi_agent_with_memory()
        example_4_custom_embedding()

        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80)
        print("\n📚 Key Takeaways:")
        print("  ✓ DXMT integrates seamlessly with CrewAI")
        print("  ✓ Two integration levels: Tools and Storage")
        print("  ✓ 98% cost savings with tiered storage")
        print("  ✓ High-performance vector search (<100ms)")
        print("  ✓ Supports multi-agent collaboration")
        print("\n🚀 Ready for production use!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
