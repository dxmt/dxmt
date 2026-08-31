"""Multi-agent shared memory example with DXMT VectorDB.

This example demonstrates:
1. Multiple agents sharing the same memory instance
2. Agent-specific memory scoping with agent_id
3. Collaborative workflows with shared context
4. Memory isolation and filtering

Requirements:
    pip install autogen-ext autogen-agentchat dxmt-python-sdk openai

Before running:
    1. Start DXMT server: cargo run -- serve --host 0.0.0.0 --port 9999
    2. Set OPENAI_API_KEY environment variable
"""

import asyncio
import sys
from pathlib import Path

# Add DXMT SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dxmt-python-sdk"))

try:
    from autogen_core.memory import MemoryContent
    from autogen_ext.memory.dxmt import DXMTVectorMemory, HttpDXMTVectorMemoryConfig
except ImportError:
    print("Error: AutoGen not installed. Install with:")
    print("  pip install autogen-ext autogen-agentchat")
    sys.exit(1)


async def example_shared_memory():
    """Example 1: Multiple agents with shared memory."""
    print("=" * 80)
    print("Example 1: Multiple Agents with Shared Memory")
    print("=" * 80)

    # Configure shared memory
    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="shared_agent_memory",
        k=3,
        enable_full_text_search=True
    )

    memory = DXMTVectorMemory(config=config)

    print("\n📝 Adding memories from different agents...\n")

    # Agent 1: Research Assistant
    research_memories = [
        "Research found that Python is the most popular language for AI",
        "Study shows 85% of data scientists prefer Jupyter notebooks",
        "Recent benchmarks indicate PyTorch is faster than TensorFlow for transformers",
    ]

    for content in research_memories:
        await memory.add(MemoryContent(
            content=content,
            mime_type="text/plain",
            metadata={"agent_id": "researcher", "source": "research"}
        ))
        print(f"  [Researcher] ✓ {content}")

    # Agent 2: Writer Assistant
    writer_memories = [
        "Blog post outline: Introduction to Machine Learning",
        "Article draft: Why Python dominates AI development",
        "Technical writing tip: Always explain jargon terms",
    ]

    for content in writer_memories:
        await memory.add(MemoryContent(
            content=content,
            mime_type="text/plain",
            metadata={"agent_id": "writer", "source": "writing"}
        ))
        print(f"  [Writer] ✓ {content}")

    # Agent 3: Code Reviewer
    reviewer_memories = [
        "Code review: Recommend using type hints in Python functions",
        "Best practice: Always write unit tests for new features",
        "Security note: Never hardcode API keys in source code",
    ]

    for content in reviewer_memories:
        await memory.add(MemoryContent(
            content=content,
            mime_type="text/plain",
            metadata={"agent_id": "reviewer", "source": "code_review"}
        ))
        print(f"  [Reviewer] ✓ {content}")

    # Query shared memory with agent filtering
    print("\n🔍 Querying shared memory with agent filtering...\n")

    queries = [
        ("researcher", "What are the findings about Python?"),
        ("writer", "What content should I write about?"),
        ("reviewer", "What are the code quality recommendations?"),
    ]

    for agent_id, query in queries:
        print(f"Agent: {agent_id}")
        print(f"Query: {query}")
        results = await memory.query(query, agent_id=agent_id, limit=2)

        if results.results:
            for i, result in enumerate(results.results, 1):
                score = result.metadata.get("score", 0.0)
                print(f"  {i}. {result.content}")
                print(f"     (score: {score:.3f})")
        else:
            print("  No results found")
        print()

    # Query without filtering - see all memories
    print("🌐 Querying without agent filter (global search)...\n")
    print("Query: Python programming")

    results = await memory.query("Python programming", limit=5)
    print(f"Found {len(results.results)} results across all agents:\n")

    for i, result in enumerate(results.results, 1):
        score = result.metadata.get("score", 0.0)
        agent = result.metadata.get("agent_id", "unknown")
        print(f"  {i}. [{agent}] {result.content}")
        print(f"     (score: {score:.3f})")

    # Clean up
    await memory.clear()
    await memory.close()
    print("\n✅ Example 1 completed!")


async def example_collaborative_workflow():
    """Example 2: Collaborative workflow with shared context."""
    print("\n" + "=" * 80)
    print("Example 2: Collaborative Workflow with Shared Context")
    print("=" * 80)

    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="collaborative_memory",
        k=5
    )

    memory = DXMTVectorMemory(config=config)

    print("\n📝 Simulating a collaborative project workflow...\n")

    # Phase 1: Requirements Gathering (Product Manager)
    print("Phase 1: Requirements Gathering")
    pm_memories = [
        "Project requirement: Build a RAG chatbot for customer support",
        "Target users: Customer service team handling 1000+ tickets/day",
        "Key features: Context-aware responses, multilingual support",
        "Technical constraints: Must integrate with existing Zendesk",
    ]

    for content in pm_memories:
        await memory.add(MemoryContent(
            content=content,
            mime_type="text/plain",
            metadata={
                "agent_id": "product_manager",
                "phase": "requirements",
                "project": "customer_support_bot"
            }
        ))
        print(f"  [PM] ✓ {content}")

    # Phase 2: Architecture Design (Architect)
    print("\nPhase 2: Architecture Design")

    # Architect queries requirements before designing
    print("  [Architect] Querying requirements...")
    req_results = await memory.query(
        "What are the project requirements?",
        agent_id="product_manager",
        limit=3
    )

    print(f"  [Architect] Found {len(req_results.results)} requirements")

    architect_memories = [
        "Architecture decision: Use DXMT for vector storage with S3 backend",
        "Tech stack: FastAPI + DXMT + OpenAI embeddings",
        "Deployment plan: Kubernetes on AWS with auto-scaling",
        "Integration approach: REST API with Zendesk webhooks",
    ]

    for content in architect_memories:
        await memory.add(MemoryContent(
            content=content,
            mime_type="text/plain",
            metadata={
                "agent_id": "architect",
                "phase": "architecture",
                "project": "customer_support_bot"
            }
        ))
        print(f"  [Architect] ✓ {content}")

    # Phase 3: Implementation (Developer)
    print("\nPhase 3: Implementation")

    # Developer queries architecture before coding
    print("  [Developer] Querying architecture decisions...")
    arch_results = await memory.query(
        "What is the technical architecture?",
        agent_id="architect",
        limit=3
    )

    print(f"  [Developer] Found {len(arch_results.results)} architecture decisions")

    dev_memories = [
        "Implementation: FastAPI endpoints created for chat and search",
        "Database: DXMT table configured with 1536-dim vectors",
        "Testing: Unit tests written with 85% coverage",
        "Documentation: API docs published to Swagger UI",
    ]

    for content in dev_memories:
        await memory.add(MemoryContent(
            content=content,
            mime_type="text/plain",
            metadata={
                "agent_id": "developer",
                "phase": "implementation",
                "project": "customer_support_bot"
            }
        ))
        print(f"  [Developer] ✓ {content}")

    # Cross-agent knowledge sharing
    print("\n🔄 Cross-agent knowledge sharing...\n")

    # Each agent can query the entire project context
    agents = ["product_manager", "architect", "developer"]

    for agent_id in agents:
        print(f"Agent: {agent_id}")
        print("Query: What is the status of the customer support bot project?")

        results = await memory.query(
            "customer support bot project status",
            limit=3  # No agent filter - see all phases
        )

        phases_seen = set()
        for result in results.results:
            phase = result.metadata.get("phase", "unknown")
            phases_seen.add(phase)

        print(f"  → Can see {len(results.results)} updates from phases: {', '.join(phases_seen)}")
        print()

    # Project summary query
    print("📊 Project Summary Query (all agents):\n")

    summary_queries = [
        "What are the key project requirements?",
        "What technology stack was chosen?",
        "What has been implemented?",
    ]

    for query in summary_queries:
        print(f"Query: {query}")
        results = await memory.query(query, limit=2)

        for result in results.results:
            agent = result.metadata.get("agent_id", "unknown")
            phase = result.metadata.get("phase", "unknown")
            print(f"  → [{agent}] [{phase}] {result.content[:80]}...")
        print()

    # Clean up
    await memory.clear()
    await memory.close()
    print("✅ Example 2 completed!")


async def example_session_based_memory():
    """Example 3: Session-based memory isolation."""
    print("\n" + "=" * 80)
    print("Example 3: Session-Based Memory Isolation")
    print("=" * 80)

    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="session_memory",
        k=3
    )

    memory = DXMTVectorMemory(config=config)

    print("\n📝 Adding memories for different sessions...\n")

    # Simulate three concurrent chat sessions
    sessions = {
        "session_001": [
            "User asked about Python data science libraries",
            "Recommended pandas, numpy, scikit-learn",
            "User is interested in machine learning",
        ],
        "session_002": [
            "User needs help with web development",
            "Suggested Django and FastAPI frameworks",
            "User prefers FastAPI for REST APIs",
        ],
        "session_003": [
            "User wants to learn Docker containers",
            "Explained Dockerfile and docker-compose",
            "User successfully deployed first container",
        ],
    }

    for session_id, messages in sessions.items():
        print(f"Session: {session_id}")
        for content in messages:
            await memory.add(MemoryContent(
                content=content,
                mime_type="text/plain",
                metadata={
                    "session_id": session_id,
                    "user_id": f"user_{session_id[-3:]}"  # Extract user from session
                }
            ))
            print(f"  ✓ {content}")
        print()

    # Query with session isolation
    print("🔍 Querying with session isolation...\n")

    for session_id in ["session_001", "session_002", "session_003"]:
        print(f"Session: {session_id}")
        print("Query: What did we discuss?")

        results = await memory.query(
            "What did we discuss?",
            session_id=session_id,
            limit=3
        )

        print(f"  Found {len(results.results)} memories:")
        for i, result in enumerate(results.results, 1):
            print(f"    {i}. {result.content}")
        print()

    # Cross-session query (without session filter)
    print("🌐 Cross-session query (no filter):\n")
    print("Query: user learning")

    results = await memory.query("user learning", limit=5)
    print(f"Found {len(results.results)} results across all sessions:\n")

    for result in results.results:
        session = result.metadata.get("session_id", "unknown")
        print(f"  [{session}] {result.content}")

    # Clean up
    await memory.clear()
    await memory.close()
    print("\n✅ Example 3 completed!")


async def example_with_real_agents():
    """Example 4: Real AutoGen agents with shared memory."""
    print("\n" + "=" * 80)
    print("Example 4: Real AutoGen Agents with Shared Memory")
    print("=" * 80)

    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        import os
    except ImportError:
        print("\n⚠️  Skipping Example 4: autogen-agentchat not installed")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Skipping Example 4: OPENAI_API_KEY not set")
        return

    # Configure shared memory
    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="real_agents_memory",
        k=3
    )

    memory = DXMTVectorMemory(config=config)

    # Add some background knowledge
    print("\n📝 Adding background knowledge...\n")

    knowledge = [
        "The company's tech stack is Python, FastAPI, PostgreSQL, and React",
        "We follow agile methodology with 2-week sprints",
        "The main project is a customer analytics platform",
        "We have a CI/CD pipeline using GitHub Actions",
        "Team size is 12 engineers across 3 time zones",
    ]

    for content in knowledge:
        await memory.add(MemoryContent(
            content=content,
            mime_type="text/plain",
            metadata={"source": "company_info"}
        ))
        print(f"  ✓ {content}")

    # Create multiple agents with shared memory
    print("\n🤖 Creating agents with shared memory...\n")

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    # Technical Lead Agent
    tech_lead = AssistantAgent(
        name="tech_lead",
        model_client=model_client,
        memory=[memory],
        system_message=(
            "You are a technical lead. Provide architectural guidance "
            "based on the company's tech stack and practices."
        )
    )

    # Project Manager Agent
    pm_agent = AssistantAgent(
        name="project_manager",
        model_client=model_client,
        memory=[memory],
        system_message=(
            "You are a project manager. Coordinate team activities "
            "and ensure alignment with company processes."
        )
    )

    print("  ✓ Created tech_lead agent")
    print("  ✓ Created project_manager agent")

    # Test agents with shared context
    print("\n💬 Testing agents with shared memory context...\n")

    # Tech Lead query
    print("Tech Lead:")
    print("Task: What technology should we use for the new microservice?")
    try:
        result = await tech_lead.run(
            task="What technology should we use for the new microservice?"
        )
        print(f"Response: {result.messages[-1].content[:200]}...")
    except Exception as e:
        print(f"  Error: {e}")

    print()

    # Project Manager query
    print("Project Manager:")
    print("Task: How should we organize the development sprint?")
    try:
        result = await pm_agent.run(
            task="How should we organize the development sprint?"
        )
        print(f"Response: {result.messages[-1].content[:200]}...")
    except Exception as e:
        print(f"  Error: {e}")

    # Clean up
    await memory.clear()
    await memory.close()
    print("\n✅ Example 4 completed!")


async def main():
    """Run all multi-agent examples."""
    print("\n" + "=" * 80)
    print("DXMT VectorDB Multi-Agent Memory Examples")
    print("=" * 80)

    try:
        await example_shared_memory()
        await example_collaborative_workflow()
        await example_session_based_memory()
        await example_with_real_agents()

        print("\n" + "=" * 80)
        print("✅ All multi-agent examples completed successfully!")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
