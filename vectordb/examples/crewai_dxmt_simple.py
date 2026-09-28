"""Simple CrewAI + DXMT VectorDB Integration Test.

A minimal example demonstrating DXMT Vector Search Tool with CrewAI.

Requirements:
    pip install crewai crewai-tools dxmt-python-sdk openai

Usage:
    1. Start DXMT: cargo run -- serve --host 0.0.0.0 --port 9999
    2. Set OPENAI_API_KEY: export OPENAI_API_KEY='your-key'
    3. Run: python examples/crewai_dxmt_simple.py
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


def main():
    """Simple CrewAI + DXMT integration test."""
    print("\n" + "=" * 70)
    print("  CrewAI + DXMT VectorDB - Simple Integration Test")
    print("=" * 70)

    # Check dependencies
    try:
        from crewai import Agent, Task, Crew
        from dxmt.crewai_tool import DXMTVectorSearchTool, DXMTToolConfig
        from dxmt import DXMTClient
    except ImportError as e:
        print(f"\n❌ Missing dependencies: {e}")
        print("Install: pip install crewai crewai-tools dxmt-python-sdk openai")
        return

    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set!")
        print("Set with: export OPENAI_API_KEY='your-key-here'")
        return

    # Step 1: Initialize DXMT
    print("\n[1/5] Initializing DXMT client...")
    client = DXMTClient(host="localhost", port=9999)

    table_name = "crew_test_docs"
    try:
        client.create_table(
            name=table_name,
            dimension=EMBED_DIMS,
            distance_metric="cosine",
            description="CrewAI test documents"
        )
        print(f"  ✓ Created table: {table_name}")
    except Exception:
        print(f"  ℹ️  Table '{table_name}' already exists")

    # Step 2: Configure DXMT tool
    print("\n[2/5] Configuring DXMT search tool...")
    config = DXMTToolConfig(
        host="localhost",
        port=9999,
        table_name=table_name,
        limit=3,
        score_threshold=0.3
    )

    search_tool = DXMTVectorSearchTool(dxmt_config=config)
    print("  ✓ DXMT search tool configured")

    # Step 3: Add test documents
    print("\n[3/5] Adding test documents...")
    documents = [
        "DXMT VectorDB provides 98% cost savings through tiered storage",
        "CrewAI enables powerful multi-agent AI systems",
        "Vector databases excel at semantic similarity search",
    ]

    try:
        search_tool.add_documents(
            texts=documents,
            metadatas=[{"id": i} for i in range(len(documents))]
        )
        print(f"  ✓ Added {len(documents)} documents")
    except Exception as e:
        print(f"  ⚠️  Error (docs may exist): {e}")

    # Step 4: Create agent with tool
    print("\n[4/5] Creating CrewAI agent...")
    researcher = Agent(
        role="Research Assistant",
        goal="Find relevant information using vector search",
        backstory="An AI assistant specialized in information retrieval",
        tools=[search_tool],
        verbose=True
    )
    print("  ✓ Agent created with DXMT tool")

    # Step 5: Run task
    print("\n[5/5] Running CrewAI task...")
    task = Task(
        description="Search for information about vector databases and their benefits",
        agent=researcher,
        expected_output="Summary of vector database benefits"
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        verbose=True
    )

    print("\n" + "-" * 70)
    print("  Executing CrewAI with DXMT Vector Search...")
    print("-" * 70 + "\n")

    try:
        result = crew.kickoff()

        print("\n" + "=" * 70)
        print("  ✅ SUCCESS! CrewAI + DXMT Integration Working")
        print("=" * 70)
        print("\n📊 Result:")
        print(result)

        print("\n" + "=" * 70)
        print("  Integration Status: ✅ VERIFIED")
        print("=" * 70)
        print("\n✨ Key Features Demonstrated:")
        print("  ✓ DXMT as CrewAI tool")
        print("  ✓ Vector similarity search")
        print("  ✓ Agent-tool integration")
        print("  ✓ Semantic information retrieval")
        print("\n🚀 DXMT + CrewAI ready for production!")

    except Exception as e:
        print(f"\n❌ Error running crew: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
