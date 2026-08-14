"""Example: Using DXMT VectorDB as LangGraph Store for AI Agent Memory.

This example demonstrates how to use DXMT VectorDB as a persistent memory store
for LangGraph agents, enabling them to remember conversations, user preferences,
and context across sessions.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "dxmt-python-sdk"))

from dxmt import DXMTClient
from dxmt.langgraph_store import DXMTLangGraphStore


def example_basic_memory():
    """Example 1: Basic key-value storage without semantic search."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Memory Storage (No Semantic Search)")
    print("=" * 80)

    # Initialize DXMT client
    client = DXMTClient(host="localhost", port=9999)

    # Create store without embedding configuration
    store = DXMTLangGraphStore(client)

    # Store user preferences
    print("\n1. Storing user preferences...")
    store.put(
        namespace=("users", "alice"),
        key="preferences",
        value={
            "theme": "dark",
            "language": "python",
            "notifications": True,
        },
    )
    print("✓ Stored preferences for user 'alice'")

    # Store conversation history
    print("\n2. Storing conversation history...")
    store.put(
        namespace=("users", "alice", "conversations"),
        key="conv_001",
        value={
            "title": "Getting started with LangGraph",
            "message_count": 5,
            "last_message": "Thanks for the help!",
        },
    )
    print("✓ Stored conversation conv_001")

    # Retrieve data
    print("\n3. Retrieving stored data...")
    prefs = store.get(("users", "alice"), "preferences")
    if prefs:
        print(f"✓ Retrieved preferences: {prefs.value}")

    conv = store.get(("users", "alice", "conversations"), "conv_001")
    if conv:
        print(f"✓ Retrieved conversation: {conv.value}")

    # List namespaces
    print("\n4. Listing namespaces...")
    namespaces = store.list_namespaces(prefix=("users",))
    print(f"✓ Found namespaces: {namespaces}")

    # Delete data
    print("\n5. Deleting conversation...")
    store.delete(("users", "alice", "conversations"), "conv_001")
    print("✓ Deleted conversation conv_001")


def example_semantic_search():
    """Example 2: Semantic search with embeddings."""
    print("\n" + "=" * 80)
    print("Example 2: Semantic Memory Search")
    print("=" * 80)

    # Initialize client
    client = DXMTClient(host="localhost", port=9999)

    # Mock embedding function (replace with real embeddings in production)
    import random

    def mock_embed(texts: list[str]) -> list[list[float]]:
        """Mock embedding function. Use OpenAI/Cohere embeddings in production."""
        print(f"  Generating embeddings for {len(texts)} text(s)...")
        return [[random.random() for _ in range(384)] for _ in texts]

    # Create store with embedding support
    store = DXMTLangGraphStore(
        client,
        embedding_dims=384,
        embed_fn=mock_embed,
        default_index_fields=["content"],  # Index the 'content' field
        distance_metric="cosine",
    )

    # Store agent memories with semantic indexing
    print("\n1. Storing agent memories...")

    memories = [
        {
            "key": "mem_001",
            "content": "User prefers Python over JavaScript for data science",
            "category": "preference",
            "importance": 0.9,
        },
        {
            "key": "mem_002",
            "content": "User is learning LangGraph for building AI agents",
            "category": "activity",
            "importance": 0.8,
        },
        {
            "key": "mem_003",
            "content": "User asked about DXMT VectorDB integration",
            "category": "question",
            "importance": 0.7,
        },
        {
            "key": "mem_004",
            "content": "User wants to deploy agents to production",
            "category": "goal",
            "importance": 0.95,
        },
    ]

    for mem in memories:
        store.put(
            namespace=("agent", "assistant_v1", "memories"),
            key=mem["key"],
            value={
                "content": mem["content"],
                "category": mem["category"],
                "importance": mem["importance"],
            },
            index=["content"],  # Index the content field for search
        )
        print(f"  ✓ Stored {mem['key']}")

    # Semantic search
    print("\n2. Performing semantic search...")
    print("Query: 'What programming languages does the user like?'")

    results = store.search(
        ("agent", "assistant_v1", "memories"),
        query="What programming languages does the user like?",
        limit=3,
    )

    print(f"\n✓ Found {len(results)} relevant memories:")
    for i, result in enumerate(results, 1):
        print(f"\n  Result {i} (score: {result.score:.3f}):")
        print(f"    Key: {result.key}")
        print(f"    Content: {result.value.get('content', '')}")
        print(f"    Category: {result.value.get('category', '')}")

    # Search with filters
    print("\n3. Searching with metadata filters...")
    print("Query: 'production deployment'")
    print("Filter: importance >= 0.8")

    results = store.search(
        ("agent", "assistant_v1", "memories"),
        query="production deployment",
        filter={"importance": {"$gte": 0.8}},
        limit=3,
    )

    print(f"\n✓ Found {len(results)} high-importance memories:")
    for i, result in enumerate(results, 1):
        print(f"\n  Result {i} (score: {result.score:.3f}):")
        print(f"    Content: {result.value.get('content', '')}")
        print(f"    Importance: {result.value.get('importance', 0)}")


def example_multi_agent_memory():
    """Example 3: Memory isolation across multiple agents."""
    print("\n" + "=" * 80)
    print("Example 3: Multi-Agent Memory Isolation")
    print("=" * 80)

    # Initialize client
    client = DXMTClient(host="localhost", port=9999)
    store = DXMTLangGraphStore(client)

    # Store memories for different agents
    print("\n1. Storing memories for multiple agents...")

    # Agent 1: Code assistant
    store.put(
        namespace=("agents", "code_assistant", "session_123"),
        key="context",
        value={
            "task": "Debug authentication error",
            "files_reviewed": ["auth.py", "middleware.py"],
            "suggestions": 3,
        },
    )
    print("  ✓ Stored context for code_assistant/session_123")

    # Agent 2: Documentation writer
    store.put(
        namespace=("agents", "doc_writer", "session_456"),
        key="context",
        value={
            "task": "Write API documentation",
            "sections_completed": ["authentication", "endpoints"],
            "word_count": 1500,
        },
    )
    print("  ✓ Stored context for doc_writer/session_456")

    # Agent 3: Code reviewer
    store.put(
        namespace=("agents", "code_reviewer", "session_789"),
        key="context",
        value={
            "task": "Review pull request #42",
            "files_checked": 8,
            "issues_found": 2,
        },
    )
    print("  ✓ Stored context for code_reviewer/session_789")

    # List all agent types
    print("\n2. Listing all agent types...")
    agent_types = store.list_namespaces(prefix=("agents",), max_depth=2)
    print(f"✓ Found {len(agent_types)} agent namespace(s):")
    for ns in agent_types:
        print(f"  - {'/'.join(ns)}")

    # Retrieve specific agent context
    print("\n3. Retrieving specific agent context...")
    context = store.get(("agents", "code_assistant", "session_123"), "context")
    if context:
        print(f"✓ Code assistant context: {context.value}")


def example_conversation_memory():
    """Example 4: Conversation memory management."""
    print("\n" + "=" * 80)
    print("Example 4: Conversation Memory Management")
    print("=" * 80)

    # Initialize client
    client = DXMTClient(host="localhost", port=9999)
    store = DXMTLangGraphStore(client)

    user_id = "user_42"
    session_id = "session_20260128"

    # Store conversation turns
    print("\n1. Storing conversation turns...")

    conversation = [
        {"role": "user", "content": "How do I use DXMT with LangGraph?"},
        {
            "role": "assistant",
            "content": "You can use DXMTLangGraphStore as a memory backend...",
        },
        {"role": "user", "content": "Can it do semantic search?"},
        {"role": "assistant", "content": "Yes, it supports semantic search..."},
    ]

    for i, turn in enumerate(conversation):
        store.put(
            namespace=("conversations", user_id, session_id),
            key=f"turn_{i:03d}",
            value={
                "role": turn["role"],
                "content": turn["content"],
                "turn_number": i,
                "timestamp": f"2026-01-28T10:{i:02d}:00Z",
            },
        )
        print(f"  ✓ Stored turn_{i:03d}")

    # Store conversation metadata
    print("\n2. Storing conversation metadata...")
    store.put(
        namespace=("conversations", user_id),
        key=session_id,
        value={
            "title": "Getting started with DXMT + LangGraph",
            "message_count": len(conversation),
            "started_at": "2026-01-28T10:00:00Z",
            "last_activity": "2026-01-28T10:03:00Z",
        },
    )
    print(f"  ✓ Stored metadata for {session_id}")

    # List user's conversations
    print("\n3. Listing user's conversations...")
    user_convs = store.list_namespaces(prefix=("conversations", user_id), max_depth=3)
    print(f"✓ Found {len(user_convs)} conversation(s) for {user_id}")


def example_real_world_integration():
    """Example 5: Real-world integration with OpenAI embeddings."""
    print("\n" + "=" * 80)
    print("Example 5: Real-World Integration (Requires OpenAI API Key)")
    print("=" * 80)

    # Check if OpenAI is available
    try:
        from openai import OpenAI

        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("\n⚠️  OPENAI_API_KEY not set. Skipping this example.")
            print("   Set OPENAI_API_KEY environment variable to run this example.")
            return

        print("\n✓ OpenAI available, proceeding with example...")

    except ImportError:
        print("\n⚠️  OpenAI package not installed. Skipping this example.")
        print("   Install with: pip install openai")
        return

    # Initialize OpenAI client
    openai_client = OpenAI(api_key=api_key)

    # Define embedding function
    def embed_texts(texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI."""
        response = openai_client.embeddings.create(
            model="text-embedding-3-small", input=texts
        )
        return [item.embedding for item in response.data]

    # Initialize DXMT store with OpenAI embeddings
    client = DXMTClient(host="localhost", port=9999)
    store = DXMTLangGraphStore(
        client,
        embedding_dims=1536,  # text-embedding-3-small dimension
        embed_fn=embed_texts,
        default_index_fields=["$"],  # Index entire document
        distance_metric="cosine",
    )

    print("\n1. Storing documents with OpenAI embeddings...")

    docs = [
        {
            "id": "doc_001",
            "title": "LangGraph Introduction",
            "content": "LangGraph is a framework for building stateful agents...",
            "category": "tutorial",
        },
        {
            "id": "doc_002",
            "title": "DXMT Architecture",
            "content": "DXMT uses tiered storage with Memory, SSD, and S3...",
            "category": "architecture",
        },
        {
            "id": "doc_003",
            "title": "Vector Search Best Practices",
            "content": "When implementing semantic search, consider indexing strategies...",
            "category": "guide",
        },
    ]

    for doc in docs:
        store.put(
            namespace=("knowledge_base", "documentation"),
            key=doc["id"],
            value=doc,
            index=["content"],  # Index content field
        )
        print(f"  ✓ Stored {doc['id']}")

    print("\n2. Performing semantic search with real embeddings...")
    print("Query: 'How to build agents with state?'")

    results = store.search(
        ("knowledge_base",),
        query="How to build agents with state?",
        limit=2,
    )

    print(f"\n✓ Found {len(results)} relevant documents:")
    for i, result in enumerate(results, 1):
        print(f"\n  Result {i} (score: {result.score:.3f}):")
        print(f"    Title: {result.value.get('title', '')}")
        print(f"    Category: {result.value.get('category', '')}")
        print(f"    Content: {result.value.get('content', '')[:80]}...")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("DXMT LangGraph Store Examples")
    print("=" * 80)
    print("\nThese examples demonstrate how to use DXMT VectorDB as a persistent")
    print("memory store for LangGraph agents.")
    print("\nPrerequisites:")
    print("  - DXMT server running on localhost:9999")
    print("  - Python packages: dxmt-python-sdk")
    print("  - Optional: OpenAI API key for Example 5")

    try:
        # Run examples
        example_basic_memory()
        example_semantic_search()
        example_multi_agent_memory()
        example_conversation_memory()
        example_real_world_integration()

        print("\n" + "=" * 80)
        print("✓ All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
