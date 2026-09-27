"""
mem0 + DXMT VectorDB Integration Example

This example demonstrates how to use DXMT as the vector store backend for mem0.
mem0 (https://mem0.ai) is an intelligent memory layer for AI assistants and agents.

Requirements:
    pip install mem0ai dxmt-python-sdk openai

Usage:
    # Set your OpenAI API key
    export OPENAI_API_KEY="sk-..."

    # Start DXMT server
    cargo run -- serve --host 0.0.0.0 --port 9999

    # Run examples
    python examples/mem0_integration.py
"""

import os
import sys
import time
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "clients", "dxmt-python-sdk"))

from dxmt import Client as DXMTClient
from dxmt.mem0 import DXMTMemoryStore, register_with_mem0

# Check if mem0 is installed
try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    print("ERROR: mem0 is not installed. Install with: pip install mem0ai")
    MEM0_AVAILABLE = False
    sys.exit(1)


# ---------- Configuration ----------

def rows(result):
    """The list inside what mem0 returns.

    `add`, `search` and `get_all` return `{"results": [...]}` now; they used to return the
    list. The examples iterated the return value directly, so the first one printed
    `TypeError: string indices must be integers` -- it was iterating the dict's keys. Both
    shapes are read here so this runs on either.
    """
    return result.get("results", result) if isinstance(result, dict) else result

def first(result, what):
    """The first memory mem0 extracted, or a clear exit if it extracted none.

    `add` runs the text through an LLM that decides what is worth remembering, so it can
    return nothing for a sentence a human would call memorable, and does so occasionally for
    the same input. The examples indexed `[0]` and died with `IndexError: list index out of
    range`, which reads like a bug in this database and is not one.
    """
    if not result:
        print(f"   mem0 extracted no memory from {what!r}; nothing to demonstrate here.")
        return None
    return result[0]


def create_dxmt_mem0_config(collection_name: str = "mem0"):
    """Create mem0 configuration using DXMT as vector store.

    Args:
        collection_name: Name of the collection/table to use.

    Returns:
        dict: mem0 configuration dictionary.
    """
    # Initialize DXMT client
    dxmt_client = DXMTClient(
        host=os.getenv("DXMT_HOST", "localhost"),
        port=int(os.getenv("DXMT_PORT", "9999")),
    )

    # Create DXMT memory store
    vector_store = DXMTMemoryStore(
        collection_name=collection_name,
        client=dxmt_client,
        embedding_model_dims=1536,  # OpenAI text-embedding-3-small
        distance_metric="cosine",
    )

    # mem0 configuration
    config = {
        # `provider: "custom"` is gone from mem0 -- it validates against a fixed table of
        # named backends now, and an unlisted one fails in pydantic before any adapter code
        # runs. `register_with_mem0` puts DXMT in that table; mem0 then builds the store
        # itself from the config below, so it takes arguments rather than an instance.
        "vector_store": {
            "provider": register_with_mem0(),
            "config": {
                "collection_name": collection_name,
                "client": dxmt_client,
                "embedding_model_dims": 1536,
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
                "embedding_dims": 1536,
            }
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "temperature": 0,
            }
        },
    }

    return config


# ---------- Example 1: User Memory ----------

def example_user_memory():
    """Example: Store and retrieve user-specific memories.

    Demonstrates:
    - Adding user memories from conversation
    - Searching memories by user_id
    - Retrieving all memories for a user
    """
    print("\n" + "="*60)
    print("Example 1: User Memory Storage and Retrieval")
    print("="*60)

    config = create_dxmt_mem0_config("user_memories")
    m = Memory.from_config(config)

    # Add user memories from conversation
    print("\n1. Adding user memories...")
    messages = [
        {"role": "user", "content": "I love programming in Python"},
        {"role": "assistant", "content": "That's great! Python is very versatile."},
        {"role": "user", "content": "I prefer dark mode for my IDE"},
        {"role": "assistant", "content": "Dark mode is easier on the eyes for long coding sessions."},
        {"role": "user", "content": "I use VS Code as my primary editor"},
    ]

    result = rows(m.add(messages, user_id="user_123"))
    print(f"   Added {len(result)} memories for user_123")
    for mem in result:
        print(f"   - {mem['memory']}")

    # Search memories
    print("\n2. Searching for programming preferences...")
    query = "What are my programming preferences?"
    memories = rows(m.search(query, top_k=5, filters={"user_id": "user_123"}))

    print(f"   Found {len(memories)} relevant memories:")
    for i, mem in enumerate(memories, 1):
        print(f"   {i}. {mem['memory']} (score: {mem['score']:.3f})")

    # Get all memories for user
    print("\n3. Getting all memories for user_123...")
    all_memories = rows(m.get_all(filters={"user_id": "user_123"}))
    print(f"   Total memories: {len(all_memories)}")

    return m


# ---------- Example 2: Agent Memory ----------

def example_agent_memory():
    """Example: Store agent conversation context.

    Demonstrates:
    - Storing agent-specific conversation history
    - Filtering memories by agent_id and run_id
    - Session-based memory retrieval
    """
    print("\n" + "="*60)
    print("Example 2: Agent Conversation Context")
    print("="*60)

    config = create_dxmt_mem0_config("agent_memories")
    m = Memory.from_config(config)

    # Agent conversation
    print("\n1. Storing weather agent conversation...")
    messages = [
        {"role": "user", "content": "What's the weather today?"},
        {"role": "assistant", "content": "It's sunny and 75°F in San Francisco."},
        {"role": "user", "content": "Should I bring an umbrella?"},
        {"role": "assistant", "content": "No need, no rain expected today."},
    ]

    result = rows(m.add(
        messages,
        agent_id="weather_agent",
        run_id="session_456"
    ))
    print(f"   Stored {len(result)} memories for weather_agent")

    # Retrieve agent context
    print("\n2. Retrieving weather conversation context...")
    memories = rows(m.search("weather information", top_k=5, filters={"agent_id": "weather_agent", "run_id": "session_456"}))

    print(f"   Agent conversation context ({len(memories)} memories):")
    for mem in memories:
        print(f"   - {mem['memory']}")

    return m


# ---------- Example 3: Multi-Scope Memory ----------

def example_multi_scope_memory():
    """Example: Multi-scope memory (user + agent + run).

    Demonstrates:
    - Storing memories with multiple scopes
    - Querying across different scope combinations
    - Custom metadata fields
    """
    print("\n" + "="*60)
    print("Example 3: Multi-Scope Memory Management")
    print("="*60)

    config = create_dxmt_mem0_config("multi_scope_memories")
    m = Memory.from_config(config)

    # Add memory with multiple scopes
    print("\n1. Adding travel booking memory...")
    messages = [
        {"role": "user", "content": "I want to book a flight to Paris next month"},
        {"role": "assistant", "content": "I'll help you book a flight to Paris."},
    ]

    result = rows(m.add(
        messages,
        user_id="user_123",
        agent_id="travel_agent",
        run_id="booking_session_789",
        metadata={"intent": "flight_booking", "destination": "Paris", "priority": "high"}
    ))
    print(f"   Added {len(result)} memories with multi-scope")

    # Query across different scopes
    print("\n2. Querying across different scopes...")

    # All user memories
    user_mems = rows(m.get_all(filters={"user_id": "user_123"}))
    print(f"   - User memories (user_123): {len(user_mems)}")

    # Agent-specific memories
    agent_mems = rows(m.get_all(filters={"agent_id": "travel_agent"}))
    print(f"   - Agent memories (travel_agent): {len(agent_mems)}")

    # Session-specific memories
    session_mems = rows(m.get_all(filters={"run_id": "booking_session_789"}))
    print(f"   - Session memories (booking_session_789): {len(session_mems)}")

    # Multi-scope query (user + agent)
    specific_mems = rows(m.search("travel plans", top_k=5, filters={"user_id": "user_123", "agent_id": "travel_agent"}))
    print(f"   - User + Agent memories: {len(specific_mems)}")
    for mem in specific_mems:
        print(f"     * {mem['memory']}")

    return m


# ---------- Example 4: Memory Updates ----------

def example_memory_update():
    """Example: Update existing memories.

    Demonstrates:
    - Updating memory content
    - Tracking memory modifications
    - Timestamp management
    """
    print("\n" + "="*60)
    print("Example 4: Memory Updates")
    print("="*60)

    config = create_dxmt_mem0_config("updatable_memories")
    m = Memory.from_config(config)

    # Add initial memory
    print("\n1. Adding initial notification preference...")
    result = rows(m.add(
        "I prefer email notifications",
        user_id="user_123",
        metadata={"preference_type": "notification", "channel": "email"}
    ))
    _m = first(result, "the update example")
    if _m is None:
        return
    memory_id = _m["id"]
    print(f"   Created memory: {memory_id}")
    print(f"   Content: {_m['memory']}")

    # Update memory
    print("\n2. Updating notification preference...")
    m.update(
        memory_id=memory_id,
        data="I prefer SMS notifications instead of email"
    )

    # Verify update
    print("\n3. Verifying update...")
    updated = m.get(memory_id)
    if updated:
        print(f"   Updated memory: {updated['memory']}")
        print(f"   Updated at: {updated.get('updated_at', 'N/A')}")

    return m


# ---------- Example 5: Memory Deletion ----------

def example_memory_deletion():
    """Example: Delete memories.

    Demonstrates:
    - Deleting specific memories
    - Bulk deletion by user_id
    - Collection reset
    """
    print("\n" + "="*60)
    print("Example 5: Memory Deletion")
    print("="*60)

    config = create_dxmt_mem0_config("deletable_memories")
    m = Memory.from_config(config)

    # Add temporary memories
    print("\n1. Adding temporary memories...")
    result1 = rows(m.add("Temporary note 1", user_id="user_temp"))
    result2 = rows(m.add("Temporary note 2", user_id="user_temp"))
    _m = first(result1, "the cleanup example")
    if _m is None:
        return
    memory_id = _m["id"]
    print(f"   Added 2 temporary memories")

    # Delete specific memory
    print("\n2. Deleting specific memory...")
    m.delete(memory_id)
    print(f"   Deleted memory: {memory_id}")

    # Verify deletion
    deleted = m.get(memory_id)
    print(f"   Memory exists after deletion: {deleted is not None}")

    # Delete all memories for user
    print("\n3. Deleting all memories for user_temp...")
    m.delete_all(user_id="user_temp")
    remaining = rows(m.get_all(filters={"user_id": "user_temp"}))
    print(f"   Remaining memories: {len(remaining)}")

    return m


# ---------- Example 6: Memory History ----------

def example_memory_history():
    """Example: Track memory history and versions.

    Demonstrates:
    - Memory operation tracking
    - Version history
    - Audit trail
    """
    print("\n" + "="*60)
    print("Example 6: Memory History Tracking")
    print("="*60)

    config = create_dxmt_mem0_config("history_memories")
    m = Memory.from_config(config)

    # Add memory and track history
    print("\n1. Adding memory and tracking history...")
    m.add("Version 1 of memory", user_id="user_123")
    m.add("Version 2 of memory", user_id="user_123")
    m.add("Version 3 of memory", user_id="user_123")

    print("\n2. Getting memory history...")
    # `history` takes one memory id now; it used to accept a scope like `user_id`. Ask for
    # the user's memories first and follow one of them.
    mems = rows(m.get_all(filters={"user_id": "user_123"}))
    if not mems:
        print("   no memories to follow; skipping")
        return m
    history = m.history(mems[0]["id"])
    print(f"   Total operations on {mems[0]['id']}: {len(history)}")

    # Show recent operations
    print("\n   Recent memory operations:")
    for i, entry in enumerate(history[:5], 1):
        print(f"   {i}. {entry.get('event', 'unknown')} at {entry.get('timestamp', 'N/A')}")
        if 'memory' in entry:
            print(f"      Memory: {entry['memory'][:60]}...")

    return m


# ---------- Example 7: Performance Benchmarking ----------

def example_performance_benchmark():
    """Example: Benchmark DXMT memory operations.

    Demonstrates:
    - Insertion throughput
    - Search latency
    - Scaling characteristics
    """
    print("\n" + "="*60)
    print("Example 7: Performance Benchmarking")
    print("="*60)

    config = create_dxmt_mem0_config("benchmark_memories")
    m = Memory.from_config(config)

    # Benchmark insertion
    print("\n1. Benchmarking memory insertion...")
    num_insertions = 100
    start = time.time()
    for i in range(num_insertions):
        m.add(f"Benchmark memory {i}", user_id="bench_user")
    insert_time = time.time() - start

    print(f"   Inserted {num_insertions} memories in {insert_time:.2f}s")
    print(f"   Throughput: {num_insertions/insert_time:.0f} memories/second")

    # Benchmark search
    print("\n2. Benchmarking memory search...")
    num_searches = 100
    start = time.time()
    for i in range(num_searches):
        m.search("benchmark query", top_k=10, filters={"user_id": "bench_user"})
    search_time = time.time() - start

    print(f"   Performed {num_searches} searches in {search_time:.2f}s")
    print(f"   Throughput: {num_searches/search_time:.0f} searches/second")
    print(f"   Average latency: {(search_time/num_searches)*1000:.1f}ms")

    # Cleanup
    print("\n3. Cleaning up benchmark data...")
    m.delete_all(user_id="bench_user")
    print("   Cleanup complete")

    return m


# ---------- Example 8: Advanced Filtering ----------

def example_advanced_filtering():
    """Example: Advanced filtering and search.

    Demonstrates:
    - Complex filter expressions
    - Range queries
    - Combined filters
    """
    print("\n" + "="*60)
    print("Example 8: Advanced Filtering")
    print("="*60)

    config = create_dxmt_mem0_config("filtered_memories")
    m = Memory.from_config(config)

    # Add memories with various metadata
    print("\n1. Adding memories with different metadata...")
    memories_data = [
        ("Python is great for data science", {"category": "tech", "rating": 9, "year": 2024}),
        ("Java is verbose but stable", {"category": "tech", "rating": 7, "year": 2023}),
        ("Rust is fast and safe", {"category": "tech", "rating": 10, "year": 2024}),
        ("I love Italian food", {"category": "food", "rating": 8, "year": 2024}),
    ]

    for content, metadata in memories_data:
        m.add(content, user_id="user_123", metadata=metadata)
    print(f"   Added {len(memories_data)} memories with metadata")

    # Search with category filter
    print("\n2. Searching tech memories...")
    tech_mems = rows(m.search(
        "programming", top_k=10,
        filters={"user_id": "user_123", "category": "tech"},
    ))
    print(f"   Found {len(tech_mems)} tech memories")
    for mem in tech_mems:
        print(f"   - {mem['memory']}")

    # Search with rating filter (if supported)
    print("\n3. Searching high-rated memories (rating >= 8)...")
    # Note: Complex filters may require direct vector store access
    # This is a simplified example
    all_mems = rows(m.get_all(filters={"user_id": "user_123"}))
    high_rated = [mem for mem in all_mems if mem.get('metadata', {}).get('rating', 0) >= 8]
    print(f"   Found {len(high_rated)} high-rated memories")
    for mem in high_rated:
        print(f"   - {mem['memory']} (rating: {mem.get('metadata', {}).get('rating')})")

    return m


# ---------- Main ----------

def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("DXMT + mem0 Integration Examples")
    print("="*60)
    print("\nThis script demonstrates various use cases for mem0 with DXMT backend.")
    print("Make sure DXMT server is running on localhost:9999")
    print("Set OPENAI_API_KEY environment variable for embeddings")

    if not os.getenv("OPENAI_API_KEY"):
        print("\nWARNING: OPENAI_API_KEY not set. Some examples may fail.")
        print("Set with: export OPENAI_API_KEY='sk-...'")

    try:
        # Run examples
        example_user_memory()
        example_agent_memory()
        example_multi_scope_memory()
        example_memory_update()
        example_memory_deletion()
        example_memory_history()
        example_performance_benchmark()
        example_advanced_filtering()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
