"""
Simple DXMT + mem0 Integration Example

This example demonstrates basic usage of DXMT VectorDB as a memory backend for mem0.
Similar to mem0's reference examples but focused on DXMT-specific features.

Requirements:
    pip install mem0ai dxmt-python-sdk openai

Usage:
    # Set your OpenAI API key
    export OPENAI_API_KEY="sk-..."

    # Start DXMT server
    cargo run -- serve --host 0.0.0.0 --port 9999

    # Run example
    python examples/dxmt_mem0_simple.py
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "clients", "dxmt-python-sdk"))

from dxmt import Client as DXMTClient
from dxmt.mem0 import DXMTMemoryStore

# Load environment variables
load_dotenv()

# Check if mem0 is available
try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    print("ERROR: mem0 is not installed. Install with: pip install mem0ai")
    MEM0_AVAILABLE = False
    sys.exit(1)


def create_dxmt_memory_config():
    """Create mem0 configuration with DXMT backend."""
    # Initialize DXMT client
    client = DXMTClient(
        host=os.getenv("DXMT_HOST", "localhost"),
        port=int(os.getenv("DXMT_PORT", "9999")),
    )

    # Create DXMT memory store
    vector_store = DXMTMemoryStore(
        collection_name="simple_demo",
        client=client,
        embedding_model_dims=1536,  # OpenAI text-embedding-ada-002
        distance_metric="cosine",
    )

    # Configure mem0
    config = {
        "vector_store": {
            "provider": "custom",
            "config": {
                "custom_vector_store": vector_store,
                "collection_name": "simple_demo",
                "embedding_model_dims": 1536,
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-ada-002",
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


def example_basic_operations():
    """Demonstrate basic memory operations."""
    print("\n" + "="*60)
    print("Example: Basic Memory Operations with DXMT")
    print("="*60)

    config = create_dxmt_memory_config()
    m = Memory.from_config(config)

    user_id = "alice"

    # 1. Add user preferences
    print("\n1. Adding user preferences...")
    conversations = [
        {"role": "user", "content": "I love Python programming"},
        {"role": "assistant", "content": "Great! Python is versatile."},
        {"role": "user", "content": "I prefer VS Code as my editor"},
        {"role": "assistant", "content": "VS Code is excellent for Python."},
        {"role": "user", "content": "I'm learning machine learning"},
    ]

    result = m.add(conversations, user_id=user_id)
    print(f"   Added {len(result)} memories")
    for mem in result:
        print(f"   - {mem['memory']}")

    # 2. Search memories
    print("\n2. Searching for programming preferences...")
    query = "What are my programming preferences?"
    memories = m.search(query, user_id=user_id, limit=5)

    print(f"   Found {len(memories)} relevant memories:")
    for i, mem in enumerate(memories, 1):
        print(f"   {i}. {mem['memory']} (score: {mem['score']:.3f})")

    # 3. Get all memories
    print("\n3. Getting all memories...")
    all_memories = m.get_all(user_id=user_id)
    print(f"   Total memories for {user_id}: {len(all_memories)}")

    return m


def example_multi_user():
    """Demonstrate multi-user memory isolation."""
    print("\n" + "="*60)
    print("Example: Multi-User Memory Isolation")
    print("="*60)

    config = create_dxmt_memory_config()
    m = Memory.from_config(config)

    # User Alice
    print("\n1. Adding memories for Alice...")
    m.add("I work as a data scientist at Google", user_id="alice")
    m.add("I live in San Francisco", user_id="alice")

    # User Bob
    print("2. Adding memories for Bob...")
    m.add("I'm a software engineer at Microsoft", user_id="bob")
    m.add("I live in Seattle", user_id="bob")

    # Search for each user
    print("\n3. Searching memories by user...")

    alice_memories = m.search("work", user_id="alice", limit=5)
    print(f"\n   Alice's work-related memories ({len(alice_memories)}):")
    for mem in alice_memories:
        print(f"   - {mem['memory']}")

    bob_memories = m.search("work", user_id="bob", limit=5)
    print(f"\n   Bob's work-related memories ({len(bob_memories)}):")
    for mem in bob_memories:
        print(f"   - {mem['memory']}")

    return m


def example_memory_updates():
    """Demonstrate memory update operations."""
    print("\n" + "="*60)
    print("Example: Memory Updates")
    print("="*60)

    config = create_dxmt_memory_config()
    m = Memory.from_config(config)

    user_id = "charlie"

    # Add initial memory
    print("\n1. Adding initial preference...")
    result = m.add("I prefer email notifications", user_id=user_id)
    memory_id = result[0]["id"]
    print(f"   Created memory: {memory_id}")
    print(f"   Content: {result[0]['memory']}")

    # Update memory
    print("\n2. Updating preference...")
    m.update(memory_id=memory_id, data="I prefer SMS notifications")

    # Verify update
    print("\n3. Verifying update...")
    updated = m.get(memory_id)
    if updated:
        print(f"   Updated memory: {updated['memory']}")

    return m


def example_cleanup():
    """Demonstrate cleanup operations."""
    print("\n" + "="*60)
    print("Example: Memory Cleanup")
    print("="*60)

    config = create_dxmt_memory_config()
    m = Memory.from_config(config)

    user_id = "test_user"

    # Add some test memories
    print("\n1. Adding test memories...")
    m.add("Test memory 1", user_id=user_id)
    m.add("Test memory 2", user_id=user_id)
    m.add("Test memory 3", user_id=user_id)

    all_memories = m.get_all(user_id=user_id)
    print(f"   Created {len(all_memories)} test memories")

    # Delete all memories for user
    print("\n2. Deleting all memories for user...")
    m.delete_all(user_id=user_id)

    remaining = m.get_all(user_id=user_id)
    print(f"   Remaining memories: {len(remaining)}")

    return m


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("DXMT + mem0 Simple Integration Examples")
    print("="*60)
    print("\nDemonstrating mem0 memory operations with DXMT backend")
    print("Make sure DXMT server is running on localhost:9999")
    print("Set OPENAI_API_KEY environment variable")

    if not os.getenv("OPENAI_API_KEY"):
        print("\nWARNING: OPENAI_API_KEY not set. Examples may fail.")
        print("Set with: export OPENAI_API_KEY='sk-...'")

    try:
        # Run examples
        example_basic_operations()
        example_multi_user()
        example_memory_updates()
        example_cleanup()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Basic memory operations (add, search, get)")
        print("✓ Multi-user memory isolation")
        print("✓ Memory updates")
        print("✓ Memory cleanup")
        print("\nDXMT provides:")
        print("- Cost-efficient tiered storage (98% cost reduction)")
        print("- High performance (3.56M QPS vector search)")
        print("- Self-hosted, no vendor lock-in")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
