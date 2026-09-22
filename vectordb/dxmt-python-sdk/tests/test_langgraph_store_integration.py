"""Integration tests for DXMT LangGraph Store (based on LangGraph patterns).

This test suite follows the same patterns used in LangGraph's PostgresStore tests,
ensuring compatibility and consistent behavior.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langgraph", reason="optional LangGraph integration dependency is not installed")

# Needs a running server, so it must carry the marker that says so. It did not, and every test
# in this file connects to one -- `pytest -m "not integration"` therefore failed on any machine
# with langgraph installed, which made the unit gate depend on a server being up. It passed
# here only because langgraph is absent and `importorskip` skipped the file first.
pytestmark = pytest.mark.integration

TEST_HOST = os.environ.get("DXMT_TEST_HOST", "localhost")
TEST_PORT = int(os.environ.get("DXMT_TEST_PORT", "9999"))

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dxmt import DXMTClient
from dxmt.langgraph_store import DXMTLangGraphStore
from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchItem,
    SearchOp,
)


class MockEmbeddings:
    """Mock embeddings for testing without external dependencies."""

    def __init__(self, dims: int = 384):
        self.dims = dims

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings based on text content."""
        import hashlib

        embeddings = []
        for text in texts:
            # Generate deterministic embedding from text hash
            hash_val = hashlib.md5(text.encode()).hexdigest()
            # Convert hash to numbers
            embedding = [
                float(int(hash_val[i : i + 2], 16)) / 255.0 for i in range(0, min(len(hash_val), self.dims * 2), 2)
            ]
            # Pad to required dimension
            while len(embedding) < self.dims:
                embedding.append(0.0)
            embeddings.append(embedding[: self.dims])
        return embeddings

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


@pytest.fixture
def client():
    """The server under test, `DXMT_TEST_HOST`/`DXMT_TEST_PORT` as everywhere else. Hard-coded
    to localhost:9999 before, so it could not be pointed at the server CI actually starts."""
    return DXMTClient(host=TEST_HOST, port=TEST_PORT)


@pytest.fixture
def store(client):
    """Create DXMTLangGraphStore for testing."""
    return DXMTLangGraphStore(client)


@pytest.fixture
def store_with_embeddings(client):
    """Create DXMTLangGraphStore with mock embeddings."""
    mock_embed = MockEmbeddings(dims=384)
    return DXMTLangGraphStore(
        client,
        embedding_dims=384,
        embed_fn=mock_embed,
        default_index_fields=["$"],
    )


class TestBasicOperations:
    """Test basic CRUD operations (following LangGraph patterns)."""

    def test_put_and_get(self, store):
        """Test storing and retrieving items."""
        # Put item
        store.put(("test",), "key1", {"data": "value1"})

        # Get item
        item = store.get(("test",), "key1")

        assert item is not None
        assert isinstance(item, Item)
        assert item.key == "key1"
        assert item.namespace == ("test",)
        assert item.value == {"data": "value1"}
        assert item.created_at is not None
        assert item.updated_at is not None

    def test_put_update(self, store):
        """Test updating existing items."""
        # Initial put
        store.put(("test",), "key1", {"data": "value1"})
        item1 = store.get(("test",), "key1")

        # Update
        time.sleep(0.1)  # Ensure different timestamp
        store.put(("test",), "key1", {"data": "value2"})
        item2 = store.get(("test",), "key1")

        assert item2 is not None
        assert item2.value == {"data": "value2"}
        assert item2.created_at == item1.created_at  # Created time preserved
        assert item2.updated_at > item1.updated_at  # Updated time changed

    def test_delete(self, store):
        """Test deleting items."""
        # Put item
        store.put(("test",), "key1", {"data": "value1"})
        assert store.get(("test",), "key1") is not None

        # Delete
        store.delete(("test",), "key1")

        # Verify deleted
        assert store.get(("test",), "key1") is None

    def test_get_nonexistent(self, store):
        """Test getting non-existent item returns None."""
        item = store.get(("nonexistent",), "key")
        assert item is None


class TestBatchOperations:
    """Test batch operations (following PostgresStore test patterns)."""

    def test_batch_order(self, store):
        """Test that batch operations maintain order (like test_batch_order)."""
        # Setup test data
        store.put(("test", "foo"), "key1", {"data": "value1"})
        store.put(("test", "bar"), "key2", {"data": "value2"})

        ops = [
            GetOp(namespace=("test", "foo"), key="key1"),
            PutOp(namespace=("test", "bar"), key="key2", value={"data": "value2"}),
            ListNamespacesOp(
                match_conditions=None, max_depth=None, limit=10, offset=0
            ),
            GetOp(namespace=("test",), key="key3"),
        ]

        results = store.batch(ops)

        assert len(results) == 4
        assert isinstance(results[0], Item)
        assert results[0].value == {"data": "value1"}
        assert results[0].key == "key1"
        assert results[1] is None  # Put operation returns None
        assert isinstance(results[2], list)
        assert len(results[2]) > 0  # Should contain our test namespaces
        assert results[3] is None  # Non-existent key returns None

    def test_batch_get_ops(self, store):
        """Test batch get operations (like test_batch_get_ops)."""
        # Setup test data
        store.put(("test",), "key1", {"data": "value1"})
        store.put(("test",), "key2", {"data": "value2"})

        ops = [
            GetOp(namespace=("test",), key="key1"),
            GetOp(namespace=("test",), key="key2"),
            GetOp(namespace=("test",), key="key3"),  # Non-existent key
        ]

        results = store.batch(ops)

        assert len(results) == 3
        assert results[0] is not None
        assert results[1] is not None
        assert results[2] is None
        assert results[0].key == "key1"
        assert results[1].key == "key2"

    def test_batch_put_ops(self, store):
        """Test batch put operations (like test_batch_put_ops)."""
        ops = [
            PutOp(namespace=("test",), key="key1", value={"data": "value1"}),
            PutOp(namespace=("test",), key="key2", value={"data": "value2"}),
            PutOp(namespace=("test",), key="key3", value=None),  # Delete operation
        ]

        results = store.batch(ops)
        assert len(results) == 3
        assert all(result is None for result in results)

        # Verify the puts worked
        item1 = store.get(("test",), "key1")
        item2 = store.get(("test",), "key2")
        item3 = store.get(("test",), "key3")

        assert item1 and item1.value == {"data": "value1"}
        assert item2 and item2.value == {"data": "value2"}
        assert item3 is None


class TestSearchOperations:
    """Test search operations with embeddings."""

    def test_semantic_search(self, store_with_embeddings):
        """Test semantic search with mock embeddings."""
        # Store documents
        docs = [
            {"text": "Python is a programming language"},
            {"text": "JavaScript is used for web development"},
            {"text": "Python is great for data science"},
        ]

        for i, doc in enumerate(docs):
            store_with_embeddings.put(
                ("docs",), f"doc{i}", doc, index=["text"]
            )

        # Search for Python-related content
        results = store_with_embeddings.search(
            ("docs",), query="Python programming", limit=10
        )

        assert len(results) > 0
        assert all(isinstance(r, SearchItem) for r in results)
        # First result should be Python-related (deterministic with mock embeddings)
        assert "Python" in results[0].value.get("text", "")

    def test_search_with_filter(self, store_with_embeddings):
        """Test search with metadata filters."""
        # Store documents with metadata
        docs = [
            {"text": "Python tutorial", "category": "tutorial", "level": "beginner"},
            {"text": "Python advanced", "category": "tutorial", "level": "advanced"},
            {"text": "JavaScript guide", "category": "guide", "level": "beginner"},
        ]

        for i, doc in enumerate(docs):
            store_with_embeddings.put(
                ("docs",), f"doc{i}", doc, index=["text"]
            )

        # Search with filter
        results = store_with_embeddings.search(
            ("docs",),
            query="Python",
            filter={"category": "tutorial"},
            limit=10,
        )

        assert len(results) > 0
        # All results should have category=tutorial
        for result in results:
            assert result.value.get("category") == "tutorial"

    def test_search_without_a_query_filters_on_the_server(self, store):
        """No query vector: the store lists the table with the filter applied server-side,
        newest first, and stops at `limit`."""
        namespace = (f"nofilterq{int(time.time() * 1000)}",)
        for i in range(12):
            store.put(namespace, f"k{i}", {"kind": "a" if i % 3 == 0 else "b", "n": i})

        found = store.search(namespace, filter={"kind": "a"}, limit=10)
        assert sorted(item.key for item in found) == ["k0", "k3", "k6", "k9"]

        first_two = store.search(namespace, filter={"kind": "b"}, limit=2)
        assert len(first_two) == 2 and all(item.value["kind"] == "b" for item in first_two)

    def test_search_with_comparison_filter(self, store_with_embeddings):
        """Test search with comparison operators."""
        # Store documents with ratings
        docs = [
            {"text": "Good tutorial", "rating": 4.5},
            {"text": "Great guide", "rating": 4.8},
            {"text": "Basic intro", "rating": 3.2},
        ]

        for i, doc in enumerate(docs):
            store_with_embeddings.put(
                ("docs",), f"doc{i}", doc, index=["text"]
            )

        # Search with rating filter
        results = store_with_embeddings.search(
            ("docs",),
            query="tutorial",
            filter={"rating": {"$gte": 4.0}},
            limit=10,
        )

        assert len(results) > 0
        # All results should have rating >= 4.0
        for result in results:
            assert result.value.get("rating", 0) >= 4.0


class TestNamespaceOperations:
    """Test namespace management."""

    def test_list_namespaces(self, store):
        """Test listing all namespaces."""
        # Create items in different namespaces
        store.put(("users", "alice"), "prefs", {"theme": "dark"})
        store.put(("users", "bob"), "prefs", {"theme": "light"})
        store.put(("docs",), "doc1", {"title": "Guide"})

        # List all namespaces
        namespaces = store.list_namespaces()

        assert len(namespaces) >= 3
        assert ("users", "alice") in namespaces
        assert ("users", "bob") in namespaces
        assert ("docs",) in namespaces

    def test_list_namespaces_with_prefix(self, store):
        """Test listing namespaces with prefix filter."""
        # Create items
        store.put(("users", "alice"), "prefs", {"theme": "dark"})
        store.put(("users", "bob"), "prefs", {"theme": "light"})
        store.put(("docs",), "doc1", {"title": "Guide"})

        # List with prefix
        namespaces = store.list_namespaces(prefix=("users",))

        assert len(namespaces) >= 2
        assert ("users", "alice") in namespaces
        assert ("users", "bob") in namespaces
        assert ("docs",) not in namespaces

    def test_list_namespaces_with_max_depth(self, store):
        """Test listing namespaces with max depth."""
        # Create deep namespace
        store.put(("a", "b", "c", "d"), "key", {"data": "value"})

        # List with max_depth=2
        namespaces = store.list_namespaces(max_depth=2)

        # Should truncate to depth 2
        assert all(len(ns) <= 2 for ns in namespaces)


class TestAsyncOperations:
    """Test async operations."""

    @pytest.mark.asyncio
    async def test_async_get(self, store):
        """Test async get operation."""
        # Put item (sync)
        store.put(("test",), "key1", {"data": "value1"})

        # Get item (async)
        item = await store.aget(("test",), "key1")

        assert item is not None
        assert item.value == {"data": "value1"}

    @pytest.mark.asyncio
    async def test_async_put(self, store):
        """Test async put operation."""
        # Put item (async)
        await store.aput(("test",), "key1", {"data": "value1"})

        # Get item to verify
        item = await store.aget(("test",), "key1")

        assert item is not None
        assert item.value == {"data": "value1"}

    @pytest.mark.asyncio
    async def test_async_search(self, store_with_embeddings):
        """Test async search operation."""
        # Store documents (async)
        await store_with_embeddings.aput(
            ("docs",), "doc1", {"text": "Python tutorial"}, index=["text"]
        )

        # Search (async)
        results = await store_with_embeddings.asearch(
            ("docs",), query="Python", limit=10
        )

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_async_batch(self, store):
        """Test async batch operations."""
        ops = [
            PutOp(namespace=("test",), key="key1", value={"data": "value1"}),
            GetOp(namespace=("test",), key="key1"),
        ]

        results = await store.abatch(ops)

        assert len(results) == 2
        assert results[0] is None  # Put returns None
        # Note: Get in same batch may not see the put (depends on implementation)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_namespace(self, store):
        """Test that empty namespace raises error."""
        with pytest.raises(ValueError):
            store.put((), "key", {"data": "value"})

    def test_empty_value(self, store):
        """Test storing empty dict."""
        store.put(("test",), "key", {})
        item = store.get(("test",), "key")

        assert item is not None
        assert item.value == {}

    def test_nested_value(self, store):
        """Test storing nested data structures."""
        nested_data = {
            "user": {
                "name": "Alice",
                "preferences": {"theme": "dark", "language": "python"},
            },
            "tags": ["python", "programming"],
        }

        store.put(("test",), "key", nested_data)
        item = store.get(("test",), "key")

        assert item is not None
        assert item.value == nested_data

    def test_unicode_values(self, store):
        """Test storing unicode data."""
        unicode_data = {
            "text": "Hello 世界 🌍",
            "emoji": "😀🎉",
        }

        store.put(("test",), "key", unicode_data)
        item = store.get(("test",), "key")

        assert item is not None
        assert item.value == unicode_data


class TestMultiAgentScenario:
    """Test multi-agent memory isolation scenario."""

    def test_agent_memory_isolation(self, store):
        """Test that different agents have isolated memory."""
        # Agent 1 memories
        store.put(
            ("agents", "agent1", "memories"),
            "mem1",
            {"content": "Agent 1 memory"},
        )

        # Agent 2 memories
        store.put(
            ("agents", "agent2", "memories"),
            "mem1",
            {"content": "Agent 2 memory"},
        )

        # Retrieve agent-specific memories
        agent1_mem = store.get(("agents", "agent1", "memories"), "mem1")
        agent2_mem = store.get(("agents", "agent2", "memories"), "mem1")

        assert agent1_mem is not None
        assert agent2_mem is not None
        assert agent1_mem.value["content"] == "Agent 1 memory"
        assert agent2_mem.value["content"] == "Agent 2 memory"

    def test_list_agent_namespaces(self, store):
        """Test listing all agent namespaces."""
        # Create memories for multiple agents
        for i in range(3):
            store.put(
                ("agents", f"agent{i}", "session1"),
                "mem1",
                {"data": f"Memory {i}"},
            )

        # List all agent namespaces
        namespaces = store.list_namespaces(prefix=("agents",), max_depth=3)

        assert len(namespaces) >= 3
        for i in range(3):
            assert ("agents", f"agent{i}", "session1") in namespaces


def test_store_compatibility_with_langraph_patterns(store):
    """Test that DXMT store works with standard LangGraph patterns."""
    # This test demonstrates compatibility with LangGraph examples

    # Pattern 1: User preferences
    store.put(("users", "123"), "prefs", {"theme": "dark"})
    item = store.get(("users", "123"), "prefs")
    assert item and item.value["theme"] == "dark"

    # Pattern 2: Document storage
    store.put(("docs",), "doc1", {"text": "Content", "title": "Title"})
    doc = store.get(("docs",), "doc1")
    assert doc and doc.value["text"] == "Content"

    # Pattern 3: Namespace organization
    store.put(("project", "users", "alice"), "data", {"role": "admin"})
    data = store.get(("project", "users", "alice"), "data")
    assert data and data.value["role"] == "admin"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
