"""Unit tests for DXMT LangGraph Store integration."""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

pytest.importorskip("langgraph", reason="optional LangGraph integration dependency is not installed")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dxmt.langgraph_store import DXMTLangGraphStore
from dxmt.client import DXMTClient
from langgraph.store.base import GetOp, SearchOp, PutOp, ListNamespacesOp, MatchCondition


@pytest.fixture
def mock_client():
    """Create a mock DXMT client."""
    client = Mock(spec=DXMTClient)
    return client


@pytest.fixture
def store(mock_client):
    """Create a store instance with mock client."""
    return DXMTLangGraphStore(
        client=mock_client,
        embedding_dims=384,
        embed_fn=lambda texts: [[0.1] * 384 for _ in texts],
        distance_metric="cosine",
    )


@pytest.fixture
def store_no_embeddings(mock_client):
    """Create a store without embedding support."""
    return DXMTLangGraphStore(client=mock_client)


class TestInitialization:
    """Test store initialization."""

    def test_init_with_embeddings(self, mock_client):
        """Test initialization with embedding configuration."""
        store = DXMTLangGraphStore(
            client=mock_client,
            embedding_dims=1536,
            embed_fn=lambda texts: [[0.1] * 1536 for _ in texts],
        )

        assert store.embedding_dims == 1536
        assert store.embed_fn is not None
        assert store.distance_metric == "cosine"

    def test_init_without_embeddings(self, mock_client):
        """Test initialization without embedding configuration."""
        store = DXMTLangGraphStore(client=mock_client)

        assert store.embedding_dims is None
        assert store.embed_fn is None

    def test_init_invalid_dims(self, mock_client):
        """Test initialization with invalid embedding dimensions."""
        with pytest.raises(ValueError, match="embedding_dims must be positive"):
            DXMTLangGraphStore(
                client=mock_client,
                embedding_dims=-1,
            )


class TestNamespaceMapping:
    """Test namespace to table name mapping."""

    def test_namespace_to_table(self, store):
        """Test converting namespace to table name."""
        namespace = ("users", "profiles")
        table_name = store._namespace_to_table(namespace)

        assert table_name == "langgraph_users_profiles"

    def test_table_to_namespace(self, store):
        """Test converting table name back to namespace."""
        table_name = "langgraph_users_profiles"
        namespace = store._table_to_namespace(table_name)

        assert namespace == ("users", "profiles")

    def test_roundtrip_conversion(self, store):
        """Test namespace ↔ table name roundtrip."""
        original_namespace = ("agent", "memory", "session_123")
        table_name = store._namespace_to_table(original_namespace)
        recovered_namespace = store._table_to_namespace(table_name)

        assert recovered_namespace == original_namespace


class TestRecordIdGeneration:
    """Test record ID generation."""

    def test_make_record_id(self, store):
        """Test generating record ID from namespace and key."""
        namespace = ("users", "alice")
        key = "preferences"

        record_id = store._make_record_id(namespace, key)

        assert isinstance(record_id, str)
        assert len(record_id) == 16  # SHA256 truncated to 16 chars

    def test_record_id_uniqueness(self, store):
        """Test that different namespace/key combinations produce different IDs."""
        id1 = store._make_record_id(("users", "alice"), "prefs")
        id2 = store._make_record_id(("users", "bob"), "prefs")
        id3 = store._make_record_id(("users", "alice"), "settings")

        assert id1 != id2
        assert id1 != id3
        assert id2 != id3

    def test_record_id_stability(self, store):
        """Test that same namespace/key always produces same ID."""
        namespace = ("users", "alice")
        key = "preferences"

        id1 = store._make_record_id(namespace, key)
        id2 = store._make_record_id(namespace, key)

        assert id1 == id2


class TestFilterConversion:
    """Test LangGraph filter to DXMT filter conversion."""

    def test_simple_equality(self, store):
        """Test simple equality filter."""
        lg_filter = {"status": "active"}
        dxmt_filter = store._convert_filter(lg_filter)

        assert "must" in dxmt_filter
        assert len(dxmt_filter["must"]) == 1
        assert dxmt_filter["must"][0]["key"] == "status"
        assert dxmt_filter["must"][0]["operation"]["type"] == "Equal"
        assert dxmt_filter["must"][0]["operation"]["value"] == "active"

    def test_greater_than(self, store):
        """Test greater than comparison."""
        lg_filter = {"score": {"$gt": 4.5}}
        dxmt_filter = store._convert_filter(lg_filter)

        assert dxmt_filter["must"][0]["operation"]["type"] == "GreaterThan"
        assert dxmt_filter["must"][0]["operation"]["value"] == 4.5

    def test_greater_than_or_equal(self, store):
        """Test greater than or equal comparison."""
        lg_filter = {"score": {"$gte": 5.0}}
        dxmt_filter = store._convert_filter(lg_filter)

        assert dxmt_filter["must"][0]["operation"]["type"] == "GreaterThanOrEqual"

    def test_less_than(self, store):
        """Test less than comparison."""
        lg_filter = {"count": {"$lt": 10}}
        dxmt_filter = store._convert_filter(lg_filter)

        assert dxmt_filter["must"][0]["operation"]["type"] == "LessThan"

    def test_less_than_or_equal(self, store):
        """Test less than or equal comparison."""
        lg_filter = {"count": {"$lte": 10}}
        dxmt_filter = store._convert_filter(lg_filter)

        assert dxmt_filter["must"][0]["operation"]["type"] == "LessThanOrEqual"

    def test_multiple_conditions(self, store):
        """Test multiple filter conditions."""
        lg_filter = {"status": "active", "score": {"$gte": 4.0}}
        dxmt_filter = store._convert_filter(lg_filter)

        assert len(dxmt_filter["must"]) == 2

    def test_empty_filter(self, store):
        """Test empty filter."""
        dxmt_filter = store._convert_filter({})
        assert dxmt_filter == {}

    def test_none_filter(self, store):
        """Test None filter."""
        dxmt_filter = store._convert_filter(None)
        assert dxmt_filter == {}


class TestTextExtraction:
    """Test text extraction from values."""

    def test_extract_simple_field(self, store):
        """Test extracting text from simple field."""
        value = {"content": "Hello world"}
        texts = store._extract_texts(value, ["content"])

        assert len(texts) == 1
        assert texts[0] == "Hello world"

    def test_extract_nested_field(self, store):
        """Test extracting text from nested field."""
        value = {"metadata": {"title": "Test Title"}}
        texts = store._extract_texts(value, ["metadata.title"])

        assert len(texts) == 1
        assert texts[0] == "Test Title"

    def test_extract_multiple_fields(self, store):
        """Test extracting text from multiple fields."""
        value = {"title": "Title", "content": "Content"}
        texts = store._extract_texts(value, ["title", "content"])

        assert len(texts) == 2
        assert "Title" in texts
        assert "Content" in texts

    def test_extract_whole_document(self, store):
        """Test extracting text from entire document."""
        value = {"field1": "Value 1", "field2": "Value 2"}
        texts = store._extract_texts(value, ["$"])

        # Should get JSON representation
        assert len(texts) >= 1


class TestGetOperation:
    """Test GetOp handling."""

    def test_get_existing_item(self, store, mock_client):
        """Test getting an existing item."""
        namespace = ("users", "alice")
        key = "preferences"

        # Mock table check
        mock_client.get_table.return_value = {"name": "langgraph_users_alice"}

        # Mock record retrieval
        mock_client.get_records.return_value = [
            {
                "id": "test_id",
                "payload": {
                    "_namespace": ["users", "alice"],
                    "_key": "preferences",
                    "_created_at": "2026-01-28T10:00:00Z",
                    "_updated_at": "2026-01-28T10:00:00Z",
                    "theme": "dark",
                },
            }
        ]

        # Execute get operation
        op = GetOp(namespace, key)
        result = store._handle_get(op)

        assert result is not None
        assert result.namespace == namespace
        assert result.key == key
        assert result.value["theme"] == "dark"

    def test_get_nonexistent_item(self, store, mock_client):
        """Test getting a non-existent item."""
        namespace = ("users", "bob")
        key = "preferences"

        # Mock table check
        mock_client.get_table.return_value = {"name": "langgraph_users_bob"}

        # Mock empty result
        mock_client.get_records.return_value = []

        # Execute get operation
        op = GetOp(namespace, key)
        result = store._handle_get(op)

        assert result is None

    def test_get_nonexistent_table(self, store, mock_client):
        """Test getting from non-existent table."""
        from dxmt.exceptions import DXMTTableError

        namespace = ("nonexistent",)
        key = "test"

        # Mock table not found
        mock_client.get_table.side_effect = DXMTTableError("Table not found")

        # Execute get operation
        op = GetOp(namespace, key)
        result = store._handle_get(op)

        assert result is None


class TestPutOperation:
    """Test PutOp handling."""

    def test_put_new_item(self, store, mock_client):
        """Test putting a new item."""
        namespace = ("users", "alice")
        key = "preferences"
        value = {"theme": "dark", "language": "python"}

        # Mock table operations
        mock_client.get_table.return_value = {"name": "langgraph_users_alice"}
        mock_client.get_records.return_value = []  # No existing record
        mock_client.insert_records.return_value = {"success": True}

        # Execute put operation
        op = PutOp(namespace, key, value, index=False)
        store._handle_put(op)

        # Verify insert was called
        mock_client.insert_records.assert_called_once()

    def test_put_with_indexing(self, store, mock_client):
        """Test putting an item with semantic indexing."""
        namespace = ("docs",)
        key = "doc1"
        value = {"content": "Test document"}

        # Mock table operations
        mock_client.get_table.return_value = {"name": "langgraph_docs"}
        mock_client.get_records.return_value = []
        mock_client.insert_records.return_value = {"success": True}

        # Execute put operation with indexing
        op = PutOp(namespace, key, value, index=["content"])
        store._handle_put(op)

        # Verify insert was called with vector
        call_args = mock_client.insert_records.call_args
        records = call_args[0][1]  # Second positional argument
        assert "vector" in records[0]

    def test_put_delete_item(self, store, mock_client):
        """Test deleting an item (value=None)."""
        namespace = ("users", "alice")
        key = "preferences"

        # Mock table operations
        mock_client.get_table.return_value = {"name": "langgraph_users_alice"}
        mock_client.delete_records.return_value = {"success": True}

        # Execute delete operation
        op = PutOp(namespace, key, None)
        store._handle_put(op)

        # Verify delete was called
        mock_client.delete_records.assert_called_once()


class TestSearchOperation:
    """Test SearchOp handling."""

    def test_search_with_query(self, store, mock_client):
        """Test searching with semantic query."""
        namespace_prefix = ("docs",)
        query = "test query"

        # Mock table listing
        mock_client.list_tables.return_value = [{"name": "langgraph_docs"}]

        # Mock search results
        mock_client.search.return_value = [
            {
                "id": "doc1",
                "score": 0.95,
                "payload": {
                    "_namespace": ["docs"],
                    "_key": "doc1",
                    "_created_at": "2026-01-28T10:00:00Z",
                    "_updated_at": "2026-01-28T10:00:00Z",
                    "content": "Test document",
                },
            }
        ]

        # Execute search operation
        op = SearchOp(namespace_prefix, query=query, limit=10)
        results = store._handle_search(op)

        assert len(results) == 1
        assert results[0].score == 0.95
        assert results[0].value["content"] == "Test document"

    def test_search_with_filter(self, store, mock_client):
        """Test searching with metadata filter."""
        namespace_prefix = ("docs",)
        filter_dict = {"category": "technical"}

        # Mock table listing
        mock_client.list_tables.return_value = [{"name": "langgraph_docs"}]

        # Mock search results
        mock_client.search.return_value = []

        # Execute search operation
        op = SearchOp(namespace_prefix, filter=filter_dict, limit=10, query="test")
        results = store._handle_search(op)

        # Verify search was called with converted filter
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert "filter" in call_args[1]


class TestListNamespacesOperation:
    """Test ListNamespacesOp handling."""

    def test_list_all_namespaces(self, store, mock_client):
        """Test listing all namespaces."""
        # Mock table listing
        mock_client.list_tables.return_value = [
            {"name": "langgraph_users_alice"},
            {"name": "langgraph_users_bob"},
            {"name": "langgraph_docs"},
        ]

        # Execute list operation
        op = ListNamespacesOp()
        namespaces = store._handle_list_namespaces(op)

        assert len(namespaces) == 3
        assert ("users", "alice") in namespaces
        assert ("users", "bob") in namespaces
        assert ("docs",) in namespaces

    def test_list_with_prefix(self, store, mock_client):
        """Test listing namespaces with prefix filter."""
        # Mock table listing
        mock_client.list_tables.return_value = [
            {"name": "langgraph_users_alice"},
            {"name": "langgraph_users_bob"},
            {"name": "langgraph_docs"},
        ]

        # Execute list operation with prefix
        prefix_condition = MatchCondition(match_type="prefix", path=("users",))
        op = ListNamespacesOp(match_conditions=(prefix_condition,))
        namespaces = store._handle_list_namespaces(op)

        assert len(namespaces) == 2
        assert ("users", "alice") in namespaces
        assert ("users", "bob") in namespaces
        assert ("docs",) not in namespaces

    def test_list_with_max_depth(self, store, mock_client):
        """Test listing namespaces with max depth."""
        # Mock table listing
        mock_client.list_tables.return_value = [
            {"name": "langgraph_users_alice_session_123"},
            {"name": "langgraph_users_bob_session_456"},
        ]

        # Execute list operation with max_depth
        op = ListNamespacesOp(max_depth=2)
        namespaces = store._handle_list_namespaces(op)

        # Should truncate to depth 2
        assert all(len(ns) <= 2 for ns in namespaces)


class TestBatchOperations:
    """Test batch operation handling."""

    def test_batch_mixed_operations(self, store, mock_client):
        """Test batch with mixed operation types."""
        # Mock table operations
        mock_client.get_table.return_value = {"name": "test_table"}
        mock_client.get_records.return_value = []
        mock_client.insert_records.return_value = {"success": True}

        # Create mixed operations
        ops = [
            PutOp(("test",), "key1", {"data": "value1"}),
            GetOp(("test",), "key1"),
            PutOp(("test",), "key2", {"data": "value2"}),
        ]

        # Execute batch
        results = store.batch(ops)

        # Should return 3 results (None for Put, Item for Get)
        assert len(results) == 3

    def test_batch_empty(self, store):
        """Test batch with empty operations list."""
        results = store.batch([])
        assert len(results) == 0


class TestAsyncOperations:
    """Test async operation handling."""

    @pytest.mark.asyncio
    async def test_async_get(self, store, mock_client):
        """Test async get operation."""
        namespace = ("users", "alice")
        key = "preferences"

        # Mock async table check
        async def mock_aget_table(name):
            return {"name": name}

        mock_client.aget_table = mock_aget_table

        # Mock async record retrieval
        async def mock_aget_records(table_name, ids):
            return [
                {
                    "id": ids[0],
                    "payload": {
                        "_namespace": ["users", "alice"],
                        "_key": "preferences",
                        "_created_at": "2026-01-28T10:00:00Z",
                        "_updated_at": "2026-01-28T10:00:00Z",
                        "theme": "dark",
                    },
                }
            ]

        mock_client.aget_records = mock_aget_records

        # Execute async get
        op = GetOp(namespace, key)
        result = await store._ahandle_get(op)

        assert result is not None
        assert result.value["theme"] == "dark"

    @pytest.mark.asyncio
    async def test_async_batch(self, store, mock_client):
        """Test async batch operations."""
        # Mock async methods
        async def mock_aget_table(name):
            return {"name": name}

        async def mock_aget_records(table_name, ids):
            return []

        async def mock_ainsert_records(table_name, records, mode="standard"):
            return {"success": True}

        mock_client.aget_table = mock_aget_table
        mock_client.aget_records = mock_aget_records
        mock_client.ainsert_records = mock_ainsert_records

        # Create operations
        ops = [
            PutOp(("test",), "key1", {"data": "value1"}),
            GetOp(("test",), "key1"),
        ]

        # Execute async batch
        results = await store.abatch(ops)

        assert len(results) == 2


class TestNamespaceMatching:
    """Test namespace pattern matching."""

    def test_matches_prefix(self, store):
        """Test prefix matching."""
        namespace = ("users", "alice", "session")
        path = ("users", "alice")

        matches = store._matches_path(namespace, path, prefix=True)
        assert matches is True

    def test_matches_prefix_wildcard(self, store):
        """Test prefix matching with wildcard."""
        namespace = ("users", "alice", "session")
        path = ("users", "*", "session")

        matches = store._matches_path(namespace, path, prefix=True)
        assert matches is True

    def test_matches_suffix(self, store):
        """Test suffix matching."""
        namespace = ("users", "alice", "session")
        path = ("alice", "session")

        matches = store._matches_path(namespace, path, prefix=False)
        assert matches is True

    def test_no_match_prefix_too_long(self, store):
        """Test no match when path is longer than namespace."""
        namespace = ("users",)
        path = ("users", "alice")

        matches = store._matches_path(namespace, path, prefix=True)
        assert matches is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
