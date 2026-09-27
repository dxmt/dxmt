"""
Unit tests for DXMT mem0 VectorStore adapter.

Tests the DXMTMemoryStore implementation against mem0's VectorStoreBase interface.
"""

from unittest.mock import Mock, patch, MagicMock
import pytest

# Try to import mem0, skip tests if not available
try:
    from mem0.vector_stores.base import VectorStoreBase
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="mem0 not installed")

from dxmt.mem0 import DXMTMemoryStore
from dxmt.exceptions import DXMTError, DXMTTableError


@pytest.fixture
def mock_dxmt_client():
    """Create a mock DXMT client."""
    client = Mock()

    # Mock table operations
    client.get_table.side_effect = DXMTError("Table not found")
    client.create_table.return_value = {"name": "test_collection", "dimension": 384}
    client.list_tables.return_value = [{"name": "test_collection"}]
    client.delete_table.return_value = None

    # Mock record operations
    client.insert_records.return_value = None
    client.search.return_value = []
    client.get_records.return_value = []
    client.update_record.return_value = None
    client.delete_records.return_value = None

    return client


@pytest.fixture
def memory_store(mock_dxmt_client):
    """Create a DXMTMemoryStore instance with mocked client."""
    return DXMTMemoryStore(
        collection_name="test_collection",
        client=mock_dxmt_client,
        embedding_model_dims=384,
        distance_metric="cosine"
    )


class TestDXMTMemoryStoreInitialization:
    """Test memory store initialization."""

    def test_init_creates_collection(self, mock_dxmt_client):
        """Test that initialization creates collection if not exists."""
        store = DXMTMemoryStore(
            collection_name="new_collection",
            client=mock_dxmt_client,
            embedding_model_dims=384
        )

        # Should try to get table first
        mock_dxmt_client.get_table.assert_called_with("new_collection")

        # Should create table when not found
        mock_dxmt_client.create_table.assert_called_once()
        call_args = mock_dxmt_client.create_table.call_args
        assert call_args[1]["name"] == "new_collection"
        assert call_args[1]["dimension"] == 384
        assert call_args[1]["distance_metric"] == "cosine"

    def test_init_with_existing_collection(self, mock_dxmt_client):
        """Test initialization with existing collection."""
        # Mock existing table
        mock_dxmt_client.get_table.side_effect = None
        mock_dxmt_client.get_table.return_value = {"name": "existing"}

        store = DXMTMemoryStore(
            collection_name="existing",
            client=mock_dxmt_client,
            embedding_model_dims=384
        )

        # Should not create table
        assert mock_dxmt_client.create_table.call_count == 0

    def test_init_with_custom_distance_metric(self, mock_dxmt_client):
        """Test initialization with custom distance metric."""
        store = DXMTMemoryStore(
            collection_name="custom",
            client=mock_dxmt_client,
            embedding_model_dims=768,
            distance_metric="euclidean"
        )

        assert store.distance_metric == "euclidean"
        assert store.embedding_model_dims == 768


class TestDXMTMemoryStoreInsert:
    """Test insert operations."""

    def test_insert_basic(self, memory_store):
        """Test basic insert operation."""
        vectors = [[0.1, 0.2, 0.3] + [0.0] * 381]  # 384 dims
        payloads = [{"data": "test memory", "user_id": "user1"}]
        ids = ["mem1"]

        memory_store.insert(vectors=vectors, payloads=payloads, ids=ids)

        # Verify insert_records was called
        memory_store.client.insert_records.assert_called_once()
        call_args = memory_store.client.insert_records.call_args

        assert call_args[0][0] == "test_collection"
        records = call_args[0][1]
        assert len(records) == 1
        assert records[0]["id"] == "mem1"
        assert records[0]["vector"] == vectors[0]
        assert records[0]["payload"]["data"] == "test memory"

    def test_insert_multiple_vectors(self, memory_store):
        """Test inserting multiple vectors."""
        vectors = [
            [0.1] * 384,
            [0.2] * 384,
            [0.3] * 384,
        ]
        payloads = [
            {"data": "memory 1", "user_id": "user1"},
            {"data": "memory 2", "user_id": "user1"},
            {"data": "memory 3", "user_id": "user2"},
        ]
        ids = ["mem1", "mem2", "mem3"]

        memory_store.insert(vectors=vectors, payloads=payloads, ids=ids)

        call_args = memory_store.client.insert_records.call_args
        records = call_args[0][1]
        assert len(records) == 3
        assert all("created_at" in r["payload"] for r in records)
        assert all("updated_at" in r["payload"] for r in records)

    def test_insert_without_ids(self, memory_store):
        """Test insert generates IDs when not provided."""
        vectors = [[0.1] * 384]
        payloads = [{"data": "memory"}]

        memory_store.insert(vectors=vectors, payloads=payloads, ids=None)

        call_args = memory_store.client.insert_records.call_args
        records = call_args[0][1]
        assert len(records) == 1
        assert records[0]["id"] is not None  # Should have generated ID

    def test_insert_without_payloads(self, memory_store):
        """Test insert with empty payloads."""
        vectors = [[0.1] * 384]
        ids = ["mem1"]

        memory_store.insert(vectors=vectors, payloads=None, ids=ids)

        call_args = memory_store.client.insert_records.call_args
        records = call_args[0][1]
        assert len(records) == 1
        assert "created_at" in records[0]["payload"]

    def test_insert_dimension_mismatch(self, memory_store):
        """Test insert fails with wrong dimension."""
        vectors = [[0.1, 0.2, 0.3]]  # Wrong dimension (3 instead of 384)
        payloads = [{"data": "memory"}]
        ids = ["mem1"]

        with pytest.raises(ValueError, match="dimension mismatch"):
            memory_store.insert(vectors=vectors, payloads=payloads, ids=ids)

    def test_insert_empty_vectors(self, memory_store):
        """Test insert with empty vector list."""
        memory_store.insert(vectors=[], payloads=[], ids=[])

        # Should not call insert_records
        memory_store.client.insert_records.assert_not_called()

    def test_insert_adds_timestamps(self, memory_store):
        """Test that insert adds created_at and updated_at timestamps."""
        vectors = [[0.1] * 384]
        payloads = [{"data": "memory"}]
        ids = ["mem1"]

        memory_store.insert(vectors=vectors, payloads=payloads, ids=ids)

        call_args = memory_store.client.insert_records.call_args
        records = call_args[0][1]
        payload = records[0]["payload"]

        assert "created_at" in payload
        assert "updated_at" in payload
        assert payload["created_at"] == payload["updated_at"]  # Should be same for new insert


class TestDXMTMemoryStoreSearch:
    """Test search operations."""

    def test_search_basic(self, memory_store):
        """Test basic search operation."""
        query_vector = [[0.1] * 384]
        mock_results = [
            {"id": "mem1", "score": 0.95, "metadata": {"data": "memory 1"}},
            {"id": "mem2", "score": 0.85, "metadata": {"data": "memory 2"}},
        ]
        memory_store.client.search.return_value = mock_results

        results = memory_store.search(query="test", vectors=query_vector, limit=5)

        # Verify search was called
        memory_store.client.search.assert_called_once()
        call_args = memory_store.client.search.call_args
        assert call_args[1]["table_name"] == "test_collection"
        assert call_args[1]["vector"] == query_vector[0]
        assert call_args[1]["k"] == 5

        # Verify results format
        assert len(results) == 2
        assert results[0]["id"] == "mem1"
        assert results[0]["payload"]["data"] == "memory 1"
        assert 0 <= results[0]["score"] <= 1  # Should be normalized

    def test_search_with_filters(self, memory_store):
        """Test search with metadata filters."""
        query_vector = [[0.1] * 384]
        filters = {"user_id": "user1", "agent_id": "agent1"}

        memory_store.search(query="test", vectors=query_vector, limit=5, filters=filters)

        call_args = memory_store.client.search.call_args
        dxmt_filter = call_args[1]["filter"]

        # Verify filters were converted to DXMT format
        assert dxmt_filter is not None
        assert "must" in dxmt_filter
        assert len(dxmt_filter["must"]) == 2

    def test_search_score_normalization_cosine(self, memory_store):
        """Test score normalization for cosine similarity.

        DXMT returns cosine similarity (higher = more similar). A score of 0.95
        from DXMT should reach mem0 as 0.95, not be compressed by 1-score/2.
        """
        memory_store.distance_metric = "cosine"
        query_vector = [[0.1] * 384]

        # DXMT returns similarity 0.95 — must pass through unchanged
        mock_results = [{"id": "mem1", "score": 0.95, "metadata": {}}]
        memory_store.client.search.return_value = mock_results

        results = memory_store.search(query="", vectors=query_vector, limit=1)

        assert results[0]["score"] == 0.95

    def test_search_no_vectors(self, memory_store):
        """Test search fails without vectors."""
        with pytest.raises(ValueError, match="At least one query vector"):
            memory_store.search(query="test", vectors=[], limit=5)

    def test_search_dimension_mismatch(self, memory_store):
        """Test search fails with wrong dimension."""
        query_vector = [[0.1, 0.2, 0.3]]  # Wrong dimension

        with pytest.raises(ValueError, match="dimension mismatch"):
            memory_store.search(query="", vectors=query_vector, limit=5)


class TestDXMTMemoryStoreFilterConversion:
    """Test filter conversion from mem0 to DXMT format."""

    def test_convert_simple_equality(self, memory_store):
        """Test converting simple equality filters."""
        mem0_filters = {"user_id": "user1"}
        dxmt_filter = memory_store._convert_filters(mem0_filters)

        assert dxmt_filter is not None
        assert "must" in dxmt_filter
        assert len(dxmt_filter["must"]) == 1
        assert dxmt_filter["must"][0]["key"] == "user_id"
        assert dxmt_filter["must"][0]["operation"]["type"] == "Equal"
        assert dxmt_filter["must"][0]["operation"]["value"] == "user1"

    def test_convert_multiple_filters(self, memory_store):
        """Test converting multiple filters."""
        mem0_filters = {
            "user_id": "user1",
            "agent_id": "agent1",
            "run_id": "session1"
        }
        dxmt_filter = memory_store._convert_filters(mem0_filters)

        assert len(dxmt_filter["must"]) == 3

    def test_convert_range_filters(self, memory_store):
        """Test converting range filters."""
        mem0_filters = {
            "created_at": {
                "gte": "2024-01-01T00:00:00Z",
                "lte": "2024-12-31T23:59:59Z"
            }
        }
        dxmt_filter = memory_store._convert_filters(mem0_filters)

        # Should create two conditions for gte and lte
        assert len(dxmt_filter["must"]) == 2
        operations = [c["operation"]["type"] for c in dxmt_filter["must"]]
        assert "GreaterThanOrEqual" in operations
        assert "LessThanOrEqual" in operations

    def test_convert_in_filter(self, memory_store):
        """Test converting 'in' filter."""
        mem0_filters = {"category": {"in": ["tech", "science"]}}
        dxmt_filter = memory_store._convert_filters(mem0_filters)

        assert dxmt_filter["must"][0]["operation"]["type"] == "In"
        assert dxmt_filter["must"][0]["operation"]["value"] == ["tech", "science"]

    def test_convert_empty_filters(self, memory_store):
        """Test converting empty filters."""
        dxmt_filter = memory_store._convert_filters({})
        assert dxmt_filter is None

    def test_convert_none_filters(self, memory_store):
        """Test converting None filters."""
        dxmt_filter = memory_store._convert_filters(None)
        assert dxmt_filter is None


class TestDXMTMemoryStoreGetUpdate:
    """Test get and update operations."""

    def test_get_existing_vector(self, memory_store):
        """Test getting an existing vector."""
        mock_record = {
            "id": "mem1",
            "metadata": {"data": "memory 1", "user_id": "user1"}
        }
        memory_store.client.get_records.return_value = [mock_record]

        result = memory_store.get("mem1")

        assert result is not None
        assert result["id"] == "mem1"
        assert result["payload"]["data"] == "memory 1"
        assert result["score"] is None  # No score for direct retrieval

    def test_get_nonexistent_vector(self, memory_store):
        """Test getting a non-existent vector."""
        memory_store.client.get_records.return_value = []

        result = memory_store.get("nonexistent")

        assert result is None

    def test_update_vector_only(self, memory_store):
        """Test updating only the vector."""
        new_vector = [0.5] * 384

        memory_store.update("mem1", vector=new_vector, payload=None)

        memory_store.client.update_record.assert_called_once()
        call_args = memory_store.client.update_record.call_args
        assert call_args[1]["record_id"] == "mem1"
        assert call_args[1]["vector"] == new_vector

    def test_update_payload_only(self, memory_store):
        """Test updating only the payload."""
        new_payload = {"data": "updated memory"}

        memory_store.update("mem1", vector=None, payload=new_payload)

        call_args = memory_store.client.update_record.call_args
        assert call_args[1]["payload"]["data"] == "updated memory"
        assert "updated_at" in call_args[1]["payload"]

    def test_update_both_vector_and_payload(self, memory_store):
        """Test updating both vector and payload."""
        new_vector = [0.5] * 384
        new_payload = {"data": "updated"}

        memory_store.update("mem1", vector=new_vector, payload=new_payload)

        call_args = memory_store.client.update_record.call_args
        assert call_args[1]["vector"] == new_vector
        assert call_args[1]["payload"]["data"] == "updated"

    def test_update_dimension_mismatch(self, memory_store):
        """Test update fails with wrong dimension."""
        wrong_vector = [0.1, 0.2, 0.3]  # Wrong dimension

        with pytest.raises(ValueError, match="dimension mismatch"):
            memory_store.update("mem1", vector=wrong_vector, payload=None)


class TestDXMTMemoryStoreDelete:
    """Test delete operations."""

    def test_delete_single_vector(self, memory_store):
        """Test deleting a single vector."""
        memory_store.delete("mem1")

        memory_store.client.delete_records.assert_called_once_with(
            "test_collection", ["mem1"]
        )

    def test_delete_collection(self, memory_store):
        """Test deleting entire collection."""
        memory_store.delete_col()

        memory_store.client.delete_table.assert_called_once_with("test_collection")

    def test_reset_collection(self, memory_store):
        """Test resetting collection."""
        memory_store.client.delete_table.side_effect = None
        memory_store.client.get_table.side_effect = DXMTError("Not found")

        memory_store.reset()

        # Should delete and recreate
        memory_store.client.delete_table.assert_called_once()
        assert memory_store.client.create_table.call_count >= 1


class TestDXMTMemoryStoreList:
    """Test list operations."""

    def test_list_all_vectors(self, memory_store):
        """Test listing all vectors."""
        mock_results = [
            {"id": "mem1", "score": 0.9, "metadata": {"data": "memory 1"}},
            {"id": "mem2", "score": 0.8, "metadata": {"data": "memory 2"}},
        ]
        memory_store.client.search.return_value = mock_results

        results = memory_store.list(filters=None, limit=100)

        assert len(results) == 2
        assert results[0]["id"] == "mem1"
        assert results[0]["score"] is None  # No score for listing

    def test_list_with_filters(self, memory_store):
        """Test listing with filters."""
        filters = {"user_id": "user1"}
        mock_results = [{"id": "mem1", "score": 0.9, "metadata": {"user_id": "user1"}}]
        memory_store.client.search.return_value = mock_results

        results = memory_store.list(filters=filters, limit=50)

        call_args = memory_store.client.search.call_args
        assert call_args[1]["k"] == 50
        assert call_args[1]["filter"] is not None

    def test_list_collections(self, memory_store):
        """Test listing all collections."""
        mock_tables = [
            {"name": "collection1"},
            {"name": "collection2"},
            {"name": "test_collection"},
        ]
        memory_store.client.list_tables.return_value = mock_tables

        collections = memory_store.list_cols()

        assert len(collections) == 3
        assert "test_collection" in collections

    def test_col_info(self, memory_store):
        """Test getting collection info."""
        mock_info = {
            "name": "test_collection",
            "dimension": 384,
            "distance_metric": "cosine",
            "record_count": 100
        }
        memory_store.client.get_table.return_value = mock_info

        info = memory_store.col_info()

        assert info["name"] == "test_collection"
        assert info["dimension"] == 384


class TestDXMTMemoryStoreScoreNormalization:
    """Test score normalization for different distance metrics."""

    def test_normalize_cosine_distance(self, memory_store):
        """Test normalizing cosine distance.

        DXMT returns cosine *similarity* in [0, 1] (higher = more similar).
        _normalize_score must pass it through unchanged, not apply 1-score/2.
        """
        memory_store.distance_metric = "cosine"

        # DXMT cosine similarity: 1.0 = identical, 0.0 = orthogonal/opposite
        assert memory_store._normalize_score(1.0) == 1.0   # Perfect match
        assert memory_store._normalize_score(0.9) == 0.9   # High similarity — must not be compressed
        assert memory_store._normalize_score(0.0) == 0.0   # No similarity
        assert memory_store._normalize_score(0.5) == 0.5   # Mid-range

    def test_normalize_euclidean_distance(self, memory_store):
        """Test normalizing Euclidean distance."""
        memory_store.distance_metric = "euclidean"

        # Euclidean: uses exponential decay
        import math
        assert memory_store._normalize_score(0.0) == 1.0  # Identical
        assert abs(memory_store._normalize_score(1.0) - math.exp(-1)) < 0.01

    def test_normalize_unknown_metric(self, memory_store):
        """Test normalizing with unknown metric returns raw score."""
        memory_store.distance_metric = "unknown"

        # Should return raw score
        assert memory_store._normalize_score(0.5) == 0.5


class TestDXMTMemoryStoreIntegration:
    """Integration tests with real operations (mocked client)."""

    def test_full_memory_lifecycle(self, memory_store):
        """Test complete memory lifecycle: insert, search, update, delete."""
        # 1. Insert
        vectors = [[0.1] * 384]
        payloads = [{"data": "initial memory", "user_id": "user1"}]
        ids = ["mem1"]
        memory_store.insert(vectors, payloads, ids)

        # 2. Search
        memory_store.client.search.return_value = [
            {"id": "mem1", "score": 0.95, "metadata": payloads[0]}
        ]
        results = memory_store.search(query="", vectors=vectors, limit=5)
        assert len(results) == 1
        assert results[0]["id"] == "mem1"

        # 3. Update
        memory_store.update("mem1", payload={"data": "updated memory"})

        # 4. Delete
        memory_store.delete("mem1")

        # Verify all operations were called
        assert memory_store.client.insert_records.called
        assert memory_store.client.search.called
        assert memory_store.client.update_record.called
        assert memory_store.client.delete_records.called

    def test_multi_user_isolation(self, memory_store):
        """Test that multi-user filtering works correctly."""
        # Insert memories for different users
        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [
            {"data": "alice memory", "user_id": "alice"},
            {"data": "bob memory", "user_id": "bob"}
        ]
        ids = ["mem_alice", "mem_bob"]

        memory_store.insert(vectors, payloads, ids)

        # Search for alice only
        memory_store.client.search.return_value = [
            {"id": "mem_alice", "score": 0.9, "metadata": payloads[0]}
        ]

        results = memory_store.search(
            query="",
            vectors=[[0.1] * 384],
            limit=5,
            filters={"user_id": "alice"}
        )

        # Verify filter was applied
        call_args = memory_store.client.search.call_args
        assert call_args[1]["filter"] is not None


# Run tests with: pytest -v dxmt-python-sdk/tests/test_mem0_adapter.py
