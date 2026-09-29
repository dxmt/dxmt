"""Unit tests for DXMT Vector Store integration with LlamaIndex."""

import pytest
from typing import List
from unittest.mock import Mock, MagicMock, patch

from llama_index.core.schema import TextNode, Document
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    FilterCondition,
)
from llama_index.vector_stores.dxmt import DXMTVectorStore


@pytest.fixture
def mock_dxmt_client():
    """Create a mock DXMT client."""
    with patch("llama_index.vector_stores.dxmt.base.DXMTClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock table operations - return table metadata when table exists
        mock_client.get_table.return_value = {
            "name": "test_table",
            "dimension": 128,
            "distance_metric": "cosine",
            "index_type": "hnsw"
        }
        mock_client.create_table.return_value = {"name": "test_table", "dimension": 128}
        mock_client.delete_table.return_value = None

        # Mock record operations
        mock_client.insert_records.return_value = None
        mock_client.delete_records.return_value = None

        # Mock search operations - return list of hits, not dict
        mock_client.search.return_value = []
        mock_client.hybrid_search.return_value = []
        mock_client.full_text_search.return_value = []

        yield mock_client


@pytest.fixture
def vector_store(mock_dxmt_client):
    """Create a DXMTVectorStore instance with mocked client."""
    return DXMTVectorStore(
        table_name="test_table",
        dimension=128,
        host="localhost",
        port=9999,
    )


@pytest.fixture
def sample_nodes():
    """Create sample nodes for testing."""
    return [
        TextNode(
            text="First test document",
            id_="node1",
            embedding=[0.1] * 128,
            metadata={"category": "test", "value": 1}
        ),
        TextNode(
            text="Second test document",
            id_="node2",
            embedding=[0.2] * 128,
            metadata={"category": "test", "value": 2}
        ),
        TextNode(
            text="Third test document",
            id_="node3",
            embedding=[0.3] * 128,
            metadata={"category": "other", "value": 3}
        ),
    ]


class TestDXMTVectorStoreInit:
    """Tests for DXMTVectorStore initialization."""

    def test_init_with_default_params(self, mock_dxmt_client):
        """Test initialization with default parameters."""
        store = DXMTVectorStore(
            table_name="test",
            dimension=128,
        )

        assert store.table_name == "test"
        assert store.dimension == 128
        assert store.host == "localhost"
        assert store.port == 9999
        assert store.index_type == "hnsw"
        assert store.distance_metric == "cosine"

    def test_init_with_custom_params(self, mock_dxmt_client):
        """Test initialization with custom parameters."""
        store = DXMTVectorStore(
            table_name="custom",
            dimension=256,
            host="example.com",
            port=8080,
            ssl=True,
            index_type="ivf",
            distance_metric="euclidean",
            batch_size=500,
        )

        assert store.table_name == "custom"
        assert store.dimension == 256
        assert store.host == "example.com"
        assert store.port == 8080
        assert store.ssl == True
        assert store.index_type == "ivf"
        assert store.distance_metric == "euclidean"
        assert store.batch_size == 500

    def test_init_with_existing_client(self):
        """Test initialization with an existing client."""
        mock_client = Mock()
        store = DXMTVectorStore(
            table_name="test",
            client=mock_client,
            dimension=128,
        )

        assert store.client == mock_client


class TestDXMTVectorStoreAdd:
    """Tests for adding nodes to vector store."""

    def test_add_single_node(self, vector_store, mock_dxmt_client, sample_nodes):
        """Test adding a single node."""
        node = sample_nodes[0]
        ids = vector_store.add([node])

        assert len(ids) == 1
        assert ids[0] == node.node_id

        # Verify insert was called
        mock_dxmt_client.insert_records.assert_called_once()

        # Verify record format
        call_args = mock_dxmt_client.insert_records.call_args
        records = call_args[0][1]  # Second positional argument
        assert len(records) == 1
        assert records[0]["id"] == node.node_id
        assert records[0]["vector"] == node.get_embedding()
        assert "payload" in records[0]

    def test_add_multiple_nodes(self, vector_store, mock_dxmt_client, sample_nodes):
        """Test adding multiple nodes."""
        ids = vector_store.add(sample_nodes)

        assert len(ids) == len(sample_nodes)
        assert ids == [node.node_id for node in sample_nodes]

        # Verify insert was called
        mock_dxmt_client.insert_records.assert_called_once()

    def test_add_with_custom_batch_size(self, vector_store, mock_dxmt_client, sample_nodes):
        """Test adding nodes with custom batch size."""
        ids = vector_store.add(sample_nodes, batch_size=1)

        assert len(ids) == len(sample_nodes)

        # Should have made 3 separate calls with batch_size=1
        assert mock_dxmt_client.insert_records.call_count == 3

    def test_add_creates_table_if_not_exists(self, vector_store, mock_dxmt_client, sample_nodes):
        """Test that add creates table if it doesn't exist."""
        # Override mock to simulate table doesn't exist
        from dxmt.exceptions import DXMTError
        mock_dxmt_client.get_table.side_effect = DXMTError("Table not found")
        vector_store._table_created = False  # Reset table created flag

        vector_store.add(sample_nodes)

        # Should have tried to get table
        mock_dxmt_client.get_table.assert_called()

        # Should have created table after get failed
        mock_dxmt_client.create_table.assert_called_once()


class TestDXMTVectorStoreQuery:
    """Tests for querying vector store."""

    def test_query_vector_search(self, vector_store, mock_dxmt_client):
        """Test vector similarity search."""
        # Setup mock response - DXMT returns list of hits directly
        mock_dxmt_client.search.return_value = [
            {
                "id": "node1",
                "score": 0.95,
                "payload": {"_node_content": "Test doc", "category": "test"}
            }
        ]

        query = VectorStoreQuery(
            query_embedding=[0.1] * 128,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.DEFAULT,
        )

        result = vector_store.query(query)

        assert len(result.nodes) == 1
        assert len(result.similarities) == 1
        assert len(result.ids) == 1
        assert result.ids[0] == "node1"
        assert result.similarities[0] == 0.95

        # Verify search was called correctly
        mock_dxmt_client.search.assert_called_once()

    def test_query_hybrid_search(self, vector_store, mock_dxmt_client):
        """Test hybrid search (vector + text)."""
        # Setup mock response - DXMT returns list of hits directly
        mock_dxmt_client.hybrid_search.return_value = [
            {
                "id": "node1",
                "score": 0.90,
                "payload": {"_node_content": "Test doc", "category": "test"}
            }
        ]

        query = VectorStoreQuery(
            query_embedding=[0.1] * 128,
            query_str="test document",
            similarity_top_k=5,
            mode=VectorStoreQueryMode.HYBRID,
            alpha=0.7,
        )

        result = vector_store.query(query)

        assert len(result.nodes) == 1

        # Verify hybrid_search was called with correct params
        mock_dxmt_client.hybrid_search.assert_called_once()
        call_args = mock_dxmt_client.hybrid_search.call_args
        assert call_args[1]["vector_weight"] == 0.7

    def test_query_text_search(self, vector_store, mock_dxmt_client):
        """Test full-text search."""
        # Setup mock response - DXMT returns list of hits directly
        mock_dxmt_client.full_text_search.return_value = [
            {
                "id": "node1",
                "score": 0.85,
                "payload": {"_node_content": "Test doc", "category": "test"}
            }
        ]

        query = VectorStoreQuery(
            query_str="test document",
            similarity_top_k=5,
            mode=VectorStoreQueryMode.TEXT_SEARCH,
        )

        result = vector_store.query(query)

        assert len(result.nodes) == 1

        # Verify full_text_search was called
        mock_dxmt_client.full_text_search.assert_called_once()

    def test_query_with_filters(self, vector_store, mock_dxmt_client):
        """Test query with metadata filters."""
        mock_dxmt_client.search.return_value = []

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="test", operator=FilterOperator.EQ)
            ]
        )

        query = VectorStoreQuery(
            query_embedding=[0.1] * 128,
            similarity_top_k=5,
            filters=filters,
        )

        result = vector_store.query(query)

        # Verify search was called with filter
        mock_dxmt_client.search.assert_called_once()
        call_args = mock_dxmt_client.search.call_args
        assert call_args[1]["filter"] is not None


class TestDXMTVectorStoreDelete:
    """Tests for deleting nodes from vector store.

    Two of these used to set `search.return_value = {"hits": [...]}` -- **a dict**, where the
    SDK returns a `list`. So the mock asserted the author's belief about the client rather
    than the client, and `delete()` passed here while raising
    `AttributeError: 'list' object has no attribute 'get'` against every real server there
    has ever been. `tests/test_integration.py::TestDeletionActuallyDeletes` is the half of
    this that a mock cannot do: it counts rows on a server afterwards.
    """

    def test_delete_by_ref_doc_id(self, vector_store, mock_dxmt_client):
        """Test deleting by reference document ID."""
        # `scroll` yields records, one dict each -- the shape `DXMTClient.scroll` returns.
        mock_dxmt_client.scroll.return_value = iter([
            {"id": "node1", "payload": {}},
            {"id": "node2", "payload": {}},
        ])

        vector_store.delete("doc123")

        mock_dxmt_client.scroll.assert_called_once()
        assert mock_dxmt_client.scroll.call_args.kwargs["filter"] == {"ref_doc_id": "doc123"}

        # Should have deleted the found nodes
        mock_dxmt_client.delete_records.assert_called_once_with(
            "test_table",
            ["node1", "node2"]
        )

    def test_delete_nodes_by_ids(self, vector_store, mock_dxmt_client):
        """Test deleting specific nodes by IDs."""
        node_ids = ["node1", "node2", "node3"]
        vector_store.delete_nodes(node_ids=node_ids)

        # Should have called delete_records directly
        mock_dxmt_client.delete_records.assert_called_once_with(
            "test_table",
            node_ids
        )

    def test_delete_nodes_by_filter(self, vector_store, mock_dxmt_client):
        """Test deleting nodes by metadata filter."""
        mock_dxmt_client.scroll.return_value = iter([{"id": "node1", "payload": {}}])

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="test", operator=FilterOperator.EQ)
            ]
        )

        vector_store.delete_nodes(filters=filters)

        mock_dxmt_client.scroll.assert_called_once()
        mock_dxmt_client.delete_records.assert_called_once_with("test_table", ["node1"])

    def test_clear(self, vector_store, mock_dxmt_client):
        """Test clearing all nodes (deleting table)."""
        vector_store.clear()

        # Should have deleted the table
        mock_dxmt_client.delete_table.assert_called_once_with("test_table")


class TestDXMTVectorStoreGetNodes:
    """Tests for retrieving nodes from vector store."""

    def test_get_nodes_by_ids(self, vector_store, mock_dxmt_client):
        """Test getting nodes by IDs."""
        mock_dxmt_client.get_records.return_value = [
            {
                "id": "node1",
                "vector": [0.1] * 128,
                "payload": {"_node_content": "Test", "category": "test"}
            }
        ]

        nodes = vector_store.get_nodes(node_ids=["node1"])

        assert len(nodes) == 1
        assert nodes[0].node_id == "node1"

        mock_dxmt_client.get_records.assert_called_once_with("test_table", ["node1"])

    def test_get_nodes_by_filter(self, vector_store, mock_dxmt_client):
        """Test getting nodes by metadata filter."""
        mock_dxmt_client.search.return_value = [
            {
                "id": "node1",
                "score": 1.0,
                "payload": {"_node_content": "Test", "category": "test"}
            }
        ]

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="test", operator=FilterOperator.EQ)
            ]
        )

        nodes = vector_store.get_nodes(filters=filters)

        assert len(nodes) == 1
        assert nodes[0].node_id == "node1"

        mock_dxmt_client.search.assert_called_once()


class TestDXMTVectorStoreFromExistingTable:
    """Tests for creating vector store from existing table."""

    @patch("llama_index.vector_stores.dxmt.base.DXMTClient")
    def test_from_existing_table(self, mock_client_class):
        """Test creating vector store from existing table."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock table info
        mock_client.get_table.return_value = {
            "name": "existing_table",
            "dimension": 256,
            "distance_metric": "euclidean",
            "index_type": "ivf",
            "enable_full_text_search": True,
        }

        store = DXMTVectorStore.from_existing_table(
            table_name="existing_table",
            host="localhost",
            port=9999,
        )

        assert store.table_name == "existing_table"
        assert store.dimension == 256
        assert store.distance_metric == "euclidean"
        assert store.index_type == "ivf"
        assert store.enable_full_text_search == True

        mock_client.get_table.assert_called_once_with("existing_table")


class TestDXMTVectorStoreProperties:
    """Tests for vector store properties and attributes."""

    def test_stores_text_property(self, vector_store):
        """Test that stores_text is True."""
        assert vector_store.stores_text == True

    def test_is_embedding_query_property(self, vector_store):
        """Test that is_embedding_query is True."""
        assert vector_store.is_embedding_query == True

    def test_client_property(self, vector_store):
        """Test client property accessor."""
        client = vector_store.client
        assert client is not None

    def test_class_name(self):
        """Test class name for serialization."""
        assert DXMTVectorStore.class_name() == "DXMTVectorStore"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
