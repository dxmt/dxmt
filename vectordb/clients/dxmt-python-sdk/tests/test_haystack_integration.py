"""Unit tests for DXMT Haystack DocumentStore integration."""

from unittest.mock import Mock
import pytest

from dxmt.haystack import DXMTDocumentStore
from dxmt.exceptions import DXMTError


@pytest.fixture
def mock_client():
    client = Mock()
    client.get_table.return_value = {"name": "docs", "dimension": 3}
    client.create_table.return_value = {"name": "docs", "dimension": 3}
    client.insert_records.return_value = None
    client.search.return_value = []
    client.delete_records.return_value = None
    client.get_records.return_value = []
    client.list_tables.return_value = []
    return client


@pytest.fixture
def document_store(mock_client):
    return DXMTDocumentStore(
        client=mock_client,
        table_name="docs",
        embedding_dim=3,
    )


class TestDeleteDocumentsParamName:
    """P0 fix: delete_records must be called with ids=, not record_ids=."""

    def test_delete_documents_uses_ids_kwarg(self, document_store, mock_client):
        document_store.delete_documents(["doc1", "doc2"])
        mock_client.delete_records.assert_called_once()
        call = mock_client.delete_records.call_args
        # Must use 'ids' kwarg, not 'record_ids'
        assert "ids" in call[1], f"Expected 'ids' kwarg, got: {call}"
        assert "record_ids" not in call[1], "Should not use deprecated 'record_ids' kwarg"
        assert call[1]["ids"] == ["doc1", "doc2"]

    def test_delete_empty_list_does_not_raise(self, document_store, mock_client):
        # Should handle gracefully
        document_store.delete_documents([])
        # May or may not call delete_records with empty list — just must not raise


class TestGetDocumentByIdParamName:
    """P0 fix: get_records must be called with ids=, not record_ids=."""

    def test_get_records_uses_ids_kwarg(self, document_store, mock_client):
        document_store._get_document_by_id("doc1")
        mock_client.get_records.assert_called_once()
        call = mock_client.get_records.call_args
        assert "ids" in call[1], f"Expected 'ids' kwarg, got: {call}"
        assert "record_ids" not in call[1], "Should not use deprecated 'record_ids' kwarg"


class TestWriteAndCountDocuments:
    """Basic write/count sanity checks."""

    def test_write_documents_calls_insert(self, document_store, mock_client):
        try:
            from haystack.dataclasses import Document
        except ImportError:
            pytest.skip("haystack not installed")

        docs = [Document(content="hello", id="d1", meta={"source": "test"})]
        document_store.write_documents(docs)
        mock_client.insert_records.assert_called_once()

    def test_count_documents(self, document_store, mock_client):
        mock_client.get_table.return_value = {"name": "docs", "record_count": 5}
        count = document_store.count_documents()
        assert count == 5
