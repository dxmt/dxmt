"""Unit tests for DXMT LangChain VectorStore integration."""

import asyncio
from unittest.mock import Mock, patch
import pytest

pytest.importorskip("langchain_core", reason="langchain-core not installed")

from dxmt.langchain_vectorstore import DXMTVectorStore
from dxmt.exceptions import DXMTError


@pytest.fixture
def mock_client():
    client = Mock()
    client.get_table.return_value = {"name": "test", "dimension": 3}
    client.insert_records.return_value = None
    client.search.return_value = []
    client.delete_records.return_value = None
    return client


@pytest.fixture
def mock_embeddings():
    emb = Mock()
    emb.embed_query.return_value = [0.1, 0.2, 0.3]
    emb.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    return emb


@pytest.fixture
def vector_store(mock_client, mock_embeddings):
    return DXMTVectorStore(
        embedding=mock_embeddings,
        collection_name="test",
        client=mock_client,
    )


class TestDeleteUsesDeleteRecords:
    """P0 fix: delete() must call delete_records(ids=...) not the nonexistent delete_record()."""

    def test_delete_calls_delete_records_with_ids(self, vector_store, mock_client):
        vector_store.delete(ids=["id1", "id2"])
        mock_client.delete_records.assert_called_once_with(
            table_name="test", ids=["id1", "id2"]
        )

    def test_delete_none_returns_none(self, vector_store, mock_client):
        result = vector_store.delete(ids=None)
        assert result is None
        mock_client.delete_records.assert_not_called()

    def test_delete_empty_list(self, vector_store, mock_client):
        vector_store.delete(ids=[])
        mock_client.delete_records.assert_called_once_with(table_name="test", ids=[])

    def test_delete_never_calls_delete_record_singular(self, vector_store, mock_client):
        """Confirm the nonexistent method is never called."""
        vector_store.delete(ids=["x"])
        assert not hasattr(mock_client, "delete_record") or not mock_client.delete_record.called


class TestFilterConversion:
    """_convert_filter delegates to convert_simple_filter_to_dxmt."""

    def test_simple_equality(self, vector_store):
        result = vector_store._convert_filter({"category": "AI"})
        assert result == {
            "must": [{"key": "category", "operation": {"type": "Equal", "value": "AI"}}]
        }

    def test_mongodb_ne_operator(self, vector_store):
        result = vector_store._convert_filter({"status": {"$ne": "deleted"}})
        assert result["must"][0]["operation"]["type"] == "NotEqual"

    def test_mongodb_in_operator(self, vector_store):
        result = vector_store._convert_filter({"cat": {"$in": ["A", "B"]}})
        assert result["must"][0]["operation"]["type"] == "In"
        assert result["must"][0]["operation"]["value"] == ["A", "B"]

    def test_and_operator(self, vector_store):
        result = vector_store._convert_filter({
            "$and": [{"year": {"$gte": 2022}}, {"rating": {"$gte": 8.0}}]
        })
        assert "must" in result
        assert len(result["must"]) == 2

    def test_or_operator(self, vector_store):
        result = vector_store._convert_filter({
            "$or": [{"cat": "A"}, {"cat": "B"}]
        })
        assert "should" in result

    def test_dxmt_native_passthrough(self, vector_store):
        native = {"must": [{"key": "x", "operation": {"type": "Equal", "value": 1}}]}
        assert vector_store._convert_filter(native) == native


class TestSimilaritySearchByVector:
    """similarity_search_by_vector is required by LangChain 0.1+."""

    def test_calls_client_search(self, vector_store, mock_client):
        mock_client.search.return_value = [
            {"id": "1", "score": 0.9, "metadata": {"text": "hello", "source": "x"}}
        ]
        docs = vector_store.similarity_search_by_vector([0.1, 0.2, 0.3], k=3)
        mock_client.search.assert_called_once()
        call = mock_client.search.call_args
        assert call[1]["vector"] == [0.1, 0.2, 0.3]
        assert call[1]["k"] == 3
        assert len(docs) == 1
        assert docs[0].page_content == "hello"

    def test_with_filter(self, vector_store, mock_client):
        mock_client.search.return_value = []
        vector_store.similarity_search_by_vector(
            [0.1, 0.2, 0.3], k=5, filter={"cat": "AI"}
        )
        call = mock_client.search.call_args
        assert call[1]["filter"] is not None


class TestAsyncVariants:
    """Async wrappers run_in_executor so coroutines must return the sync result."""

    def test_aadd_texts_returns_ids(self, vector_store, mock_embeddings):
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        ids = asyncio.run(vector_store.aadd_texts(["hello"], ids=["id1"]))
        assert ids == ["id1"]

    def test_asimilarity_search_returns_docs(self, vector_store, mock_client):
        mock_client.search.return_value = [
            {"id": "1", "score": 0.8, "metadata": {"text": "foo"}}
        ]
        docs = asyncio.run(vector_store.asimilarity_search("query", k=1))
        assert len(docs) == 1

    def test_asimilarity_search_with_score(self, vector_store, mock_client):
        mock_client.search.return_value = [
            {"id": "1", "score": 0.75, "metadata": {"text": "bar"}}
        ]
        pairs = asyncio.run(
            vector_store.asimilarity_search_with_score("query", k=1)
        )
        assert len(pairs) == 1
        doc, score = pairs[0]
        assert score == 0.75
