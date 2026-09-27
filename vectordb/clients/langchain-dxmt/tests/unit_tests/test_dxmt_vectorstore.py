"""Unit tests for DXMT VectorStore."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.documents import Document

from langchain_dxmt import DXMTVectorDB, DXMTClient


@pytest.fixture
def mock_client() -> Mock:
    """Create mock DXMT client."""
    client = Mock(spec=DXMTClient)
    # Setup async methods
    client.ainsert_records = AsyncMock(return_value={"success": True})
    client.asearch = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_embeddings() -> Mock:
    """Create mock embeddings function."""
    embeddings = Mock()
    embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    # Use AsyncMock for async methods
    embeddings.aembed_documents = AsyncMock(
        return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )
    embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return embeddings


def test_init_with_client(mock_client: Mock, mock_embeddings: Mock) -> None:
    """Test initialization with provided client."""
    mock_client.get_table.return_value = {
        "data": {"table": {"dimension": 3}}
    }

    store = DXMTVectorDB(
        table_name="test",
        embedding=mock_embeddings,
        client=mock_client,
        create_table_if_not_exists=False,
    )

    assert store._table_name == "test"
    assert store._dimension == 3


def test_init_without_dimension_error(mock_embeddings: Mock) -> None:
    """Test initialization without dimension raises error."""
    with pytest.raises(ValueError, match="dimension must be provided"):
        DXMTVectorDB(
            table_name="test",
            embedding=mock_embeddings,
            create_table_if_not_exists=True,
        )


def test_init_invalid_client_type(mock_embeddings: Mock) -> None:
    """Test initialization with invalid client type."""
    with pytest.raises(TypeError, match="client must be DXMTClient"):
        DXMTVectorDB(
            table_name="test",
            embedding=mock_embeddings,
            dimension=3,
            client="not a client",  # type: ignore
        )


def test_add_texts_without_embedding() -> None:
    """Test add_texts without embedding function."""
    with patch("langchain_dxmt.vectorstores.DXMTClient") as MockClient:
        mock_instance = Mock()
        MockClient.return_value = mock_instance
        mock_instance.get_table.return_value = {
            "data": {"table": {"dimension": 3}}
        }

        store = DXMTVectorDB(
            table_name="test",
            embedding=None,
            create_table_if_not_exists=False,
        )

        with pytest.raises(ValueError, match="embedding function is required"):
            store.add_texts(["text1", "text2"])


def test_add_texts_success(mock_client: Mock, mock_embeddings: Mock) -> None:
    """Test successful text addition."""
    mock_client.get_table.return_value = {
        "data": {"table": {"dimension": 3}}
    }
    mock_client.insert_records.return_value = {"success": True}

    store = DXMTVectorDB(
        table_name="test",
        embedding=mock_embeddings,
        client=mock_client,
        create_table_if_not_exists=False,
    )

    ids = store.add_texts(["text1", "text2"])

    assert len(ids) == 2
    mock_embeddings.embed_documents.assert_called_once_with(["text1", "text2"])
    mock_client.insert_records.assert_called_once()

    # Check records structure
    call_args = mock_client.insert_records.call_args
    records = call_args[1]["records"]
    assert len(records) == 2
    assert records[0]["payload"]["text"] == "text1"
    assert records[0]["vector"] == [0.1, 0.2, 0.3]


def test_add_texts_with_metadata(mock_client: Mock, mock_embeddings: Mock) -> None:
    """Test text addition with metadata."""
    mock_client.get_table.return_value = {
        "data": {"table": {"dimension": 3}}
    }
    mock_client.insert_records.return_value = {"success": True}

    store = DXMTVectorDB(
        table_name="test",
        embedding=mock_embeddings,
        client=mock_client,
        create_table_if_not_exists=False,
    )

    metadatas = [{"source": "doc1"}, {"source": "doc2"}]
    ids = store.add_texts(["text1", "text2"], metadatas=metadatas)

    assert len(ids) == 2

    # Check metadata is included
    call_args = mock_client.insert_records.call_args
    records = call_args[1]["records"]
    assert records[0]["payload"]["source"] == "doc1"
    assert records[1]["payload"]["source"] == "doc2"


def test_add_texts_with_ids(mock_client: Mock, mock_embeddings: Mock) -> None:
    """Test text addition with provided IDs."""
    mock_client.get_table.return_value = {
        "data": {"table": {"dimension": 3}}
    }
    mock_client.insert_records.return_value = {"success": True}

    store = DXMTVectorDB(
        table_name="test",
        embedding=mock_embeddings,
        client=mock_client,
        create_table_if_not_exists=False,
    )

    custom_ids = ["id1", "id2"]
    ids = store.add_texts(["text1", "text2"], ids=custom_ids)

    assert ids == custom_ids

    # Check IDs are used
    call_args = mock_client.insert_records.call_args
    records = call_args[1]["records"]
    assert records[0]["id"] == "id1"
    assert records[1]["id"] == "id2"


def test_add_texts_metadata_mismatch(mock_client: Mock, mock_embeddings: Mock) -> None:
    """Test error when metadata count doesn't match texts count."""
    mock_client.get_table.return_value = {
        "data": {"table": {"dimension": 3}}
    }

    store = DXMTVectorDB(
        table_name="test",
        embedding=mock_embeddings,
        client=mock_client,
        create_table_if_not_exists=False,
    )

    with pytest.raises(ValueError, match="Number of metadatas"):
        store.add_texts(["text1", "text2"], metadatas=[{"source": "doc1"}])


def test_delete_success(mock_client: Mock, mock_embeddings: Mock) -> None:
    """Test successful deletion."""
    mock_client.get_table.return_value = {
        "data": {"table": {"dimension": 3}}
    }
    mock_client.delete_records.return_value = {"success": True}

    store = DXMTVectorDB(
        table_name="test",
        embedding=mock_embeddings,
        client=mock_client,
        create_table_if_not_exists=False,
    )

    result = store.delete(ids=["id1", "id2"])

    assert result is True
    mock_client.delete_records.assert_called_once_with(
        table_name="test",
        ids=["id1", "id2"],
    )


def test_delete_no_ids(mock_client: Mock, mock_embeddings: Mock) -> None:
    """Test deletion with no IDs."""
    mock_client.get_table.return_value = {
        "data": {"table": {"dimension": 3}}
    }

    store = DXMTVectorDB(
        table_name="test",
        embedding=mock_embeddings,
        client=mock_client,
        create_table_if_not_exists=False,
    )

    result = store.delete(ids=[])

    assert result is False
    mock_client.delete_records.assert_not_called()


def test_get_by_ids(mock_client: Mock, mock_embeddings: Mock) -> None:
    """Test get documents by IDs."""
    mock_client.get_table.return_value = {
        "data": {"table": {"dimension": 3}}
    }
    mock_client.get_records.return_value = [
        {
            "id": "id1",
            "payload": {"text": "content1", "source": "doc1"}
        },
        {
            "id": "id2",
            "payload": {"text": "content2", "source": "doc2"}
        },
    ]

    store = DXMTVectorDB(
        table_name="test",
        embedding=mock_embeddings,
        client=mock_client,
        create_table_if_not_exists=False,
    )

    docs = store.get_by_ids(["id1", "id2"])

    assert len(docs) == 2
    assert docs[0].id == "id1"
    assert docs[0].page_content == "content1"
    assert docs[0].metadata == {"source": "doc1"}
    assert docs[1].id == "id2"
    assert docs[1].page_content == "content2"


def test_similarity_search(mock_client: Mock, mock_embeddings: Mock) -> None:
    """Test similarity search."""
    mock_client.get_table.return_value = {
        "data": {"table": {"dimension": 3}}
    }
    mock_client.search.return_value = [
        {
            "id": "id1",
            "score": 0.9,
            "payload": {"text": "result1", "source": "doc1"}
        },
    ]

    store = DXMTVectorDB(
        table_name="test",
        embedding=mock_embeddings,
        client=mock_client,
        create_table_if_not_exists=False,
    )

    docs = store.similarity_search("query", k=1)

    assert len(docs) == 1
    assert docs[0].page_content == "result1"
    assert docs[0].metadata == {"source": "doc1"}

    # Check embedding was called
    mock_embeddings.embed_query.assert_called_once_with("query")

    # Check search was called
    mock_client.search.assert_called_once()


def test_similarity_search_with_score(
    mock_client: Mock, mock_embeddings: Mock
) -> None:
    """Test similarity search with scores."""
    mock_client.get_table.return_value = {
        "data": {"table": {"dimension": 3}}
    }
    mock_client.search.return_value = [
        {
            "id": "id1",
            "score": 0.9,
            "payload": {"text": "result1", "source": "doc1"}
        },
        {
            "id": "id2",
            "score": 0.8,
            "payload": {"text": "result2", "source": "doc2"}
        },
    ]

    store = DXMTVectorDB(
        table_name="test",
        embedding=mock_embeddings,
        client=mock_client,
        create_table_if_not_exists=False,
    )

    docs_and_scores = store.similarity_search_with_score("query", k=2)

    assert len(docs_and_scores) == 2
    assert docs_and_scores[0][0].page_content == "result1"
    assert docs_and_scores[0][1] == 0.9
    assert docs_and_scores[1][0].page_content == "result2"
    assert docs_and_scores[1][1] == 0.8


def test_similarity_search_by_vector(mock_client: Mock, mock_embeddings: Mock) -> None:
    """Test similarity search by vector."""
    mock_client.get_table.return_value = {
        "data": {"table": {"dimension": 3}}
    }
    mock_client.search.return_value = [
        {
            "id": "id1",
            "score": 0.95,
            "payload": {"text": "result1"}
        },
    ]

    store = DXMTVectorDB(
        table_name="test",
        embedding=mock_embeddings,
        client=mock_client,
        create_table_if_not_exists=False,
    )

    embedding = [0.1, 0.2, 0.3]
    docs = store.similarity_search_by_vector(embedding, k=1)

    assert len(docs) == 1
    assert docs[0].page_content == "result1"

    # Check search was called with vector
    mock_client.search.assert_called_once()
    call_args = mock_client.search.call_args
    assert call_args[1]["vector"] == embedding


def test_similarity_search_with_filter(
    mock_client: Mock, mock_embeddings: Mock
) -> None:
    """Test similarity search with metadata filter."""
    mock_client.get_table.return_value = {
        "data": {"table": {"dimension": 3}}
    }
    mock_client.search.return_value = []

    store = DXMTVectorDB(
        table_name="test",
        embedding=mock_embeddings,
        client=mock_client,
        create_table_if_not_exists=False,
    )

    filter_dict = {"source": "doc1"}
    store.similarity_search("query", k=5, filter=filter_dict)

    # Check filter was passed
    mock_client.search.assert_called_once()
    call_args = mock_client.search.call_args
    assert call_args[1]["filter"] == filter_dict


def test_hybrid_search(mock_client: Mock, mock_embeddings: Mock) -> None:
    """Test hybrid search."""
    mock_client.get_table.return_value = {
        "data": {"table": {"dimension": 3}}
    }
    mock_client.hybrid_search.return_value = [
        {
            "id": "id1",
            "score": 0.85,
            "payload": {"text": "hybrid result"}
        },
    ]

    store = DXMTVectorDB(
        table_name="test",
        embedding=mock_embeddings,
        client=mock_client,
        enable_full_text_search=True,
        create_table_if_not_exists=False,
    )

    docs = store.hybrid_search("query", k=5, vector_weight=0.7)

    assert len(docs) == 1
    assert docs[0].page_content == "hybrid result"

    # Check hybrid_search was called with correct params
    mock_client.hybrid_search.assert_called_once()
    call_args = mock_client.hybrid_search.call_args
    assert call_args[1]["text"] == "query"
    assert call_args[1]["vector_weight"] == 0.7


def test_hybrid_search_not_enabled(mock_client: Mock, mock_embeddings: Mock) -> None:
    """Test hybrid search when full-text search not enabled."""
    mock_client.get_table.return_value = {
        "data": {"table": {"dimension": 3}}
    }

    store = DXMTVectorDB(
        table_name="test",
        embedding=mock_embeddings,
        client=mock_client,
        enable_full_text_search=False,
        create_table_if_not_exists=False,
    )

    with pytest.raises(ValueError, match="Full-text search not enabled"):
        store.hybrid_search("query")


def test_from_texts(mock_embeddings: Mock) -> None:
    """Test creating store from texts."""
    with patch("langchain_dxmt.vectorstores.DXMTClient") as MockClient:
        mock_instance = Mock()
        MockClient.return_value = mock_instance
        mock_instance.get_table.side_effect = Exception("Table not found")
        mock_instance.create_table.return_value = {"success": True}
        mock_instance.insert_records.return_value = {"success": True}

        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

        store = DXMTVectorDB.from_texts(
            texts=["text1", "text2"],
            embedding=mock_embeddings,
            table_name="new_table",
        )

        assert store._table_name == "new_table"
        assert store._dimension == 3  # Inferred from sample embedding

        # Check table was created
        mock_instance.create_table.assert_called_once()

        # Check texts were added
        mock_instance.insert_records.assert_called_once()


def test_from_existing_table(mock_embeddings: Mock) -> None:
    """Test connecting to existing table."""
    with patch("langchain_dxmt.vectorstores.DXMTClient") as MockClient:
        mock_instance = Mock()
        MockClient.return_value = mock_instance
        mock_instance.get_table.return_value = {
            "data": {"table": {"dimension": 128}}
        }

        store = DXMTVectorDB.from_existing_table(
            table_name="existing_table",
            embedding=mock_embeddings,
        )

        assert store._table_name == "existing_table"
        assert store._dimension == 128

        # Check table was NOT created
        mock_instance.create_table.assert_not_called()


@pytest.mark.asyncio
async def test_async_add_texts(mock_client: Mock, mock_embeddings: Mock) -> None:
    """Test async text addition."""
    mock_client.get_table.return_value = {
        "data": {"table": {"dimension": 3}}
    }
    mock_client.ainsert_records.return_value = {"success": True}

    store = DXMTVectorDB(
        table_name="test",
        embedding=mock_embeddings,
        client=mock_client,
        create_table_if_not_exists=False,
    )

    ids = await store.aadd_texts(["text1", "text2"])

    assert len(ids) == 2
    mock_embeddings.aembed_documents.assert_called_once()
    mock_client.ainsert_records.assert_called_once()


@pytest.mark.asyncio
async def test_async_similarity_search(
    mock_client: Mock, mock_embeddings: Mock
) -> None:
    """Test async similarity search."""
    mock_client.get_table.return_value = {
        "data": {"table": {"dimension": 3}}
    }
    mock_client.asearch.return_value = [
        {
            "id": "id1",
            "score": 0.9,
            "payload": {"text": "result1"}
        },
    ]

    store = DXMTVectorDB(
        table_name="test",
        embedding=mock_embeddings,
        client=mock_client,
        create_table_if_not_exists=False,
    )

    docs = await store.asimilarity_search("query", k=1)

    assert len(docs) == 1
    assert docs[0].page_content == "result1"

    mock_embeddings.aembed_query.assert_called_once_with("query")
    mock_client.asearch.assert_called_once()
