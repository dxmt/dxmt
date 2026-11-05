"""`embedding=None` means the server embeds. What the client is asked to send, with a mock.

The live counterpart is `tests/integration_tests/test_server_embedding_live.py`, which needs
a server with a model behind it. These check the wire shape the adapter produces -- a record
with `text` and no `vector`, a search with `text=` and no `vector=` -- without one.
"""

from unittest.mock import AsyncMock, Mock, create_autospec, patch

import pytest
from dxmt.exceptions import DXMTTableError

from langchain_dxmt import DXMTClient, DXMTVectorDB

MODEL = {"model": "text-embedding-3-small"}
TABLE = {"name": "docs", "dimension": 1536, "distance_metric": "cosine", "embedding": MODEL}
HIT = {"id": "id1", "score": 0.9, "payload": {"text": "hello", "source": "a"}}


@pytest.fixture
def client() -> Mock:
    """A client whose table already exists and embeds server-side."""
    c = Mock(spec=DXMTClient)
    c.get_table.return_value = TABLE
    c.search.return_value = [HIT]
    c.hybrid_search.return_value = [HIT]
    c.insert_records.return_value = {"success": True}
    c.asearch = AsyncMock(return_value=[HIT])
    c.ahybrid_search = AsyncMock(return_value=[HIT])
    c.ainsert_records = AsyncMock(return_value={"success": True})
    return c


@pytest.fixture
def store(client: Mock) -> DXMTVectorDB:
    return DXMTVectorDB(table_name="docs", embedding=None, client=client)


def test_creates_table_with_server_embedding() -> None:
    """A missing table is created with `embedding=server_embedding` and no dimension."""
    with patch("langchain_dxmt.vectorstores.DXMTClient") as MockClient:
        c = MockClient.return_value
        # Missing on the first look, there (with the model's width) after the create.
        c.get_table.side_effect = [DXMTTableError("TABLE_NOT_FOUND: docs"), TABLE]

        store = DXMTVectorDB(table_name="docs", embedding=None, server_embedding=MODEL)

        c.create_table.assert_called_once()
        kwargs = c.create_table.call_args.kwargs
        assert kwargs["name"] == "docs"
        assert kwargs["embedding"] == MODEL
        assert kwargs["dimension"] is None  # the server asks the model
        # Read back so `_dimension` is the model's width, not None.
        assert store._dimension == 1536


def test_neither_embedding_nor_server_embedding_nor_table_is_an_error() -> None:
    """Nothing to embed with and no table to read a model from: say what to pass."""
    with patch("langchain_dxmt.vectorstores.DXMTClient") as MockClient:
        c = MockClient.return_value
        c.get_table.side_effect = DXMTTableError("TABLE_NOT_FOUND: docs")

        with pytest.raises(ValueError, match="embedding=.*server_embedding="):
            DXMTVectorDB(table_name="docs", embedding=None)

        c.create_table.assert_not_called()


def test_existing_table_needs_neither(client: Mock) -> None:
    """A table that exists is enough: its model does the embedding."""
    store = DXMTVectorDB(table_name="docs", embedding=None, client=client)

    assert store.embeddings is None
    assert store._dimension == 1536
    client.create_table.assert_not_called()


def test_add_texts_sends_text_not_vector(store: DXMTVectorDB, client: Mock) -> None:
    ids = store.add_texts(["hello", "world"], ids=["a", "b"])

    assert ids == ["a", "b"]
    records = client.insert_records.call_args.kwargs["records"]
    assert records == [
        {"id": "a", "text": "hello", "payload": {"text": "hello"}},
        {"id": "b", "text": "world", "payload": {"text": "world"}},
    ]


async def test_aadd_texts_sends_text_not_vector(store: DXMTVectorDB, client: Mock) -> None:
    await store.aadd_texts(["hello"], ids=["a"])

    records = client.ainsert_records.call_args.kwargs["records"]
    assert records == [{"id": "a", "text": "hello", "payload": {"text": "hello"}}]


def test_similarity_search_uses_text(store: DXMTVectorDB, client: Mock) -> None:
    docs = store.similarity_search("hello", k=3, filter={"source": "a"})

    assert [d.page_content for d in docs] == ["hello"]
    assert docs[0].metadata == {"source": "a"}
    client.search.assert_called_once_with(
        table_name="docs", text="hello", k=3, filter={"source": "a"}
    )


def test_similarity_search_with_score_uses_text(
    store: DXMTVectorDB, client: Mock
) -> None:
    pairs = store.similarity_search_with_score("hello", k=1)

    assert pairs[0][1] == 0.9
    assert client.search.call_args.kwargs["text"] == "hello"
    assert "vector" not in client.search.call_args.kwargs


async def test_asimilarity_search_uses_text(store: DXMTVectorDB, client: Mock) -> None:
    docs = await store.asimilarity_search("hello", k=2)

    assert len(docs) == 1
    client.asearch.assert_awaited_once_with(
        table_name="docs", text="hello", k=2, filter=None
    )


def test_relevance_scores_work_without_embedding(store: DXMTVectorDB) -> None:
    """LangChain's threshold retriever goes through this; it must not need a vector."""
    pairs = store.similarity_search_with_relevance_scores("hello", k=1)

    assert pairs == [(pairs[0][0], 0.9)]


def test_k_zero_sends_nothing(store: DXMTVectorDB, client: Mock) -> None:
    assert store.similarity_search("hello", k=0) == []
    client.search.assert_not_called()


def test_mmr_uses_text(store: DXMTVectorDB, client: Mock) -> None:
    docs = store.max_marginal_relevance_search("hello", k=2, fetch_k=10, lambda_mult=0.3)

    assert len(docs) == 1
    client.search.assert_called_once_with(
        table_name="docs",
        vector=None,
        text="hello",
        k=2,
        filter=None,
        mmr={"lambda_mult": 0.3, "fetch_k": 10},
    )


def test_hybrid_search_uses_text_and_no_vector(client: Mock) -> None:
    store = DXMTVectorDB(
        table_name="docs", embedding=None, client=client, enable_full_text_search=True
    )

    docs = store.hybrid_search("hello", k=5, vector_weight=0.5)

    assert len(docs) == 1
    client.hybrid_search.assert_called_once_with(
        table_name="docs",
        vector=None,
        text="hello",
        vector_weight=0.5,
        k=5,
        filter=None,
    )


async def test_ahybrid_search_uses_text_and_no_vector(client: Mock) -> None:
    store = DXMTVectorDB(
        table_name="docs", embedding=None, client=client, enable_full_text_search=True
    )

    await store.ahybrid_search("hello", k=5)

    kwargs = client.ahybrid_search.call_args.kwargs
    assert kwargs["vector"] is None
    assert kwargs["text"] == "hello"


def test_by_vector_methods_still_take_a_vector(store: DXMTVectorDB, client: Mock) -> None:
    """A vector the caller computed is searched as such, embedding function or not."""
    docs = store.similarity_search_by_vector([0.1, 0.2], k=1)

    assert len(docs) == 1
    assert client.search.call_args.kwargs["vector"] == [0.1, 0.2]
    assert client.search.call_args.kwargs.get("text") is None


def test_by_vector_without_a_vector_is_a_clear_error(
    store: DXMTVectorDB, client: Mock
) -> None:
    """No embedding function and no vector: say so, before any request goes out."""
    with pytest.raises(ValueError, match="needs a query vector"):
        store.similarity_search_by_vector(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="needs a query vector"):
        store.max_marginal_relevance_search_by_vector(None)  # type: ignore[arg-type]
    client.search.assert_not_called()


async def test_aby_vector_without_a_vector_is_a_clear_error(
    store: DXMTVectorDB, client: Mock
) -> None:
    with pytest.raises(ValueError, match="needs a query vector"):
        await store.asimilarity_search_by_vector(None)  # type: ignore[arg-type]
    client.asearch.assert_not_called()


def test_from_texts_without_embedding() -> None:
    """`from_texts(embedding=None, server_embedding=...)`: no probe embedding, text records."""
    with patch("langchain_dxmt.vectorstores.DXMTClient") as MockClient:
        c = MockClient.return_value
        c.get_table.side_effect = [DXMTTableError("TABLE_NOT_FOUND: docs"), TABLE]
        c.insert_records.return_value = {"success": True}

        store = DXMTVectorDB.from_texts(
            ["hello", "world"],
            embedding=None,
            table_name="docs",
            server_embedding=MODEL,
        )

        assert store.embeddings is None
        assert c.create_table.call_args.kwargs["embedding"] == MODEL
        records = c.insert_records.call_args.kwargs["records"]
        assert [r["text"] for r in records] == ["hello", "world"]
        assert all("vector" not in r for r in records)


def test_dimension_not_required_with_server_embedding() -> None:
    """The model decides the width; the constructor must not demand `dimension`."""
    with patch("langchain_dxmt.vectorstores.DXMTClient") as MockClient:
        c = MockClient.return_value
        c.get_table.side_effect = [DXMTTableError("TABLE_NOT_FOUND: docs"), TABLE]

        DXMTVectorDB(table_name="docs", server_embedding=MODEL)  # no ValueError

    # But with vectors computed here and no model, a new table still needs one.
    with pytest.raises(ValueError, match="dimension must be provided"):
        DXMTVectorDB(table_name="docs", embedding=Mock())


def test_mmr_by_vector_drops_unknown_kwargs() -> None:
    """LangChain passes retriever kwargs through; the client takes a fixed set.

    `create_autospec` enforces `Client.search`'s signature, so forwarding `**kwargs` into
    it -- which this used to do -- raises `TypeError` here exactly as it did in the SDK.
    """
    client = create_autospec(DXMTClient, instance=True)
    client.get_table.return_value = {"dimension": 2, "distance_metric": "cosine"}
    client.search.return_value = [HIT]
    emb = Mock()
    emb.embed_query.return_value = [0.1, 0.2]

    store = DXMTVectorDB(table_name="docs", embedding=emb, dimension=2, client=client)

    docs = store.max_marginal_relevance_search_by_vector(
        [0.1, 0.2], k=1, fetch_k=5, lambda_mult=0.7, score_threshold=0.5, nonsense=1
    )

    assert len(docs) == 1
    client.search.assert_called_once_with(
        table_name="docs",
        vector=[0.1, 0.2],
        text=None,
        k=1,
        filter=None,
        mmr={"lambda_mult": 0.7, "fetch_k": 5},
    )

    # The retriever route, which is where those kwargs come from in practice.
    retriever = store.as_retriever(
        search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5, "score_threshold": 0.5}
    )
    assert len(retriever.invoke("hello")) == 1
