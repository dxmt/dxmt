"""`embedding=None` against a server that embeds.

Needs a server with an embedding provider behind it, which the ordinary test server has not
got, so this is skipped unless `DXMT_EMBEDDING_PORT` names one:

    DXMT_HOST=127.0.0.1 DXMT_EMBEDDING_PORT=19996 \\
    DXMT_EMBEDDING_MODEL=text-embedding-3-small \\
    pytest tests/integration_tests/test_server_embedding_live.py -v

`DXMT_EMBEDDING_MODEL` is the model a new table is created with (and
`DXMT_EMBEDDING_BASE_URL` its OpenAI-compatible endpoint, when it is not OpenAI's); the
server reads the key from its own environment. Both are passed through as
`server_embedding=`, so the test is also the example of how to use it.
"""

import contextlib
import os
import time
from collections.abc import Iterator

import pytest

from langchain_dxmt import DXMTVectorDB

PORT = os.getenv("DXMT_EMBEDDING_PORT")
HOST = os.getenv("DXMT_HOST", "localhost")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not PORT, reason="DXMT_EMBEDDING_PORT not set: no server with an embedding model"
    ),
]


def _server_embedding() -> dict:
    model = os.getenv("DXMT_EMBEDDING_MODEL")
    if not model:
        pytest.skip("DXMT_EMBEDDING_MODEL not set: which model should the table use?")
    config = {"model": model}
    base_url = os.getenv("DXMT_EMBEDDING_BASE_URL")
    if base_url:
        config["base_url"] = base_url
    return config


@pytest.fixture
def table_name() -> str:
    return f"langchain_server_embedding_{int(time.time() * 1000)}"


@pytest.fixture
def store(table_name: str) -> Iterator[DXMTVectorDB]:
    s = DXMTVectorDB(
        table_name=table_name,
        embedding=None,
        host=HOST,
        port=int(PORT),  # type: ignore[arg-type]
        server_embedding=_server_embedding(),
        enable_full_text_search=True,
    )
    try:
        yield s
    finally:
        with contextlib.suppress(Exception):
            s._client.delete_table(table_name)


def test_table_created_from_the_model(store: DXMTVectorDB) -> None:
    """No `dimension` was passed; the server asked the model and the store read it back."""
    assert store.embeddings is None
    assert isinstance(store._dimension, int) and store._dimension > 0
    table = store._client.get_table(store._table_name)
    assert table["embedding"]["model"] == os.environ["DXMT_EMBEDDING_MODEL"]


def test_add_and_search_by_text(store: DXMTVectorDB) -> None:
    ids = store.add_texts(
        ["The cat sat on the mat", "Quarterly revenue grew eight percent"],
        metadatas=[{"topic": "pets"}, {"topic": "finance"}],
        ids=["cat", "money"],
    )
    assert ids == ["cat", "money"]

    # Stored text comes back as page_content, from the payload.
    docs = store.get_by_ids(["cat"])
    assert docs[0].page_content == "The cat sat on the mat"
    assert docs[0].metadata == {"topic": "pets"}

    hits = store.similarity_search("a kitten on a rug", k=2)
    assert [d.id for d in hits][0] == "cat"

    with_scores = store.similarity_search_with_score("earnings went up", k=1)
    assert with_scores[0][0].id == "money"
    assert isinstance(with_scores[0][1], float)

    filtered = store.similarity_search("anything", k=2, filter={"topic": "finance"})
    assert [d.id for d in filtered] == ["money"]


def test_mmr_and_hybrid_by_text(store: DXMTVectorDB) -> None:
    store.add_texts(
        ["cats are small", "cats are furry", "the stock market fell"],
        ids=["c1", "c2", "m"],
    )

    mmr = store.max_marginal_relevance_search("cats", k=2, fetch_k=3, lambda_mult=0.0)
    assert len(mmr) == 2

    hybrid = store.hybrid_search("stock market", k=1)
    assert [d.id for d in hybrid] == ["m"]


def test_from_texts_and_from_existing_table(table_name: str) -> None:
    store = DXMTVectorDB.from_texts(
        ["one", "two"],
        embedding=None,
        table_name=table_name,
        host=HOST,
        port=int(PORT),  # type: ignore[arg-type]
        server_embedding=_server_embedding(),
    )
    try:
        assert len(store.similarity_search("one", k=2)) == 2

        # Reopened without a model config: the table already has one.
        again = DXMTVectorDB.from_existing_table(
            table_name, host=HOST, port=int(PORT)  # type: ignore[arg-type]
        )
        assert again.embeddings is None
        assert again._dimension == store._dimension
        assert len(again.similarity_search("two", k=1)) == 1
    finally:
        store._client.delete_table(table_name)


def test_by_vector_without_a_vector_is_refused(store: DXMTVectorDB) -> None:
    with pytest.raises(ValueError, match="needs a query vector"):
        store.similarity_search_by_vector(None)  # type: ignore[arg-type]


async def test_async_paths(store: DXMTVectorDB) -> None:
    await store.aadd_texts(["async hello", "async world"], ids=["h", "w"])
    hits = await store.asimilarity_search("hello", k=1)
    assert hits[0].id == "h"
    assert await store.adelete(["h"]) is True
    assert await store.aget_by_ids(["h"]) == []
