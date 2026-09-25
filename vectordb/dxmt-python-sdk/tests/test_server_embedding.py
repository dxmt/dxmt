"""The client half of server-side embedding: what goes on the wire, and what is refused here.

The server's behaviour is covered by `tests/server_embedding_test.rs`, which runs a real
OpenAI-protocol provider. What is left for the SDK is narrow and worth pinning anyway: that
`embedding` reaches the create body at all (a dropped field would create a table that answers
200 and then refuses every text insert), that a text search sends `text` and no `vector`, and
that asking for both is refused here rather than at the far end of a round trip.
"""

from unittest.mock import Mock

import pytest
from httpx import Response

from dxmt import Client

EMBEDDING = {"model": "text-embedding-3-small", "base_url": "http://127.0.0.1:1/v1"}


@pytest.fixture
def client_and_post():
    client = Client(host="localhost", port=9999)
    transport = Mock()
    transport.raise_for_status = Mock()
    transport.json = Mock(return_value={"success": True, "data": {"hits": []}})
    client._client.post = Mock(return_value=transport)
    yield client, client._client.post
    client.close()


def _body(post_mock):
    """The JSON the client sent, however it chose to encode it."""
    kwargs = post_mock.call_args.kwargs
    if "json" in kwargs:
        return kwargs["json"]
    import json

    return json.loads(kwargs["content"])


def test_create_table_sends_the_embedding_config(client_and_post) -> None:
    client, post = client_and_post
    client.create_table("docs", dimension=1536, embedding=EMBEDDING)
    assert _body(post)["embedding"] == EMBEDDING


def test_create_table_omits_it_when_unset(client_and_post) -> None:
    # Sent as null it would be a field the server has to decide the meaning of, where its
    # absence already means "this table takes vectors".
    client, post = client_and_post
    client.create_table("plain", dimension=4)
    assert "embedding" not in _body(post)


def test_a_text_search_sends_text_and_no_vector(client_and_post) -> None:
    client, post = client_and_post
    client.search("docs", text="memory management", k=5)
    body = _body(post)
    assert body["text"] == "memory management"
    assert "vector" not in body, body


def test_a_vector_search_is_unchanged(client_and_post) -> None:
    client, post = client_and_post
    client.search("docs", [0.1, 0.2], k=5)
    body = _body(post)
    assert body["vector"] == [0.1, 0.2]
    assert "text" not in body, body


def test_both_or_neither_is_refused_before_the_request(client_and_post) -> None:
    client, post = client_and_post
    with pytest.raises(ValueError):
        client.search("docs", [0.1, 0.2], text="memory")
    with pytest.raises(ValueError):
        client.search("docs")
    post.assert_not_called()


@pytest.mark.asyncio
async def test_the_async_half_behaves_the_same() -> None:
    # The two halves of this client have drifted before -- a keyword added to one and not the
    # other -- and the drift is invisible until an async caller hits it.
    from unittest.mock import AsyncMock

    client = Client(host="localhost", port=9999)
    response = Mock()
    response.raise_for_status = Mock()
    response.json = Mock(return_value={"success": True, "data": {"hits": []}})
    client.async_client.post = AsyncMock(return_value=response)
    try:
        await client.asearch("docs", text="memory management", k=5)
        body = _body(client.async_client.post)
        assert body["text"] == "memory management"
        assert "vector" not in body
        with pytest.raises(ValueError):
            await client.asearch("docs", [0.1], text="memory")
    finally:
        await client.aclose()
