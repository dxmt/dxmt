"""Memory from the client's side.

What memory does -- the table made on first use, recall by meaning and keyword, superseding --
is the server's, tested there (`tests/interfaces/mcp_stdio_test.rs`). What is here is what the
client decides: the paths, bodies that carry only what the caller set so the server's defaults
apply, an id encoded as one path segment, and errors raised as the client's own.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from dxmt import Client
from dxmt.exceptions import DXMTRecordError


def answer(status, payload):
    response = Mock()
    response.status_code = status
    response.text = str(payload)
    response.json = Mock(return_value=payload)
    if status >= 400:
        import httpx

        error = httpx.HTTPStatusError("boom", request=Mock(), response=response)
        response.raise_for_status = Mock(side_effect=error)
    else:
        response.raise_for_status = Mock()
    return response


@pytest.fixture
def client():
    c = Client(host="localhost", port=9999)
    yield c
    c.close()


def test_remember_sends_only_what_was_set(client):
    client._client.request = Mock(
        return_value=answer(200, {"success": True, "data": {"id": "m1", "superseded": []}})
    )
    kept = client.remember("Prefers answers in Chinese", kind="preference")
    assert kept == {"id": "m1", "superseded": []}
    method, path = client._client.request.call_args.args
    assert (method, path) == ("POST", "/memories")
    assert client._client.request.call_args.kwargs["json"] == {
        "text": "Prefers answers in Chinese",
        "kind": "preference",
    }

    client.remember("Lives in Berlin", supersedes=["m0"], memory_id="home", tags=["profile"])
    assert client._client.request.call_args.kwargs["json"] == {
        "text": "Lives in Berlin", "tags": ["profile"], "supersedes": ["m0"], "id": "home",
    }


def test_recall_returns_the_memories(client):
    client._client.request = Mock(return_value=answer(200, {
        "success": True,
        "data": {"memories": [{"id": "m1", "text": "Prefers Chinese", "score": 0.9}]},
    }))
    memories = client.recall("language", limit=3, include_superseded=True)
    assert memories[0]["id"] == "m1"
    method, path = client._client.request.call_args.args
    assert (method, path) == ("POST", "/memories/recall")
    assert client._client.request.call_args.kwargs["json"] == {
        "query": "language", "limit": 3, "include_superseded": True,
    }


def test_forget_encodes_the_id_and_list_passes_query_parameters(client):
    client._client.request = Mock(
        return_value=answer(200, {"success": True, "data": {"forgotten": 1}})
    )
    assert client.forget("a/b c") == 1
    method, path = client._client.request.call_args.args
    assert (method, path) == ("DELETE", "/memories/a%2Fb%20c")

    client._client.request = Mock(
        return_value=answer(200, {"success": True, "data": {"memories": [], "total": 0}})
    )
    assert client.list_memories(kind="fact", limit=5) == {"memories": [], "total": 0}
    assert client._client.request.call_args.kwargs["params"] == {
        "limit": 5, "offset": 0, "kind": "fact",
    }


def test_a_refusal_is_the_client_s_own_error(client):
    client._client.request = Mock(return_value=answer(400, {
        "success": False,
        "error": {"code": "INVALID_INPUT", "message": "a memory needs `text`"},
    }))
    with pytest.raises(DXMTRecordError, match="remember"):
        client.remember("")


@pytest.mark.asyncio
async def test_the_async_versions_call_the_same_routes(client):
    client._async_client = Mock()
    client._async_client.request = AsyncMock(
        return_value=answer(200, {"success": True, "data": {"memories": [], "total": 0}})
    )
    assert await client.arecall("x") == []
    assert client._async_client.request.call_args.args == ("POST", "/memories/recall")
