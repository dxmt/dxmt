"""Documents from the client's side.

Chunking, embedding and replacement are the server's and are tested there
(`tests/interfaces/documents_test.rs`). What is here is what the client decides: the path,
with a document id that is a file path encoded as one segment; a body that sends only what the
caller set, so the server's defaults apply; and a missing document read as `None` while a
missing table is still an error.
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


def test_put_sends_only_what_was_set_to_an_encoded_path(client):
    client._client.put = Mock(
        return_value=answer(200, {"success": True, "data": {"doc_id": "a/b.md", "chunks": 3, "removed": 0}})
    )
    written = client.put_document("kb", "notes/a b.md", "全文", metadata={"title": "T"})
    assert written == {"doc_id": "a/b.md", "chunks": 3, "removed": 0}
    path = client._client.put.call_args.args[0]
    assert path.endswith("/tables/kb/documents/notes%2Fa%20b.md"), path
    assert client._client.put.call_args.kwargs["json"] == {"text": "全文", "metadata": {"title": "T"}}

    client.put_document("kb", "d", "x", chunk_size=300, chunk_overlap=30)
    assert client._client.put.call_args.kwargs["json"] == {
        "text": "x", "chunk_size": 300, "chunk_overlap": 30,
    }


def test_a_missing_document_is_none_and_a_missing_table_is_an_error(client):
    client._client.get = Mock(return_value=answer(404, {
        "success": False,
        "error": {"code": "NOT_FOUND", "message": "Record not found: no document 'd' in table 'kb'"},
    }))
    assert client.get_document("kb", "d") is None
    client._client.get = Mock(return_value=answer(404, {
        "success": False, "error": {"code": "NOT_FOUND", "message": "Table not found: nope"},
    }))
    with pytest.raises(DXMTRecordError):
        client.get_document("nope", "d")


def test_delete_returns_the_chunk_count(client):
    client._client.delete = Mock(return_value=answer(200, {"success": True, "data": {"deleted_chunks": 4}}))
    assert client.delete_document("kb", "d") == 4


async def test_the_async_twins_take_the_same_path(client):
    client.async_client.put = AsyncMock(
        return_value=answer(200, {"success": True, "data": {"doc_id": "d", "chunks": 1, "removed": 2}})
    )
    client.async_client.delete = AsyncMock(
        return_value=answer(200, {"success": True, "data": {"deleted_chunks": 1}})
    )
    assert (await client.aput_document("kb", "d", "t"))["removed"] == 2
    assert await client.adelete_document("kb", "d") == 1
    assert client.async_client.put.call_args.args[0].endswith("/tables/kb/documents/d")
