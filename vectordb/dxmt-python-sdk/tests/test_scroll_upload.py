"""`scroll` follows the server's cursor; `upload` batches, retries what a retry can fix, and says
what did not go in. Against a mock transport, so what is checked is exactly what goes on the wire."""

import json

import httpx
import pytest

from dxmt import DXMTClient


def rec(i):
    return {"id": f"r{i}", "vector": [0.1, 0.2], "payload": {"i": i}}


def page(ids, cursor, cursor_field=True):
    data = {"records": [{"id": i, "payload_json": "{}", "vector": []} for i in ids], "total": 99}
    if cursor_field:
        data["next_cursor"] = cursor
    return httpx.Response(200, json={"success": True, "data": data})


def test_scroll_follows_the_cursor_until_there_is_none():
    bodies = []
    pages = iter([page(["a", "b"], "c1"), page(["c", "d"], "c2"), page(["e"], None)])

    def handler(request):
        bodies.append(json.loads(request.content))
        return next(pages)

    client = DXMTClient(transport=httpx.MockTransport(handler))
    got = [r["id"] for r in client.scroll("t", batch_size=2, filter={"must": [{"key": "k", "operation": {"type": "Equal", "value": 1}}]})]
    assert got == ["a", "b", "c", "d", "e"]
    assert [b.get("cursor") for b in bodies] == [None, "c1", "c2"]
    assert all(b["limit"] == 2 and "filter" in b for b in bodies)


@pytest.mark.asyncio
async def test_ascroll_does_the_same():
    pages = iter([page(["a"], "c1"), page(["b"], None)])

    async def handler(request):
        return next(pages)

    client = DXMTClient(transport=httpx.MockTransport(handler))
    assert [r["id"] async for r in client.ascroll("t", batch_size=1)] == ["a", "b"]


def test_a_full_page_from_a_server_without_cursors_is_an_error_not_the_end():
    # Treating the missing field as "no next page" would end every scroll after one page.
    client = DXMTClient(transport=httpx.MockTransport(lambda r: page(["a", "b"], None, cursor_field=False)))
    with pytest.raises(Exception, match="does not page by cursor"):
        list(client.scroll("t", batch_size=2))


def recording(statuses):
    """A server answering each request with the next status, recording the ids it was sent."""
    sent = []
    queue = list(statuses)

    def handler(request):
        sent.append([r["id"] for r in json.loads(request.content)["records"]])
        status = queue.pop(0) if queue else 200
        if status == "drop":
            raise httpx.ConnectError("connection dropped")
        return httpx.Response(status, json={"success": status == 200, "data": {}, "error": None})

    return sent, handler


def test_upload_sends_batches_of_the_size_asked_for():
    sent, handler = recording([])
    result = DXMTClient(transport=httpx.MockTransport(handler)).upload("t", (rec(i) for i in range(10)), batch_size=4)
    assert [len(b) for b in sent] == [4, 4, 2]
    assert (result.uploaded, result.batches, result.ok) == (10, 3, True)


def test_a_busy_server_and_a_dropped_connection_are_retried():
    sent, handler = recording([503, "drop", 200, 429, 200])
    result = DXMTClient(transport=httpx.MockTransport(handler)).upload("t", [rec(i) for i in range(4)], batch_size=2, backoff=0)
    assert result.ok and result.uploaded == 4
    assert result.retries == 3
    assert len(sent) == 5


def test_a_request_that_is_wrong_is_not_retried_and_is_reported_with_its_ids():
    sent, handler = recording([400, 200])
    result = DXMTClient(transport=httpx.MockTransport(handler)).upload("t", [rec(i) for i in range(4)], batch_size=2, backoff=0)
    assert len(sent) == 2, "a 400 was retried"
    assert result.uploaded == 2 and not result.ok
    assert result.failed[0].ids == ["r0", "r1"] and result.failed[0].status == 400


def test_retries_give_up_after_max_retries():
    sent, handler = recording([503] * 10)
    result = DXMTClient(transport=httpx.MockTransport(handler)).upload("t", [rec(0)], max_retries=2, backoff=0)
    assert len(sent) == 3 and result.failed[0].status == 503


def test_metadata_is_sent_as_payload():
    sent_bodies = []

    def handler(request):
        sent_bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"success": True, "data": {}})

    DXMTClient(transport=httpx.MockTransport(handler)).upload("t", [{"id": "a", "vector": [1.0], "metadata": {"m": 1}}])
    assert sent_bodies[0]["records"][0]["payload"] == {"m": 1}


def test_parallel_uploads_everything_once():
    sent, handler = recording([])
    result = DXMTClient(transport=httpx.MockTransport(handler)).upload("t", (rec(i) for i in range(100)), batch_size=7, parallel=4)
    assert result.uploaded == 100 and result.batches == 15
    flat = sorted(i for b in sent for i in b)
    assert flat == sorted(f"r{i}" for i in range(100)), "a record was sent twice or not at all"
