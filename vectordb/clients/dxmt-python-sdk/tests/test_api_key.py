"""The API key reaches the server on every request, sync and async, and only when given.

The server's `DXMT_API_KEY` is off by default. When it is on, every route but the health probes
answers 401 without the key, so a client that cannot send one cannot be used at all — which is
where this SDK was until it grew the argument.
"""

import warnings

import httpx
import pytest

from dxmt import DXMTClient

OK = {"success": True, "data": [], "error": None}


def recording():
    """A transport that answers every request and remembers what was sent."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=OK)

    return seen, handler


def test_the_key_is_sent_as_the_api_key_header():
    seen, handler = recording()
    client = DXMTClient(api_key="s3cret", transport=httpx.MockTransport(handler))
    client.list_tables()
    assert seen[0].headers["api-key"] == "s3cret"


@pytest.mark.asyncio
async def test_the_async_client_sends_it_too():
    seen, handler = recording()

    async def ahandler(request: httpx.Request) -> httpx.Response:
        return handler(request)

    client = DXMTClient(api_key="s3cret", transport=httpx.MockTransport(ahandler))
    await client.alist_tables()
    assert seen[0].headers["api-key"] == "s3cret"


def test_no_key_sends_no_header():
    seen, handler = recording()
    DXMTClient(transport=httpx.MockTransport(handler)).list_tables()
    assert "api-key" not in seen[0].headers


def test_the_argument_wins_over_a_header_of_the_same_name_and_the_callers_dict_is_untouched():
    seen, handler = recording()
    theirs = {"api-key": "stale", "x-trace": "1"}
    DXMTClient(api_key="fresh", headers=theirs, transport=httpx.MockTransport(handler)).list_tables()
    assert seen[0].headers["api-key"] == "fresh"
    assert seen[0].headers["x-trace"] == "1"
    assert theirs == {"api-key": "stale", "x-trace": "1"}


def test_plain_http_to_a_remote_host_warns_and_local_or_tls_does_not():
    with pytest.warns(UserWarning, match="plain HTTP"):
        DXMTClient(host="db.example.com", api_key="k")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DXMTClient(host="localhost", api_key="k")
        DXMTClient(host="127.0.0.1", api_key="k")
        DXMTClient(host="db.example.com", ssl=True, api_key="k")
        DXMTClient(host="db.example.com")  # no key, nothing to protect
