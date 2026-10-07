"""The client passes `rerank` through on create and on each search route, and omits it otherwise.

The server does the work; what the client owes is the field in the body, exactly where the
server reads it. These run against a mock transport. `test_integration.py` has the live half.
"""

import json
from unittest.mock import Mock

import pytest

from dxmt import Client

RERANK = {"model": "rerank-v3.5", "base_url": "http://127.0.0.1:1/v2", "format": "cohere"}


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
    kwargs = post_mock.call_args.kwargs
    if "json" in kwargs:
        return kwargs["json"]
    return json.loads(kwargs["content"])


def test_create_table_sends_the_rerank_config(client_and_post) -> None:
    client, post = client_and_post
    client.create_table("docs", dimension=4, rerank=RERANK)
    assert _body(post)["rerank"] == RERANK


def test_create_table_omits_it_when_unset(client_and_post) -> None:
    client, post = client_and_post
    client.create_table("plain", dimension=4)
    assert "rerank" not in _body(post)


def test_search_sends_rerank_beside_text(client_and_post) -> None:
    client, post = client_and_post
    client.search("docs", text="memory management", k=5, rerank={"top_n": 20})
    body = _body(post)
    assert body["rerank"] == {"top_n": 20}
    assert body["text"] == "memory management"


def test_search_omits_rerank_when_unset(client_and_post) -> None:
    client, post = client_and_post
    client.search("docs", vector=[0.1, 0.2, 0.3, 0.4], k=5)
    assert "rerank" not in _body(post)


def test_an_empty_rerank_is_still_sent(client_and_post) -> None:
    # `{}` means "rerank with the table's defaults"; it is not the same as not asking.
    client, post = client_and_post
    client.search("docs", text="memory", k=5, rerank={})
    assert _body(post)["rerank"] == {}


def test_full_text_search_sends_rerank(client_and_post) -> None:
    client, post = client_and_post
    client.full_text_search("docs", query="memory", k=5, rerank={"field": "body"})
    assert _body(post)["rerank"] == {"field": "body"}


def test_hybrid_search_sends_rerank(client_and_post) -> None:
    client, post = client_and_post
    client.hybrid_search("docs", text="memory", k=5, rerank={})
    assert _body(post)["rerank"] == {}


def test_rerank_score_comes_back_on_the_hit(client_and_post) -> None:
    client, post = client_and_post
    post.return_value.json = Mock(
        return_value={
            "success": True,
            "data": {"hits": [{"id": "a", "score": 0.4, "payload": {}, "rerank_score": 0.93}]},
        }
    )
    hits = client.search("docs", text="memory", k=1, rerank={})
    assert hits[0]["rerank_score"] == 0.93
    assert hits[0]["score"] == 0.4
