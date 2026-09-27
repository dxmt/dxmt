"""Value counts from the client's side.

The counting is the server's and is tested there. What is here is the part the client decides:
the request shape, the default limit, and that `covered` and `matched` survive the trip. A
client that returned only `hits` would leave every caller unable to tell that their chart omits
part of the corpus, which is the one thing this feature has to be honest about.
"""

import json
from unittest.mock import Mock

import pytest

from dxmt import Client

ANSWER = {
    "success": True,
    "data": {
        "hits": [{"value": "rust", "count": 2}],
        "distinct_values": 1,
        "covered": 2,
        "matched": 3,
    },
}


@pytest.fixture
def client_and_post():
    client = Client(host="localhost", port=9999)
    response = Mock()
    response.raise_for_status = Mock()
    response.json = Mock(return_value=ANSWER)
    client._client.post = Mock(return_value=response)
    yield client, client._client.post
    client.close()


def body(post_mock):
    kwargs = post_mock.call_args.kwargs
    return kwargs["json"] if "json" in kwargs else json.loads(kwargs["content"])


def test_sends_the_key_and_a_default_limit(client_and_post):
    client, post = client_and_post
    client.facets("docs", "topic")
    assert body(post) == {"key": "topic", "limit": 20}
    assert "/tables/docs/facets" in post.call_args.args[0]


def test_passes_a_filter_and_a_limit(client_and_post):
    client, post = client_and_post
    client.facets("docs", "topic", filter={"lang": "en"}, limit=5)
    sent = body(post)
    assert sent["key"] == "topic" and sent["limit"] == 5
    assert sent["filter"], sent


def test_hands_back_covered_and_matched_not_only_the_buckets(client_and_post):
    client, _ = client_and_post
    result = client.facets("docs", "topic")
    assert result["hits"] == [{"value": "rust", "count": 2}]
    assert (result["covered"], result["matched"]) == (2, 3)
