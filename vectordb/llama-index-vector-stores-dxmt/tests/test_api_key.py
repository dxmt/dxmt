"""`api_key` reaches the client and nowhere else.

Every other constructor argument of `DXMTVectorStore` is a pydantic field, and pydantic fields come
back out of `repr()`, `to_dict()` and a serialised index — so the key must not become one.
"""

import json

import pytest
from dxmt import DXMTClient

from llama_index.vector_stores.dxmt import DXMTVectorStore

# These check the argument itself, so the keyed-server default from conftest stays out.
pytestmark = pytest.mark.own_api_key


def test_the_key_reaches_the_client():
    store = DXMTVectorStore(api_key="s3cret", dimension=4)
    assert store.client._headers["api-key"] == "s3cret"


def test_the_key_is_not_a_field():
    store = DXMTVectorStore(api_key="s3cret", dimension=4)
    assert "s3cret" not in repr(store)
    assert "s3cret" not in json.dumps(store.to_dict(), default=str)
    assert "api_key" not in store.to_dict()


def test_a_given_client_is_used_as_configured():
    client = DXMTClient(api_key="theirs")
    store = DXMTVectorStore(client=client, api_key="ignored", dimension=4)
    assert store.client is client
    assert client._headers["api-key"] == "theirs"
