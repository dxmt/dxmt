"""`api_key` reaches the client the store builds, and a given client is used as it is."""

import pytest
from dxmt import DXMTClient

from langchain_dxmt import DXMTVectorDB

# These check the argument itself, so the keyed-server default from conftest stays out.
pytestmark = pytest.mark.own_api_key


@pytest.fixture(autouse=True)
def no_server(monkeypatch):
    # The store checks that its table exists as it is built; there is no server here, and what
    # is under test is the client it builds, not the round trip.
    monkeypatch.setattr(DXMTClient, "get_table", lambda self, name: {"name": name, "dimension": 4})


def test_the_key_reaches_the_client_the_store_builds():
    store = DXMTVectorDB(api_key="s3cret", create_table_if_not_exists=False)
    assert store._client._headers["api-key"] == "s3cret"


def test_no_key_means_no_header():
    store = DXMTVectorDB(create_table_if_not_exists=False)
    assert "api-key" not in store._client._headers


def test_a_given_client_is_used_as_configured():
    client = DXMTClient(api_key="theirs")
    store = DXMTVectorDB(client=client, api_key="ignored", create_table_if_not_exists=False)
    assert store._client is client
    assert client._headers["api-key"] == "theirs"
