from __future__ import annotations

import os
from uuid import uuid4

import pytest
from agent_framework import VectorStoreCollectionDefinition, VectorStoreField
from dxmt import DXMTClient
from dxmt.exceptions import DXMTError

SERVER_MARKS = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("DXMT_TEST_URL"), reason="Set DXMT_TEST_URL to a disposable DXMT server."),
]


@pytest.fixture(autouse=True)
def clear_connection_environment(monkeypatch):
    monkeypatch.delenv("DXMT_URL", raising=False)
    monkeypatch.delenv("DXMT_API_KEY", raising=False)


@pytest.fixture
def definition():
    return VectorStoreCollectionDefinition(
        [
            VectorStoreField("key", name="id", type_="int", storage_name="row_key"),
            VectorStoreField("data", name="text", type_="str", storage_name="body", is_full_text_indexed=True),
            VectorStoreField("data", name="number", type_="float", storage_name="price", is_indexed=True),
            VectorStoreField("data", name="integer", type_="int"),
            VectorStoreField("data", name="flag", type_="bool"),
            VectorStoreField("data", name="tags", type_="list"),
            VectorStoreField(
                "vector", name="embedding", storage_name="dense", dimensions=3, distance_function="cosine_similarity"
            ),
        ]
    )


@pytest.fixture
def record():
    def make(
        key: int,
        text: str = "hello world",
        number: float = 1.5,
        integer: int = 1,
        flag: bool = True,
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> dict:
        return {
            "id": key,
            "text": text,
            "number": number,
            "integer": integer,
            "flag": flag,
            "tags": tags if tags is not None else ["a", "b"],
            "embedding": embedding if embedding is not None else [1.0, 0.0, 0.0],
        }

    return make


@pytest.fixture
def server_url() -> str:
    return os.environ.get("DXMT_TEST_URL", "http://127.0.0.1:9999")


@pytest.fixture
def table_name(server_url):
    name = f"af_{uuid4().hex[:10]}"
    yield name
    try:
        from urllib.parse import urlparse

        p = urlparse(server_url)
        DXMTClient(
            host=p.hostname, port=p.port or 9999, ssl=p.scheme == "https", api_key=os.getenv("DXMT_TEST_API_KEY")
        ).delete_table(name)
    except DXMTError:
        pass  # the test already dropped it
