"""The payload round trip, over HTTP, against a running server.

A document put in through `add_texts` has to come back out of `get_by_ids` and out of a search
with its text and its metadata intact. The adapter writes both into the record's `payload`, so
every one of those paths reads the same field, and a rename on either side breaks all three at
once -- which is why they are asserted together here rather than mocked apart.

This was `tests/test_payload_fixes.py`: a `print`-driven script that returned an exit code,
imported a class name (`DXMT`) this package has never exported, and sat in `tests/` where
pytest collected it. It failed at collection, so the payload contract it describes was checked
by nothing.

Needs a server on `localhost:9999` (or `DXMT_HOST`/`DXMT_PORT`); skips when there is none.
"""

import os
import urllib.request
import uuid

import pytest
from langchain_core.embeddings import Embeddings

from langchain_dxmt import DXMTClient, DXMTVectorDB

DXMT_HOST = os.getenv("DXMT_HOST", "localhost")
DXMT_PORT = int(os.getenv("DXMT_PORT", "9999"))
DIMENSION = 128

pytestmark = pytest.mark.integration


class SimpleEmbeddings(Embeddings):
    """Deterministic embeddings: no model, no network, and the same vector every run.

    What is under test is where the text and metadata are stored, not retrieval quality, so
    the only property needed is that a document and its query agree.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(i + 1) / 100] * DIMENSION for i in range(len(texts))]

    def embed_query(self, text: str) -> list[float]:
        return [0.02] * DIMENSION


@pytest.fixture(scope="module")
def client() -> DXMTClient:
    # Probed over plain HTTP rather than through a client method, so that "there is no server"
    # cannot be confused with "this SDK version spells the health call differently" -- a skip
    # that swallows an API mismatch is how a broken adapter reports as green.
    probe = f"http://{DXMT_HOST}:{DXMT_PORT}/ready"
    try:
        with urllib.request.urlopen(probe, timeout=5) as response:
            if response.status != 200:
                pytest.skip(f"DXMT server at {probe} is not ready (HTTP {response.status})")
    except OSError as exc:
        pytest.skip(f"no DXMT server at {DXMT_HOST}:{DXMT_PORT} ({exc})")
    return DXMTClient(host=DXMT_HOST, port=DXMT_PORT)


@pytest.fixture
def store(client: DXMTClient):
    table = f"langchain_payload_{uuid.uuid4().hex[:8]}"
    vs = DXMTVectorDB(
        table_name=table,
        embedding=SimpleEmbeddings(),
        dimension=DIMENSION,
        client=client,
    )
    try:
        yield vs
    finally:
        try:
            client.delete_table(table)
        except Exception:  # noqa: BLE001 -- cleanup must not mask a test failure
            pass


TEXTS = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks",
    "Natural language processing analyzes text",
]
METADATAS = [
    {"author": "Alice", "category": "ML"},
    {"author": "Bob", "category": "DL"},
    {"author": "Charlie", "category": "NLP"},
]


def test_get_by_ids_returns_content_and_metadata(store) -> None:
    ids = store.add_texts(texts=TEXTS, metadatas=METADATAS)
    assert len(ids) == len(TEXTS)

    docs = {doc.id: doc for doc in store.get_by_ids(ids)}
    assert set(docs) == set(ids)

    for id_, text, metadata in zip(ids, TEXTS, METADATAS):
        doc = docs[id_]
        assert doc.page_content == text
        # The whole point of the payload field: metadata survives the round trip. An empty
        # dict here is the symptom the original script was written to catch.
        assert doc.metadata["author"] == metadata["author"]
        assert doc.metadata["category"] == metadata["category"]


def test_similarity_search_results_carry_the_payload(store) -> None:
    store.add_texts(texts=TEXTS, metadatas=METADATAS)

    results = store.similarity_search("neural networks", k=2)
    assert len(results) == 2
    for doc in results:
        assert doc.page_content in TEXTS
        assert doc.metadata.get("author") in {m["author"] for m in METADATAS}


def test_similarity_search_with_score_carries_the_payload(store) -> None:
    store.add_texts(texts=TEXTS, metadatas=METADATAS)

    results = store.similarity_search_with_score("neural networks", k=2)
    assert len(results) == 2
    for doc, score in results:
        assert doc.page_content in TEXTS
        assert doc.metadata.get("category") in {m["category"] for m in METADATAS}
        assert isinstance(score, float)
