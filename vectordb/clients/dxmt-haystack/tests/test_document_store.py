"""Haystack's own DocumentStore test suite, run against a live server.

``DocumentStoreBaseTests`` is count, write (every duplicate policy), delete and 39 filter
cases -- ``in`` / ``not in``, ``None``, ISO dates, strings, nested AND / OR / NOT -- with
Haystack's in-memory evaluator as the oracle. Passing it is what "a Haystack document store"
means; it is the same bar the stores in deepset's own integrations repository are held to.

Needs a server: ``DXMT_HOST`` / ``DXMT_PORT`` (default localhost:9999).
"""

import os
import uuid

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import DocumentStoreBaseTests

from haystack_integrations.document_stores.dxmt import DXMTDocumentStore

HOST = os.environ.get("DXMT_HOST", "localhost")
PORT = int(os.environ.get("DXMT_PORT", "9999"))
API_KEY = os.environ.get("DXMT_API_KEY")

pytestmark = pytest.mark.integration


class TestDocumentStore(DocumentStoreBaseTests):
    @pytest.fixture
    def document_store(self):
        store = DXMTDocumentStore(
            table_name=f"hs_{uuid.uuid4().hex[:10]}", embedding_dim=768, host=HOST, port=PORT, api_key=API_KEY
        )
        yield store
        try:
            store.client.delete_table(store.table_name)
        except Exception:
            pass

    def test_write_documents(self, document_store):
        """The default policy (NONE) is an upsert: a repeated id replaces the row, nothing raises."""
        docs = [Document(id="d", content="first", embedding=[0.1] * 768)]
        assert document_store.write_documents(docs) == 1
        assert document_store.write_documents([Document(id="d", content="second", embedding=[0.2] * 768)]) == 1
        assert document_store.count_documents() == 1
        assert document_store.filter_documents()[0].content == "second"

    def assert_documents_are_equal(self, received, expected):
        # Order is not part of the contract; identity, content, meta and embedding are.
        def key(d):
            return d.id

        received, expected = sorted(received, key=key), sorted(expected, key=key)
        assert [d.id for d in received] == [d.id for d in expected]
        for r, e in zip(received, expected):
            assert r.content == e.content
            assert r.meta == e.meta
            if e.embedding is None:
                assert r.embedding is None
            else:
                assert r.embedding is not None and len(r.embedding) == len(e.embedding)
                assert all(abs(a - b) < 1e-6 for a, b in zip(r.embedding, e.embedding))


class TestBeyondTheSuite:
    @pytest.fixture
    def store(self):
        store = DXMTDocumentStore(
            table_name=f"hs_{uuid.uuid4().hex[:10]}",
            embedding_dim=4,
            host=HOST,
            port=PORT,
            api_key=API_KEY,
            enable_full_text_search=True,
        )
        yield store
        try:
            store.client.delete_table(store.table_name)
        except Exception:
            pass

    def test_round_trip_through_to_dict(self, store):
        again = DXMTDocumentStore.from_dict(store.to_dict())
        assert again.table_name == store.table_name and again.port == store.port
        assert again.enable_full_text_search is True

    def test_a_document_without_an_embedding_is_kept_but_never_ranked(self, store):
        store.write_documents(
            [
                Document(id="e", content="has a vector", embedding=[1.0, 0, 0, 0]),
                Document(id="n", content="has none"),
            ]
        )
        assert {d.id for d in store.filter_documents()} == {"e", "n"}
        assert [d.id for d in store.embedding_retrieval([1.0, 0, 0, 0], top_k=5)] == ["e"]
        # It comes back as it went in: no stand-in vector leaks out.
        assert next(d for d in store.filter_documents() if d.id == "n").embedding is None

    def test_bm25_and_hybrid(self, store):
        store.write_documents(
            [
                Document(id="a", content="rust borrows memory instead of collecting it", embedding=[1.0, 0, 0, 0]),
                Document(id="b", content="go collects garbage instead of borrowing", embedding=[0, 1.0, 0, 0]),
            ]
        )
        assert [d.id for d in store.bm25_retrieval("garbage", top_k=2)][0] == "b"
        hits = store.hybrid_retrieval("garbage", [1.0, 0, 0, 0], top_k=2)
        assert {d.id for d in hits} == {"a", "b"} and all(d.score is not None for d in hits)

    def test_a_filter_without_a_server_form_is_refused_by_retrieval_and_answered_by_filter_documents(
        self, store
    ):
        store.write_documents(
            [
                Document(id="a", content="x", meta={"n": 1, "d": "2020-01-01T00:00:00"}, embedding=[1.0, 0, 0, 0]),
                Document(id="b", content="y", meta={"n": 2, "d": "2021-01-01T00:00:00"}, embedding=[0, 1.0, 0, 0]),
            ]
        )
        by_date = {"field": "meta.d", "operator": ">", "value": "2020-06-01T00:00:00"}
        assert [d.id for d in store.filter_documents(by_date)] == ["b"]
        from haystack.document_stores.errors import DocumentStoreError

        with pytest.raises(DocumentStoreError, match="no flat form"):
            store.embedding_retrieval([1.0, 0, 0, 0], filters=by_date)

    def test_duplicate_policies(self, store):
        d = Document(id="dup", content="first", embedding=[1.0, 0, 0, 0])
        assert store.write_documents([d]) == 1
        assert store.write_documents([Document(id="dup", content="second", embedding=[1.0, 0, 0, 0])], policy=DuplicatePolicy.SKIP) == 0
        assert store.filter_documents()[0].content == "first"
        assert store.write_documents([Document(id="dup", content="third", embedding=[1.0, 0, 0, 0])], policy=DuplicatePolicy.OVERWRITE) == 1
        assert store.filter_documents()[0].content == "third"
        assert store.count_documents() == 1
