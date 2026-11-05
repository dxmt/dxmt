"""The three retriever components, in a pipeline and through serialisation, against a server."""

import os
import uuid

import pytest
from haystack import Document, Pipeline
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.dxmt import (
    DXMTBM25Retriever,
    DXMTEmbeddingRetriever,
    DXMTHybridRetriever,
)
from haystack_integrations.document_stores.dxmt import DXMTDocumentStore

HOST = os.environ.get("DXMT_HOST", "localhost")
PORT = int(os.environ.get("DXMT_PORT", "9999"))
API_KEY = os.environ.get("DXMT_API_KEY")

pytestmark = pytest.mark.integration


@pytest.fixture
def store():
    store = DXMTDocumentStore(
        table_name=f"hs_{uuid.uuid4().hex[:10]}", embedding_dim=4, host=HOST, port=PORT, api_key=API_KEY,
        enable_full_text_search=True,
    )
    store.write_documents([
        Document(id="rust", content="rust borrows memory instead of collecting it", meta={"lang": "rust", "year": 2015}, embedding=[1.0, 0, 0, 0]),
        Document(id="go", content="go collects garbage instead of borrowing", meta={"lang": "go", "year": 2012}, embedding=[0, 1.0, 0, 0]),
        Document(id="c", content="c frees memory by hand", meta={"lang": "c", "year": 1972}, embedding=[0, 0, 1.0, 0]),
    ])
    yield store
    try:
        store.client.delete_table(store.table_name)
    except Exception:
        pass


def test_embedding_retriever_in_a_pipeline(store):
    p = Pipeline()
    p.add_component("r", DXMTEmbeddingRetriever(store, top_k=2))
    out = p.run({"r": {"query_embedding": [1.0, 0, 0, 0]}})
    assert [d.id for d in out["r"]["documents"]][0] == "rust"
    assert out["r"]["documents"][0].score is not None


def test_bm25_and_hybrid_retrievers(store):
    assert [d.id for d in DXMTBM25Retriever(store, top_k=1).run(query="garbage")["documents"]] == ["go"]
    hits = DXMTHybridRetriever(store, top_k=3).run(query="memory", query_embedding=[0, 1.0, 0, 0])["documents"]
    assert {d.id for d in hits} >= {"go"} and len(hits) <= 3


def test_filter_policy_merge_ands_the_two_filters(store):
    r = DXMTEmbeddingRetriever(
        store, top_k=3,
        filters={"field": "meta.year", "operator": ">", "value": 2000},
        filter_policy=FilterPolicy.MERGE,
    )
    got = r.run(query_embedding=[1.0, 1.0, 1.0, 0], filters={"field": "meta.lang", "operator": "==", "value": "go"})
    assert [d.id for d in got["documents"]] == ["go"]
    # REPLACE: the runtime filter alone, so `c` (1972) is allowed back in.
    r.filter_policy = FilterPolicy.REPLACE
    got = r.run(query_embedding=[0, 0, 1.0, 0], filters={"field": "meta.lang", "operator": "==", "value": "c"})
    assert [d.id for d in got["documents"]] == ["c"]


def test_serialisation_round_trip(store):
    for cls, kw in (
        (DXMTEmbeddingRetriever, {"return_embedding": True}),
        (DXMTBM25Retriever, {}),
        (DXMTHybridRetriever, {"vector_weight": 0.7}),
    ):
        r = cls(store, top_k=7, filters={"field": "meta.lang", "operator": "==", "value": "go"},
                filter_policy=FilterPolicy.MERGE, **kw)
        again = cls.from_dict(r.to_dict())
        assert again.top_k == 7 and again.filter_policy == FilterPolicy.MERGE
        assert again.document_store.table_name == store.table_name
        assert again.document_store.enable_full_text_search is True
