"""Full-text search, and what it does when it is not compiled in.

`text-bm25` is not a default feature, so on a plain build these routes answer 501 with
`FEATURE_DISABLED` and a message naming the flag. That is the behaviour worth pinning first:
several defects fixed in this repo were endpoints that returned an empty result where they
should have returned an error, and an absent feature reporting "no matches" would be the same
mistake in a more misleading place.

The tests that need the feature skip themselves when it is absent, rather than failing, so this
file is meaningful on both builds. Run the server with `--features text-bm25` to exercise the
other half:

    cargo run --features text-bm25 -- serve

Rewritten from `examples/full_text_e2e_test.py`, a 580-line script with coloured output that
lived in `examples/` and was run by nothing.
"""

import time
import uuid

import pytest

from .helpers.client import post, succeeded
from .helpers.corpus import create_table

DOCS = [
    ("d1", "vector databases index high dimensional embeddings"),
    ("d2", "full text search ranks documents by term frequency"),
    ("d3", "hybrid search combines vector similarity and keyword matching"),
    ("d4", "quantization compresses vectors into compact codes"),
]


def vector_for(i):
    v = [0.0, 0.0, 0.0, 0.0]
    v[i % 4] = 1.0
    return v


@pytest.fixture(scope="module")
def text_table():
    name = f"ft_{uuid.uuid4().hex[:8]}"
    post(
        "/tables",
        {
            "name": name,
            "dimension": 4,
            "distance_metric": "cosine",
            "enable_full_text_search": True,
            # Both halves are required. `enable_full_text_search` alone builds no index over
            # anything, and a search then returns an empty result rather than an error -- see
            # `test_enabling_the_feature_without_naming_a_field_indexes_nothing`.
            # `name` is the field's name *in the text index*; `source.metadata_field` says
            # which payload key feeds it. They need not match, and the working examples use
            # "content" fed from a payload key of another name -- a first attempt here set
            # both to "text" and found nothing.
            "text_fields": [
                {
                    "name": "content",
                    "source": {"metadata_field": "text"},
                    "analyzer": "standard",
                    "boost": 1.0,
                }
            ],
        },
    )
    records = [
        {"id": rid, "vector": vector_for(i), "payload": {"text": text}}
        for i, (rid, text) in enumerate(DOCS)
    ]
    post(f"/tables/{name}/records", {"records": records})
    # Tantivy commits asynchronously; a search issued immediately after the insert can see an
    # index that is still empty. The working example sleeps here too.
    time.sleep(1.5)
    yield name
    from .helpers.client import delete

    delete(f"/tables/{name}", expect_ok=False)


def text_search(table, body):
    return post(f"/tables/{table}/search/text", body, expect_ok=False)


def feature_is_absent(response):
    if succeeded(response):
        return False
    body = response.json()
    error = body.get("error") or {}
    return error.get("code") == "FEATURE_DISABLED"


def test_an_absent_feature_says_so_rather_than_returning_nothing(text_table):
    """Either it works, or it reports 501 and names the flag. What must not happen is an empty
    result set, which a caller cannot tell from "no documents matched"."""
    response = text_search(text_table, {"query": "vector", "limit": 5})
    if succeeded(response):
        return  # the feature is compiled in; the tests below cover it
    assert response.status_code == 501, f"expected 501, got {response.status_code}"
    error = response.json()["error"]
    assert error["code"] == "FEATURE_DISABLED"
    assert "text-bm25" in error["message"], "the message should name the flag to compile with"


def test_a_term_finds_the_document_that_contains_it(text_table):
    response = text_search(text_table, {"query": "quantization", "limit": 5})
    if feature_is_absent(response):
        pytest.skip("server built without text-bm25")
    assert succeeded(response)
    hits = response.json()["data"]["hits"]
    assert [h["id"] for h in hits] == ["d4"]


def test_a_term_in_several_documents_returns_all_of_them(text_table):
    response = text_search(text_table, {"query": "search", "limit": 5})
    if feature_is_absent(response):
        pytest.skip("server built without text-bm25")
    ids = {h["id"] for h in response.json()["data"]["hits"]}
    assert ids == {"d2", "d3"}


def test_a_term_in_no_document_returns_nothing(text_table):
    response = text_search(text_table, {"query": "zebra", "limit": 5})
    if feature_is_absent(response):
        pytest.skip("server built without text-bm25")
    assert response.json()["data"]["hits"] == []


def test_enabling_the_feature_without_naming_a_field_indexes_nothing(text_table):
    """A table with `enable_full_text_search` but no `text_fields` answers a text search with
    an empty result, not an error.

    Recorded rather than asserted as correct: it is the same shape as several defects fixed in
    this repo, where a caller who has configured something incompletely is told "no matches"
    instead of "nothing is indexed". Pinned so that a decision to make it explicit shows up
    here as a deliberate change rather than a surprise.
    """
    import uuid as _uuid

    from .helpers.client import delete

    name = f"ftbare_{_uuid.uuid4().hex[:8]}"
    post(
        "/tables",
        {
            "name": name,
            "dimension": 4,
            "distance_metric": "cosine",
            "enable_full_text_search": True,
        },
    )
    try:
        post(
            f"/tables/{name}/records",
            {"records": [{"id": "x", "vector": [1.0, 0, 0, 0], "payload": {"text": "quantization"}}]},
        )
        time.sleep(1.5)
        response = text_search(name, {"query": "quantization", "limit": 5})
        if feature_is_absent(response):
            pytest.skip("server built without text-bm25")
        assert succeeded(response)
        assert response.json()["data"]["hits"] == [], (
            "a table that named no text field returned matches; if that is now supported, "
            "this test is the place to say so"
        )
    finally:
        delete(f"/tables/{name}", expect_ok=False)


def test_hybrid_search_accepts_both_a_vector_and_text(text_table):
    response = post(
        f"/tables/{text_table}/search/hybrid",
        {"vector": vector_for(3), "text": "quantization", "limit": 3},
        expect_ok=False,
    )
    if feature_is_absent(response):
        pytest.skip("server built without text-bm25")
    assert succeeded(response), response.text[:400]
