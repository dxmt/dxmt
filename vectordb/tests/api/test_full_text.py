"""Full-text search, and what it does when it is not compiled in.

`text-bm25` is a default feature since 2026-10-02; a build without it (CI's `minimal-core`
job compiles one) answers these routes with 501 `FEATURE_DISABLED` and a message naming the flag. That is the behaviour worth pinning first:
several defects fixed in this repo were endpoints that returned an empty result where they
should have returned an error, and an absent feature reporting "no matches" would be the same
mistake in a more misleading place.

The tests that need the feature skip themselves when it is absent, rather than failing, so this
file is meaningful on both builds. Run the server with `--features text-bm25` to exercise the
other half:

    DXMT_FEATURES=text-bm25 scripts/api_tests.sh

`DXMT_FEATURES` also makes the skip a failure. A skipped test and an absent one are the same
thing, and until CI gained a job that sets it, these four tests had only ever skipped there.

Rewritten from `examples/full_text_e2e_test.py`, a 580-line script with coloured output that
lived in `examples/` and was run by nothing.
"""

import os
import re
import time
import uuid

import pytest

from .helpers.client import delete, post, succeeded
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
    delete(f"/tables/{name}", expect_ok=False)


def text_search(table, body):
    return post(f"/tables/{table}/search/text", body, expect_ok=False)


def feature_is_absent(response):
    if succeeded(response):
        return False
    body = response.json()
    error = body.get("error") or {}
    return error.get("code") == "FEATURE_DISABLED"


def absent(feature):
    """Skip a test whose feature the server lacks -- unless the run declared the server has it,
    in which case the absence is the failure."""
    declared = {f for f in re.split(r"[,\s]+", os.environ.get("DXMT_FEATURES", "")) if f}
    if feature in declared:
        pytest.fail(f"DXMT_FEATURES says the server has {feature}, but it answered FEATURE_DISABLED")
    pytest.skip(f"server built without {feature}")


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


def test_hybrid_refuses_a_text_half_it_cannot_answer(text_table):
    """The same rule as above, on the endpoint where breaking it is hardest to notice.

    `/search/text` with no feature has nothing to return but an error. `/search/hybrid` has the
    dense half, and it used to return exactly that -- 200, a ranked page, `tookms`, everything a
    fusion produces -- with the text component silently dropped in the storage layer, where
    `full_text_engine` is `None` when the feature is off. A caller tuning `fusion_method` and
    `vector_weight` against that page was tuning nothing, and had no way to find out.

    That is the defect `/hybrid_search` was fixed for, and the two endpoints were answering it
    opposite ways until 2026-09-28. It also defeated `test_request_fields.py`, whose
    `needs_text()` decides to skip by looking for `FEATURE_DISABLED`: a 200 meant no skip, so
    its two hybrid probes ran on a dense-only answer and failed -- which is how this was found,
    in CI's dashboard job, whose server is built with `api-dashboard` and no `text-bm25`.

    A body carrying only a vector still has to work on both builds: nothing is being dropped
    there, so there is nothing to refuse.
    """
    vector_only = post(f"/tables/{text_table}/search/hybrid",
                       {"vector": vector_for(0), "limit": 3}, expect_ok=False)
    assert succeeded(vector_only), (
        "a hybrid body with no text component must be answered whatever the build: "
        f"{vector_only.status_code} {vector_only.text[:200]}"
    )

    with_text = post(f"/tables/{text_table}/search/hybrid",
                     {"vector": vector_for(0), "text": "vector", "limit": 3}, expect_ok=False)
    if succeeded(with_text):
        return  # the feature is compiled in; the fusion tests below cover it
    assert with_text.status_code == 501, f"expected 501, got {with_text.status_code}"
    error = with_text.json()["error"]
    assert error["code"] == "FEATURE_DISABLED"
    assert "text-bm25" in error["message"], "the message should name the flag to compile with"


def test_a_term_finds_the_document_that_contains_it(text_table):
    response = text_search(text_table, {"query": "quantization", "limit": 5})
    if feature_is_absent(response):
        absent("text-bm25")
    assert succeeded(response)
    hits = response.json()["data"]["hits"]
    assert [h["id"] for h in hits] == ["d4"]


def test_a_term_in_several_documents_returns_all_of_them(text_table):
    response = text_search(text_table, {"query": "search", "limit": 5})
    if feature_is_absent(response):
        absent("text-bm25")
    ids = {h["id"] for h in response.json()["data"]["hits"]}
    assert ids == {"d2", "d3"}


def test_a_term_in_no_document_returns_nothing(text_table):
    response = text_search(text_table, {"query": "zebra", "limit": 5})
    if feature_is_absent(response):
        absent("text-bm25")
    assert response.json()["data"]["hits"] == []


def test_enabling_the_feature_without_naming_a_field_indexes_nothing(text_table):
    """A table with `enable_full_text_search` but no `text_fields` answers a text search with
    an empty result, not an error.

    Recorded rather than asserted as correct: it is the same shape as several defects fixed in
    this repo, where a caller who has configured something incompletely is told "no matches"
    instead of "nothing is indexed". Pinned so that a decision to make it explicit shows up
    here as a deliberate change rather than a surprise.
    """
    name = f"ftbare_{uuid.uuid4().hex[:8]}"
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
            absent("text-bm25")
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
        absent("text-bm25")
    assert succeeded(response), response.text[:400]


def test_the_two_endpoints_spell_their_timing_field_differently(text_table):
    """`/search/text` answers `took_ms`; `/search` and `/search/hybrid` answer `tookms`.

    Two Rust structs — `FullTextSearchResult` and `SearchResult` — written at different times
    and never reconciled. This is pinned rather than fixed because renaming either breaks
    whichever clients read it, and a client that guesses wrong gets `undefined` rather than an
    error: the dashboard's BM25 column rendered `NaN ms` from exactly this. If the two are ever
    unified, this test is the place that says what it costs.
    """
    text = text_search(text_table, {"query": "quantization", "limit": 3})
    if feature_is_absent(text):
        absent("text-bm25")
    assert succeeded(text), text.text[:400]
    body = text.json()["data"]
    assert "took_ms" in body, f"/search/text lost took_ms: {sorted(body)}"
    assert "tookms" not in body

    dense = post(
        f"/tables/{text_table}/search",
        {"vector": vector_for(3), "top_k": 3, "limit": 3},
    ).json()["data"]
    assert "tookms" in dense, f"/search lost tookms: {sorted(dense)}"
    assert "took_ms" not in dense

    hybrid = post(
        f"/tables/{text_table}/search/hybrid",
        {"vector": vector_for(3), "text": "quantization", "limit": 3},
    ).json()["data"]
    assert "tookms" in hybrid, f"/search/hybrid lost tookms: {sorted(hybrid)}"


def test_hybrid_reads_vector_weight_and_refuses_a_misspelled_knob(text_table):
    """`HybridSearchRequest` has no `alpha` and no `top_k` -- the names this crate's own Swagger
    example advertised. They used to be accepted and ignored, so someone tuning `alpha` watched
    a number that changed nothing. Pinned two ways: the real names work, the invented ones are
    a 400 that names them."""
    query = {"vector": vector_for(3), "text": "quantization", "limit": 3}

    real = post(
        f"/tables/{text_table}/search/hybrid",
        {**query, "fusion_method": "weighted_sum", "vector_weight": 0.9},
        expect_ok=False,
    )
    if feature_is_absent(real):
        absent("text-bm25")
    assert succeeded(real), real.text[:400]

    for knob in ("alpha", "top_k"):
        invented = post(
            f"/tables/{text_table}/search/hybrid", {**query, knob: 0.1}, expect_ok=False
        )
        assert invented.status_code == 400, invented.text[:400]
        error = invented.json()["error"]
        assert error["code"] == "INVALID_INPUT"
        assert knob in error["message"], error


def test_a_text_hit_carries_what_the_console_draws(text_table):
    """id, score and payload on every hit, and `highlights` present as a key even when the
    request did not ask for highlighting — a console that reads `hit.highlights.length` without
    narrowing would white-screen on a missing key rather than an explicit null."""
    response = text_search(text_table, {"query": "quantization", "limit": 3, "highlight": False})
    if feature_is_absent(response):
        absent("text-bm25")
    hits = response.json()["data"]["hits"]
    assert hits, "expected at least one hit for a term that is in the corpus"
    for hit in hits:
        assert set(hit) >= {"id", "score", "payload", "highlights"}, sorted(hit)
        assert isinstance(hit["score"], (int, float))


def test_rrf_is_weighted_here_despite_what_the_name_suggests(text_table):
    """`reciprocal_rank_fusion` reads `vector_weight`; only `dbsf` ignores it.

    Textbook RRF is unweighted — that is the property it is chosen for — but
    `fuse_reciprocal_rank` takes `text_weight = 1 - vector_weight` and scales each side's
    `1/(k + rank)` by it. The console's fusion controls say so, and this is what makes that
    sentence checkable: if RRF ever becomes unweighted, the weight slider silently starts lying
    rather than breaking.

    The assertion is on the **scores**, not on the ranking. On a corpus this small every
    document is in the dense list, so a document that also matches the text stays on top at any
    weight while the scores underneath it move by a factor of twenty. Asserting on the top hit
    would have made this test a property of the corpus.
    """
    # The vector points at d4 ("quantization compresses vectors"); the text asks about d2
    # ("full text search ranks documents"). The two sides disagree on purpose.
    query = {"vector": vector_for(3), "text": "term frequency ranks documents", "limit": 4}

    def fuse(method, weight):
        response = post(
            f"/tables/{text_table}/search/hybrid",
            {**query, "fusion_method": method, "vector_weight": weight},
            expect_ok=False,
        )
        if feature_is_absent(response):
            absent("text-bm25")
        assert succeeded(response), response.text[:400]
        return {hit["id"]: hit["score"] for hit in response.json()["data"]["hits"]}

    dense_heavy = fuse("reciprocal_rank_fusion", 0.95)
    text_heavy = fuse("reciprocal_rank_fusion", 0.05)
    assert dense_heavy.keys() == text_heavy.keys()
    moved = [k for k in dense_heavy if abs(dense_heavy[k] - text_heavy[k]) > 1e-6]
    assert moved, (
        "every RRF score was identical at vector_weight 0.95 and 0.05, so it is ignoring the "
        "weight; the console's fusion controls claim it does not"
    )
    # d4 is the vector's own document and matches no text, so it is the clearest case: its
    # score is the dense side's contribution alone, scaled by the weight.
    assert dense_heavy["d4"] > text_heavy["d4"] * 5, (
        f"d4 scored {dense_heavy['d4']} dense-heavy and {text_heavy['d4']} text-heavy; the "
        "weight should scale a dense-only document's contribution directly"
    )

    # And dbsf, which genuinely ignores it, does not move at all.
    assert fuse("dbsf", 0.95) == fuse("dbsf", 0.05), (
        "dbsf normalises each list by its own distribution and should not read vector_weight"
    )
