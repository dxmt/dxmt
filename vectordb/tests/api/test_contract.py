"""The HTTP contract itself: paging, error codes, and the envelope every answer arrives in.

The files next door assert what the database *computes*. This one asserts what a client can rely
on regardless of the answer — the shape of the envelope, which status code a failure carries,
which string names it, and what paging does at its edges. Those are the parts an SDK hard-codes
and a release can break without any search returning a wrong row.

Two things found while writing it, both deliberately pinned as they are rather than changed:

* **The two paging routes do not agree on a cap.** `records/list` clamps `limit` to 1,000;
  `GET /tables` has no ceiling at all and will return everything. Asserted here so the
  difference is a decision on record rather than a surprise.
* **There are two independent error-code-to-status maps.** `error_response` goes through
  `Error::status_code` (by enum), while a handler that returns `ApiResponse::error(..)` directly
  goes through `IntoResponse` (by code string). They agree everywhere reachable; they disagree
  only on `Timeout` (408 by string, 500 by enum), and nothing in the HTTP path can construct
  `Error::Timeout`. The matrix below is what both paths must keep answering.
"""

import uuid

import pytest

from .helpers.client import delete, get, post
from .helpers.corpus import CORPUS, DIM, create_table, query_at

#: `GET /tables` when the caller names no page.
DEFAULT_TABLE_LIMIT = 50
#: `POST /tables/{t}/records/list` when the caller names no page, and its ceiling.
DEFAULT_RECORD_LIMIT = 100
MAX_RECORD_LIMIT = 1000


def envelope(response):
    """Every route answers `{"success", "data", "error", "timestamp"}`, success or failure."""
    body = response.json()
    assert set(body) >= {"success", "data", "error"}, body
    return body


def list_page(table, **body):
    return post(f"/tables/{table}/records/list", body).json()["data"]


def tables_page(query=""):
    return get(f"/tables{query}").json()["data"]


# ---- The envelope ----------------------------------------------------------------------------


def test_a_successful_answer_carries_data_and_no_error(table):
    body = envelope(get(f"/tables/{table}"))
    assert body["success"] is True
    assert body["error"] is None
    assert body["data"] is not None


def test_a_failed_answer_carries_an_error_and_no_data():
    body = envelope(get("/tables/definitely_not_a_table", expect_ok=False))
    assert body["success"] is False
    assert body["data"] is None
    assert body["error"]["code"], "an error must name itself, not just describe itself"
    assert body["error"]["message"]


# ---- The error-code matrix -------------------------------------------------------------------
#
# Status code and code string together: a client switching on either one must keep working. Only
# the reachable half of the matrix is here — 401 has no auth to refuse a request, 503 needs
# durability to degrade (tests/wal_recovery_test.rs, feature `test-failpoints`), and 500 means a
# bug rather than a contract.


def test_a_missing_table_is_404_table_not_found():
    response = get("/tables/definitely_not_a_table", expect_ok=False)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TABLE_NOT_FOUND"


def test_a_missing_record_is_404(table):
    response = get(f"/tables/{table}/records/definitely_not_a_record", expect_ok=False)
    assert response.status_code == 404
    assert response.json()["error"]["code"] in {"RECORD_NOT_FOUND", "NOT_FOUND"}


def test_a_duplicate_table_is_409_table_already_exists(table):
    body = {"name": table, "dimension": DIM, "distance_metric": "cosine"}
    response = post("/tables", body, expect_ok=False)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "TABLE_ALREADY_EXISTS"


def test_a_wrong_dimension_is_400_but_skips_the_envelope(fresh_table):
    """Pinned as found. This is the one REST route that answers a failure without the envelope.

    `POST /tables/{t}/records` validates the batch in the handler and returns a bare
    `{"error": "...", "details": ...}` — no `success`, no `data`, and `error` is a string rather
    than the `{code, message}` object every other route returns. A client switching on
    `error.code`, which is what the rest of this matrix teaches it to do, gets a `TypeError`
    here instead of a code. The 429 overload reply on the same route has the same shape.

    Left alone rather than fixed: changing it is a breaking change for anyone reading `error`
    as a string or reading `details`, so it is a decision to take deliberately, not a drive-by.
    """
    body = {"records": [{"id": "wrong", "vector": [1.0, 2.0]}]}
    response = post(f"/tables/{fresh_table}/records", body, expect_ok=False)

    assert response.status_code == 400
    payload = response.json()
    assert isinstance(payload["error"], str), "if this became an object, the contract improved"
    assert "details" in payload
    assert "success" not in payload, "this route does not use the envelope"


def test_an_illegal_table_name_is_400_invalid_table_name():
    body = {"name": "bad name!", "dimension": DIM, "distance_metric": "cosine"}
    response = post("/tables", body, expect_ok=False)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_TABLE_NAME"


def test_hybrid_search_refuses_what_it_cannot_fuse_with_501(table):
    """`/hybrid_search` serves a lone dense vector and fuses nothing else.

    It used to run the dense search, drop the sparse and text components, and return the result
    as though the requested fusion had happened — the caller's alpha/beta/gamma changed nothing.
    Refusing is the contract; the message names the endpoints that do work.
    """
    body = {"dense_vector": query_at("r1"), "text_query": "anything", "top_k": 3}
    response = post(f"/tables/{table}/hybrid_search", body, expect_ok=False)

    assert response.status_code == 501, response.text
    assert response.json()["error"]["code"] == "NOT_IMPLEMENTED"


def test_hybrid_search_still_serves_a_lone_dense_vector(table):
    body = {"dense_vector": query_at("r1"), "top_k": 3}
    hits = post(f"/tables/{table}/hybrid_search", body).json()["data"]["hits"]
    assert [hit["id"] for hit in hits][0] == "r1"


# ---- Paging: POST /tables/{t}/records/list ---------------------------------------------------


def test_records_list_reports_the_page_it_gave_you(table):
    page = list_page(table, limit=3, offset=1)

    assert len(page["records"]) == 3
    assert page["limit"] == 3
    assert page["offset"] == 1
    assert page["total"] == len(CORPUS), "total counts the table, not the page"


def test_records_list_defaults_to_a_hundred(table):
    assert list_page(table)["limit"] == DEFAULT_RECORD_LIMIT


def test_records_list_clamps_an_oversized_limit(table):
    """A caller asking for more than the ceiling is answered at the ceiling, not refused."""
    assert list_page(table, limit=5000)["limit"] == MAX_RECORD_LIMIT


def test_records_list_past_the_end_is_empty_but_still_counts(table):
    page = list_page(table, limit=10, offset=len(CORPUS) + 5)

    assert page["records"] == []
    assert page["total"] == len(CORPUS)


def test_paging_records_covers_every_id_once(table):
    seen = []
    for offset in range(0, len(CORPUS), 3):
        seen += [record["id"] for record in list_page(table, limit=3, offset=offset)["records"]]

    assert sorted(seen) == sorted(record["id"] for record in CORPUS)


# ---- Paging: GET /tables ---------------------------------------------------------------------


def test_tables_listing_reports_the_page_it_gave_you(table):
    page = tables_page("?limit=1&offset=0")

    assert len(page["tables"]) == 1
    assert page["limit"] == 1
    assert page["offset"] == 0
    assert page["total"] >= 1, "total counts every table, not the page"


def test_tables_listing_defaults_to_fifty(table):
    assert tables_page()["limit"] == DEFAULT_TABLE_LIMIT


def test_tables_listing_has_no_ceiling_unlike_records_list(table):
    """Pinned as found, not as preferred: `records/list` clamps to 1,000 and this route does not.

    A client that assumes one ceiling holds everywhere would silently truncate here, so the
    difference is worth a failing test the day someone changes either side.
    """
    assert tables_page("?limit=5000")["limit"] == 5000


def test_tables_listing_past_the_end_is_empty_but_still_counts(table):
    total = tables_page()["total"]
    page = tables_page(f"?offset={total + 10}")

    assert page["tables"] == []
    assert page["total"] == total


def test_a_table_absent_from_the_first_page_is_still_reachable_by_paging():
    """The listing helper other files use takes the default page, so a table beyond it would
    look deleted. Paging has to reach it."""
    name = f"api_page_{uuid.uuid4().hex[:8]}"
    create_table(name, DIM)
    try:
        total = tables_page()["total"]
        seen = []
        for offset in range(0, total, 25):
            seen += [t["name"] for t in tables_page(f"?limit=25&offset={offset}")["tables"]]
        assert name in seen
    finally:
        delete(f"/tables/{name}", expect_ok=False)


# ---- Score direction -------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_score_is_a_similarity_whatever_the_metric(metric):
    """Higher is better on every metric, and the ordering agrees with the scores.

    `test_scores_decrease_monotonically` pins this for the default cosine table; the risk this
    adds is a metric whose distance is never converted, which would rank correctly while
    handing back numbers that shrink as relevance grows.
    """
    name = f"api_score_{uuid.uuid4().hex[:8]}"
    create_table(name, DIM, metric=metric)
    try:
        post(f"/tables/{name}/records", {"records": CORPUS})
        hits = post(f"/tables/{name}/search", {"vector": query_at("r1"), "top_k": 5}).json()[
            "data"
        ]["hits"]
        scores = [hit["score"] for hit in hits]

        assert hits[0]["id"] == "r1"
        assert scores == sorted(scores, reverse=True), scores
    finally:
        delete(f"/tables/{name}", expect_ok=False)
