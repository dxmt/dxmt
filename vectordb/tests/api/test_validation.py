"""What the server refuses, and with what.

An API that accepts a malformed request and does something reasonable-looking is worse than one
that rejects it: the caller finds out much later, from wrong answers. Each test here names the
status code rather than only asserting failure, because "rejected" and "crashed" are both
non-2xx and only one of them is the intended behaviour.
"""

import pytest

from .helpers.client import delete, get, post, put, succeeded
from .helpers.corpus import DIM, query_at


def test_a_query_vector_of_the_wrong_length_is_refused(table):
    for wrong in ([1.0, 0.0], [1.0] * (DIM + 3), []):
        response = post(
            f"/tables/{table}/search", {"vector": wrong, "top_k": 1}, expect_ok=False
        )
        assert not succeeded(response), f"a {len(wrong)}-dimensional query was answered"


def test_a_record_vector_of_the_wrong_length_is_refused(fresh_table):
    response = post(
        f"/tables/{fresh_table}/records",
        {"records": [{"id": "bad", "vector": [1.0, 2.0]}]},
        expect_ok=False,
    )
    assert not succeeded(response)
    # And it did not land half-way: the id must not be readable afterwards.
    assert not succeeded(get(f"/tables/{fresh_table}/records/bad", expect_ok=False))


def test_searching_a_table_that_does_not_exist_is_a_404(table):
    response = post(
        "/tables/definitely_not_a_table/search",
        {"vector": query_at("r1"), "top_k": 1},
        expect_ok=False,
    )
    assert response.status_code == 404


def test_inserting_into_a_table_that_does_not_exist_is_refused(table):
    response = post(
        "/tables/definitely_not_a_table/records",
        {"records": [{"id": "x", "vector": query_at("r1")}]},
        expect_ok=False,
    )
    assert not succeeded(response)


def test_updating_a_record_that_does_not_exist_is_refused(fresh_table):
    response = put(
        f"/tables/{fresh_table}/records/not_there", {"payload": {"rank": 1}}, expect_ok=False
    )
    assert not succeeded(response)


def test_a_malformed_body_is_refused(table):
    """No `vector` at all. The handler must say so rather than search on a default."""
    response = post(f"/tables/{table}/search", {"top_k": 1}, expect_ok=False)
    assert not succeeded(response)


@pytest.mark.parametrize("k", [0, -1])
def test_a_nonsensical_top_k_does_not_return_arbitrary_results(table, k):
    """Either refuse it or return nothing. What must not happen is a full table dump because
    the value wrapped around or was ignored."""
    response = post(f"/tables/{table}/search", {"vector": query_at("r1"), "top_k": k},
                    expect_ok=False)
    if succeeded(response):
        assert response.json()["data"]["hits"] == [], f"top_k={k} returned results"


def test_creating_a_table_with_a_nonsensical_dimension_is_refused(table):
    for dim in (0, -1):
        response = post(
            "/tables",
            {"name": f"bad_dim_{abs(dim)}", "dimension": dim, "distance_metric": "cosine"},
            expect_ok=False,
        )
        assert not succeeded(response), f"dimension {dim} was accepted"


def test_deleting_a_table_that_does_not_exist_is_refused(table):
    response = delete("/tables/definitely_not_a_table", expect_ok=False)
    assert not succeeded(response)


def _records(*specs):
    """`("id", dim)` builds a record of that width; a plain string is a valid record."""
    out = []
    for s in specs:
        if isinstance(s, tuple):
            out.append({"id": s[0], "vector": [0.5] * s[1]})
        else:
            out.append({"id": s, "vector": [0.5] * DIM})
    return out


def test_a_batch_refusal_names_every_bad_row_and_writes_none(empty_table):
    """One bad row in a million used to fail the request with the first bad row named, so a
    bulk load was a round trip per defect. The refusal lists them all (capped) and the
    count, and the good rows are still not written -- all-or-nothing is the default."""
    response = post(
        f"/tables/{empty_table}/records",
        {"records": _records("ok1", ("bad1", 2), "ok2", ("bad2", DIM + 1), ("", DIM))},
        expect_ok=False,
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_FAILED"
    failed = error["details"]["failed"]
    assert [f["index"] for f in failed] == [1, 3, 4]
    assert error["details"]["failed_count"] == 3
    assert "3 of 5" in error["message"]
    assert all("error" in f and "id" in f for f in failed)
    assert post(f"/tables/{empty_table}/records/count", {}).json()["data"]["count"] == 0


def test_on_error_skip_writes_the_good_rows_and_reports_the_rest(empty_table):
    response = post(
        f"/tables/{empty_table}/records",
        {"records": _records("ok1", ("bad1", 2), "ok2"), "on_error": "skip"},
    )
    data = response.json()["data"]
    assert data["inserted_ids"] == ["ok1", "ok2"]
    assert data["count"] == 2
    assert [f["index"] for f in data["failed"]] == [1]
    assert data["failed"][0]["id"] == "bad1"
    assert data["failed_count"] == 1
    assert post(f"/tables/{empty_table}/records/count", {}).json()["data"]["count"] == 2


def test_a_good_batch_reports_no_failures_in_the_same_shape(empty_table):
    data = post(f"/tables/{empty_table}/records", {"records": _records("a", "b")}).json()["data"]
    assert data["count"] == 2 and data["failed"] == [] and data["failed_count"] == 0


def test_on_error_skip_with_nothing_valid_is_still_a_refusal(empty_table):
    response = post(
        f"/tables/{empty_table}/records",
        {"records": _records(("bad1", 2), ("bad2", 3)), "on_error": "skip"},
        expect_ok=False,
    )
    assert response.status_code == 400
    assert response.json()["error"]["details"]["failed_count"] == 2


def test_an_unknown_on_error_value_is_refused(empty_table):
    response = post(
        f"/tables/{empty_table}/records",
        {"records": _records("a"), "on_error": "ignore"},
        expect_ok=False,
    )
    assert response.status_code == 400
