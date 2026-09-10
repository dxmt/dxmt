"""What a search returns, for a table whose right answer is written down.

Order is asserted, not just membership. A test that checks `len(hits) == 3` passes when the
wrong three come back, and "the right number of wrong records" is the failure this suite exists
to catch.
"""

import pytest

from .helpers.client import post, succeeded
from .helpers.corpus import BY_ID, CORPUS, ORDER_FROM_R1, ids_of, query_at


def search(table, body):
    return post(f"/tables/{table}/search", body).json()["data"]


def test_a_record_is_its_own_nearest_neighbour(table):
    for record in CORPUS:
        hits = search(table, {"vector": list(record["vector"]), "top_k": 1})["hits"]
        assert ids_of(hits) == [record["id"]]


def test_neighbours_come_back_in_distance_order(table):
    hits = search(table, {"vector": query_at("r1"), "top_k": 5})["hits"]
    assert ids_of(hits) == ORDER_FROM_R1


def test_scores_decrease_monotonically(table):
    hits = search(table, {"vector": query_at("r1"), "top_k": 8})["hits"]
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True), scores
    assert scores[0] == pytest.approx(1.0, abs=1e-4), "a record against itself is a perfect match"


def test_top_k_bounds_the_answer(table):
    for k in (1, 3, 8):
        hits = search(table, {"vector": query_at("r1"), "top_k": k})["hits"]
        assert len(hits) == k
    # More than the table holds returns the table, not an error.
    hits = search(table, {"vector": query_at("r1"), "top_k": 50})["hits"]
    assert len(hits) == len(CORPUS)


def test_payload_comes_back_and_matches_what_went_in(table):
    hits = search(table, {"vector": query_at("r3"), "top_k": 1})["hits"]
    assert hits[0]["payload"] == BY_ID["r3"]["payload"]


def test_the_vector_is_withheld_unless_asked_for(table):
    plain = search(table, {"vector": query_at("r1"), "top_k": 1})["hits"][0]
    assert plain["vector"] is None, "a search should not pay to return vectors nobody asked for"

    withv = search(table, {"vector": query_at("r1"), "top_k": 1, "with_vector": True})["hits"][0]
    assert withv["vector"] is not None
    assert len(withv["vector"]) == len(BY_ID["r1"]["vector"])


def test_an_empty_table_returns_no_hits_rather_than_an_error(empty_table):
    data = search(empty_table, {"vector": query_at("r1"), "top_k": 5})
    assert data["hits"] == []
    assert data["total"] == 0


def test_the_same_query_twice_returns_the_same_order(table):
    """Ties used to be broken by hash iteration order, so the same query could answer
    differently on two runs -- and because the limit is applied after ordering, it chose which
    tied records came back at all (`5c89817`)."""
    first = ids_of(search(table, {"vector": query_at("r1"), "top_k": 8})["hits"])
    for _ in range(5):
        assert ids_of(search(table, {"vector": query_at("r1"), "top_k": 8})["hits"]) == first


def test_a_wrong_dimension_is_refused(table):
    response = post(
        f"/tables/{table}/search",
        {"vector": [1.0, 0.0], "top_k": 1},
        expect_ok=False,
    )
    assert not succeeded(response), (
        "a 2-dimensional query against a 4-dimensional table must not answer; "
        f"got {response.status_code}"
    )
