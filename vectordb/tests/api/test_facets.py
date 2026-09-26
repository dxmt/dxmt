"""Value counts over HTTP.

`tests/facet_test.rs` owns the counting itself. What is left here is the wire: the request
shape, the two honesty numbers actually arriving in the body, and the refusals being refusals.
"""

import uuid

import pytest

from .helpers.client import delete, post


@pytest.fixture
def shelf():
    name = f"facet_{uuid.uuid4().hex[:8]}"
    post("/tables", {"name": name, "dimension": 4})
    post(f"/tables/{name}/records", {"records": [
        {"id": "a", "vector": [1, 0, 0, 0], "payload": {"topic": "rust", "year": 2024}},
        {"id": "b", "vector": [1, 0, 0, 0], "payload": {"topic": "rust", "year": 2024}},
        {"id": "c", "vector": [1, 0, 0, 0], "payload": {"topic": "go", "year": 2025}},
        {"id": "d", "vector": [1, 0, 0, 0], "payload": {"other": 1}},
    ]})
    yield name
    delete(f"/tables/{name}", expect_ok=False)


def test_counts_come_back_most_common_first(shelf):
    data = post(f"/tables/{shelf}/facets", {"key": "topic"}).json()["data"]
    assert [(h["value"], h["count"]) for h in data["hits"]] == [("rust", 2), ("go", 1)]
    assert data["distinct_values"] == 2


def test_the_answer_says_what_it_cannot_describe(shelf):
    # Record `d` has no `topic` and so is in no bucket. Without these two numbers a chart drawn
    # from `hits` shows a corpus of three where there are four.
    data = post(f"/tables/{shelf}/facets", {"key": "topic"}).json()["data"]
    assert data["matched"] == 4, data
    assert data["covered"] == 3, data


def test_an_integer_field_comes_back_as_integers(shelf):
    # Every number lives in the index as an f64, so this is the one place a year could arrive
    # as 2024.0 -- the same number in JSON, and not the same thing in a chart legend.
    data = post(f"/tables/{shelf}/facets", {"key": "year"}).json()["data"]
    assert [(h["value"], h["count"]) for h in data["hits"]] == [(2024, 2), (2025, 1)]
    assert all(isinstance(h["value"], int) for h in data["hits"]), data["hits"]


def test_a_filter_narrows_the_counts_and_the_denominator(shelf):
    body = {"key": "topic",
            "filter": {"must": [{"key": "year", "operation": {"type": "Equal", "value": 2024}}]}}
    data = post(f"/tables/{shelf}/facets", body).json()["data"]
    assert [(h["value"], h["count"]) for h in data["hits"]] == [("rust", 2)]
    assert (data["covered"], data["matched"]) == (2, 2)


def test_the_limit_is_a_page_and_distinct_values_is_the_whole(shelf):
    data = post(f"/tables/{shelf}/facets", {"key": "topic", "limit": 1}).json()["data"]
    assert len(data["hits"]) == 1
    assert data["distinct_values"] == 2


def test_a_field_nothing_wrote_is_empty_and_a_field_not_indexed_is_a_400(shelf):
    # The distinction is the point: "nobody wrote this" is an answer, "this server does not
    # index it" is not, and an empty page is indistinguishable between them.
    empty = post(f"/tables/{shelf}/facets", {"key": "nobody_wrote_this"}).json()["data"]
    assert empty["hits"] == [] and empty["matched"] == 4

    name = f"facet_excl_{uuid.uuid4().hex[:8]}"
    post("/tables", {"name": name, "dimension": 4, "payload_index": {"except": ["topic"]}})
    try:
        post(f"/tables/{name}/records", {"records": [
            {"id": "a", "vector": [1, 0, 0, 0], "payload": {"topic": "rust"}}]})
        r = post(f"/tables/{name}/facets", {"key": "topic"}, expect_ok=False)
        assert r.status_code == 400, (r.status_code, r.text[:300])
        assert "topic" in r.text, r.text[:300]
    finally:
        delete(f"/tables/{name}", expect_ok=False)


def test_a_missing_key_is_refused(shelf):
    r = post(f"/tables/{shelf}/facets", {"limit": 5}, expect_ok=False)
    assert r.status_code in (400, 422), (r.status_code, r.text[:200])
