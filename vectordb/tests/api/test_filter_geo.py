"""Filtering by place, over HTTP.

The in-process suite (`tests/geo_filter_test.rs`) owns the property that matters -- that the
index and `Record::matches_filter` agree on a randomised corpus. What is left for this one is
the half that only exists on the wire: that the request shape is what the documentation says,
that the refusals are refusals and not empty pages, and that a geo filter works on the
endpoints other than `search`.
"""

import uuid

import pytest

from .helpers.client import delete, get, post

BERLIN = {"lat": 52.5163, "lon": 13.3777}


def radius(center, metres):
    return {"must": [{"key": "shop",
                      "operation": {"type": "GeoRadius",
                                    "value": {"center": center, "radius_m": metres}}}]}


def box(north, west, south, east):
    return {"must": [{"key": "shop",
                      "operation": {"type": "GeoBox",
                                    "value": {"top_left": {"lat": north, "lon": west},
                                              "bottom_right": {"lat": south, "lon": east}}}}]}


@pytest.fixture
def places():
    name = f"geo_{uuid.uuid4().hex[:8]}"
    post("/tables", {"name": name, "dimension": 4})
    post(f"/tables/{name}/records", {"records": [
        {"id": "gate", "vector": [1, 0, 0, 0],
         "payload": {"shop": {"lat": 52.5163, "lon": 13.3777}, "kind": "cafe"}},
        {"id": "reichstag", "vector": [1, 0, 0, 0],
         "payload": {"shop": {"lat": 52.5186, "lon": 13.3761}, "kind": "bank"}},
        {"id": "munich", "vector": [1, 0, 0, 0],
         "payload": {"shop": {"lat": 48.1351, "lon": 11.5820}, "kind": "cafe"}},
    ]})
    yield name
    delete(f"/tables/{name}", expect_ok=False)


def ids(response):
    return sorted(hit["id"] for hit in response.json()["data"]["hits"])


def test_a_radius_search_returns_only_what_is_inside_it(places):
    near = post(f"/tables/{places}/search",
                {"vector": [1, 0, 0, 0], "top_k": 10, "filter": radius(BERLIN, 2000)})
    assert ids(near) == ["gate", "reichstag"], near.json()

    far = post(f"/tables/{places}/search",
               {"vector": [1, 0, 0, 0], "top_k": 10, "filter": radius(BERLIN, 1_000_000)})
    assert ids(far) == ["gate", "munich", "reichstag"], far.json()


def test_a_box_search_and_a_combined_filter(places):
    inside = post(f"/tables/{places}/search",
                  {"vector": [1, 0, 0, 0], "top_k": 10, "filter": box(53.0, 13.0, 52.0, 14.0)})
    assert ids(inside) == ["gate", "reichstag"], inside.json()

    combined = radius(BERLIN, 2000)
    combined["must"].append({"key": "kind", "operation": {"type": "Equal", "value": "cafe"}})
    r = post(f"/tables/{places}/search",
             {"vector": [1, 0, 0, 0], "top_k": 10, "filter": combined})
    assert ids(r) == ["gate"], r.json()


def test_records_list_and_count_take_a_geo_filter_too(places):
    # A filter that works on `search` and not on `records/list` is the shape this repository
    # has shipped before: they go through different code and only one of them was checked.
    listed = post(f"/tables/{places}/records/list", {"filter": radius(BERLIN, 2000), "limit": 10})
    assert sorted(r["id"] for r in listed.json()["data"]["records"]) == ["gate", "reichstag"]

    counted = post(f"/tables/{places}/records/count", {"filter": radius(BERLIN, 2000)})
    assert counted.json()["data"]["count"] == 2, counted.json()


def test_geometry_that_is_not_a_place_is_a_400(places):
    # An empty page is a plausible answer to a real question, which makes it the worst possible
    # answer to a latitude of 452 -- a swapped pair, or degrees where radians were meant.
    for why, filter in [
        ("latitude past the pole", radius({"lat": 452.0, "lon": 13.0}, 100)),
        ("longitude past the line", radius({"lat": 52.0, "lon": 613.0}, 100)),
        ("a negative radius", radius(BERLIN, -5)),
        ("an upside-down box", box(52.0, 13.0, 53.0, 14.0)),
    ]:
        r = post(f"/tables/{places}/search",
                 {"vector": [1, 0, 0, 0], "top_k": 10, "filter": filter}, expect_ok=False)
        assert r.status_code == 400, (why, r.status_code, r.text[:300])


def test_a_point_written_as_an_array_matches_nothing_rather_than_guessing(places):
    # GeoJSON is [lon, lat]; every map UI is [lat, lon]. Guessing puts a record on the wrong
    # continent and nothing about the answer looks wrong, so an array is not a point.
    post(f"/tables/{places}/records", {"records": [
        {"id": "array", "vector": [1, 0, 0, 0], "payload": {"shop": [52.5163, 13.3777]}}]})
    r = post(f"/tables/{places}/search",
             {"vector": [1, 0, 0, 0], "top_k": 10, "filter": radius(BERLIN, 1000)})
    # `reichstag` is 270 m away and belongs here; `array` holds the same coordinates as
    # `gate` written the other way and must not.
    assert ids(r) == ["gate", "reichstag"], r.json()


def test_an_excluded_field_is_refused_here_as_everywhere():
    # The index is the only thing that answers a filter, so an excluded field would otherwise
    # give an empty page indistinguishable from "nothing is near there".
    name = f"geo_excl_{uuid.uuid4().hex[:8]}"
    post("/tables", {"name": name, "dimension": 4, "payload_index": {"except": ["shop"]}})
    try:
        post(f"/tables/{name}/records", {"records": [
            {"id": "a", "vector": [1, 0, 0, 0], "payload": {"shop": BERLIN}}]})
        r = post(f"/tables/{name}/search",
                 {"vector": [1, 0, 0, 0], "top_k": 10, "filter": radius(BERLIN, 1000)},
                 expect_ok=False)
        assert r.status_code == 400, (r.status_code, r.text[:300])
        assert "shop" in r.text, r.text[:300]
    finally:
        delete(f"/tables/{name}", expect_ok=False)


def test_the_field_report_counts_the_points_it_keeps(places):
    report = get(f"/tables/{places}/payload_index").json()["data"]
    fields = {f["field"]: f for f in report["fields"]}
    assert fields["shop"]["geo_points"] == 3, fields["shop"]  # the array record has no point
    # And the two coordinates are still in the numeric half, which is what keeps a range
    # filter on `shop.lat` -- the only way to ask a geographic question before this -- working.
    assert fields["shop.lat"]["numeric_entries"] == 3, fields
