"""Creating, listing, describing and dropping tables.

Two response shapes worth knowing, both of which this file got wrong on the first attempt by
assuming the obvious one:

    GET /tables        -> {"data": {"tables": [...], "limit": 50, "offset": 0}}
    GET /tables/{name} -> {"data": {"table": {...}, "record_count": N}}

Neither is a bare object or a bare list. The SDK flattens both, so a caller who only ever uses
the SDK never learns this, and a caller who reads the HTTP API does -- which is the asymmetry
this suite exists to remove.
"""

import uuid

from .helpers.client import call, delete, get, post, succeeded
from .helpers.corpus import CORPUS, DIM, create_table, insert_corpus, query_at


def a_name():
    return f"api_tbl_{uuid.uuid4().hex[:8]}"


def describe(name):
    return get(f"/tables/{name}").json()["data"]


def table_names():
    return [t["name"] for t in get("/tables").json()["data"]["tables"]]


def test_a_new_table_reports_what_it_was_asked_for(table):
    data = describe(table)
    assert data["table"]["name"] == table
    assert data["table"]["dimension"] == DIM
    assert data["table"]["distance_metric"] == "cosine"


def test_describing_a_table_reports_how_many_records_it_holds(table):
    assert describe(table)["record_count"] == len(CORPUS)


def test_a_new_table_appears_in_the_listing(table):
    assert table in table_names()


def test_a_duplicate_name_is_refused_rather_than_silently_reused(table):
    """409, not 200. Quietly handing back the existing table would let a caller believe it had
    a fresh one and then find someone else's records in it."""
    response = post(
        "/tables",
        {"name": table, "dimension": DIM, "distance_metric": "cosine"},
        expect_ok=False,
    )
    assert response.status_code == 409


def test_dropping_a_table_removes_it_from_the_listing(empty_table):
    delete(f"/tables/{empty_table}")
    assert empty_table not in table_names()
    assert not succeeded(get(f"/tables/{empty_table}", expect_ok=False))


def test_a_recreated_name_starts_empty(empty_table):
    """The same name after a drop must not inherit the old table's records. On disk that means
    the row files have to be cleared, and a stale `records/<t>.rec` left behind would show up
    here as records the caller never inserted."""
    insert_corpus(empty_table)
    hits = post(f"/tables/{empty_table}/search",
                {"vector": query_at("r1"), "top_k": 50}).json()["data"]["hits"]
    assert len(hits) == len(CORPUS)

    delete(f"/tables/{empty_table}")
    create_table(empty_table, DIM)

    hits = post(f"/tables/{empty_table}/search",
                {"vector": query_at("r1"), "top_k": 50}).json()["data"]["hits"]
    assert hits == [], "a recreated table inherited the old one's records"


def test_a_table_with_a_different_metric_answers_differently(table):
    """Euclidean and cosine disagree about which record is nearest once vectors differ in
    length, so a metric that was accepted and then ignored would show up here."""
    name = a_name()
    create_table(name, DIM, metric="euclidean")
    try:
        # Same directions as r1 and r2, but r1 is much shorter. Cosine calls the short one a
        # perfect match; Euclidean calls the long one closer to a long query.
        post(f"/tables/{name}/records", {"records": [
            {"id": "short", "vector": [0.10, 0.0, 0.0, 0.0]},
            {"id": "long", "vector": [1.00, 0.0, 0.0, 0.0]},
        ]})
        hits = post(f"/tables/{name}/search",
                    {"vector": [1.0, 0.0, 0.0, 0.0], "top_k": 1}).json()["data"]["hits"]
        assert [h["id"] for h in hits] == ["long"]
    finally:
        delete(f"/tables/{name}", expect_ok=False)


def test_getting_a_table_that_does_not_exist_is_a_404(table):
    assert get("/tables/definitely_not_a_table", expect_ok=False).status_code == 404


def test_a_recall_target_is_set_by_patch_and_calibration_reports_what_it_measured(empty_table):
    """`recall_target` is the one table setting that drives a measurement rather than a
    rebuild. The probe here is small on purpose: the point is the contract -- the field is
    stored, a calibration can be asked for, and what comes back names the budget, the recall
    it measured and the curve it chose from. Whether the budget is *right* is measured in
    `tests/recall/calibration_test.rs` on a corpus where it can be wrong."""
    rows = [
        {"id": f"c{i}", "vector": [((i * 7919 + d * 104729) % 1000) / 1000.0 - 0.5 for d in range(DIM)]}
        for i in range(64)
    ]
    post(f"/tables/{empty_table}/records", {"records": rows})

    call("PATCH", f"/tables/{empty_table}", {"recall_target": 0.9})
    table = get(f"/tables/{empty_table}").json()["data"]["table"]
    assert table["recall_target"] == 0.9
    assert table["recall"] is None, "nothing has measured anything yet"

    cal = post(f"/tables/{empty_table}/recall/calibrate").json()["data"]["calibration"]
    assert cal["target"] == 0.9
    assert cal["sample"] > 0 and cal["at_rows"] == 64
    assert cal["curve"][0]["ef_search"] <= cal["ef_search"]
    assert all(0.0 <= p["recall"] <= 1.0 for p in cal["curve"])
    assert cal["met"] is True, f"64 rows should be exact at any budget: {cal}"
    stats = get(f"/tables/{empty_table}/stats").json()["data"]
    assert stats["ef_search"]["calibrated"] == cal["ef_search"]
    assert stats["recall"]["measured"] == cal["measured"]

    # `0` opts the table out, and discards what was measured.
    call("PATCH", f"/tables/{empty_table}", {"recall_target": 0})
    assert get(f"/tables/{empty_table}").json()["data"]["table"]["recall"] is None
    assert post(f"/tables/{empty_table}/recall/calibrate").json()["data"]["calibration"] is None


def test_a_reachability_repair_reports_what_it_found_and_the_stats_keep_the_count(table):
    """The repair's effect is measured in `HnswIndex`'s own tests and on GIST (2.25% of stored
    vectors not returned by their own query before, 0.45% after, 1,563 nodes given an in-edge
    in 80 ms at 100K rows). Over HTTP the contract is the shape: two counts, and the stats
    carrying them forward."""
    data = post(f"/tables/{table}/index/repair").json()["data"]
    assert data["table"] == table
    assert isinstance(data["stranded"], int) and isinstance(data["repaired"], int)
    assert data["repaired"] <= data["stranded"]
    reach = get(f"/tables/{table}/stats").json()["data"]["reachability"]
    assert reach["stranded_at_last_repair"] == data["stranded"]
    assert reach["repaired_total"] >= data["repaired"]
