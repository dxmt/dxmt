"""Table statistics, dropping and rebuilding the vector index, and bulk insert.

Probed before this was written (2026-09-11), in both storage modes and across a restart: all
ten `/records/bulk` modes store the same records in the same way. Eight are wrappers of a few
lines that end in `insert_records`, the path a plain insert takes, and `deferred_index` calls
it directly -- so the modes are asserted here to be interchangeable, not tested as ten
features. Names such as `nvme_optimized` and `ultimate` describe optimisations that are not
there.

The same probe found the statistics wrong in four ways, now fixed and pinned below: the index
size was a literal 0 for every index, disk usage was 0 in disk mode, memory usage counted
vectors that live in a row file, and after `DELETE /index` the table still named the index it
had just lost -- which is also why the rebuild that followed was skipped.
"""

import json
import os

import pytest

from .helpers.client import call, get, post
from .helpers.corpus import BY_ID, CORPUS, DIM, ORDER_FROM_R1, ids_of, query_at

STORAGE_MODE = os.environ.get("DXMT_STORAGE_MODE", "memory")

#: `/records/bulk` used to take a `mode` with these ten values. Every one of them reached the
#: same function -- eight seven-line wrappers and a `insert_records_batch` whose body read
#: `// For now, just use regular insert` -- so they were ten names for one code path, each
#: promising an optimisation that did not exist. The field is gone; these are kept here to
#: check that a client still sending one is told so, rather than having it quietly ignored.
FORMER_BULK_MODES = [
    "standard",
    "lock_free",
    "streaming",
    "memory_pool",
    "deferred_index",
    "bulk_optimized",
    "ultra_fast",
    "vm_optimized",
    "nvme_optimized",
    "ultimate",
]
ALL_IDS = sorted(r["id"] for r in CORPUS)


def stats(table):
    return get(f"/tables/{table}/stats").json()["data"]


def nearest(table, vector, k=5):
    body = {"vector": list(vector), "top_k": k}
    return ids_of(post(f"/tables/{table}/search", body).json()["data"]["hits"])


def payload_of(record):
    """`GET /records/{id}` carries the payload as a JSON string; see test_crud.py."""
    raw = record.get("payload_json")
    return json.loads(raw) if isinstance(raw, str) else record.get("payload")


# ---- Statistics -----------------------------------------------------------------------------


def test_stats_describe_the_table(table):
    s = stats(table)
    assert s["name"] == table
    assert s["record_count"] == len(CORPUS)
    assert s["dimension"] == DIM
    assert s["resident"] is True
    # Every field the JavaScript SDK's `TableStats` declares is present and a count of bytes.
    for field in ("index_size_bytes", "memory_usage_bytes", "disk_usage_bytes"):
        assert isinstance(s[field], int) and s[field] >= 0, (field, s[field])


def test_the_index_is_counted_in_bytes(table):
    s = stats(table)
    assert s["index_size_bytes"] > 0, "eight indexed vectors reported an index of no size"
    assert s["memory_usage_bytes"] >= s["index_size_bytes"], "memory usage leaves the index out"


# Each of these two runs in one storage mode; scripts/api_tests.sh runs both modes, so each is
# exercised once per full run.
@pytest.mark.skipif(STORAGE_MODE != "disk", reason="row files exist only in disk mode")
def test_disk_usage_is_the_row_files_and_grows_with_them(fresh_table):
    before = stats(fresh_table)["disk_usage_bytes"]
    assert before > 0, "records and vectors are in row files, and disk_usage_bytes said 0"
    extra = [
        {"id": f"x{i}", "vector": [0.1, 0.2, 0.3, 0.1 * i], "payload": {"note": "x" * 200}}
        for i in range(20)
    ]
    post(f"/tables/{fresh_table}/records", {"records": extra})
    assert stats(fresh_table)["disk_usage_bytes"] > before


@pytest.mark.skipif(STORAGE_MODE != "memory", reason="a memory-mode table has no row files")
def test_a_table_held_in_memory_has_no_files_of_its_own(table):
    assert stats(table)["disk_usage_bytes"] == 0


def test_stats_follow_inserts_and_deletes(fresh_table):
    post(f"/tables/{fresh_table}/records", {"records": [{"id": "extra", "vector": [0.5] * DIM}]})
    assert stats(fresh_table)["record_count"] == len(CORPUS) + 1
    post("/query", {"query": f"DELETE FROM {fresh_table} WHERE id = 'extra'"})
    assert stats(fresh_table)["record_count"] == len(CORPUS)


def test_stats_for_a_table_that_does_not_exist_are_a_404():
    assert get("/tables/definitely_not_a_table/stats", expect_ok=False).status_code == 404


# ---- Dropping and rebuilding the index ------------------------------------------------------


def test_dropping_the_index_keeps_every_record_and_every_answer(fresh_table):
    assert nearest(fresh_table, query_at("r1")) == ORDER_FROM_R1

    call("DELETE", f"/tables/{fresh_table}/index")

    assert stats(fresh_table)["index_type"] == "brute_force", (
        "after a drop the table still named the index it no longer had"
    )
    # Brute force must give the answer the index gave, for every record. The fifth neighbour of
    # r1 is one of four records at exactly the same distance; it came back as r7 or r8 from
    # brute force, which cut to k before breaking the tie by id.
    assert nearest(fresh_table, query_at("r1")) == ORDER_FROM_R1
    for record in CORPUS:
        assert nearest(fresh_table, record["vector"], k=1) == [record["id"]]
    assert stats(fresh_table)["record_count"] == len(CORPUS)


def test_rebuilding_after_a_drop_really_rebuilds(fresh_table):
    call("DELETE", f"/tables/{fresh_table}/index")
    data = post(f"/tables/{fresh_table}/index/rebuild").json()["data"]

    assert data["table"] == fresh_table
    assert "  " not in data["message"], (
        f"the message has a hole where a number was meant to go: {data['message']!r}"
    )
    # The rebuild used to be skipped here: the dropped table still claimed its old algorithm,
    # so a brute-force index holding every row looked up to date.
    assert stats(fresh_table)["index_type"] != "brute_force", "the rebuild after a drop was skipped"
    # The whole order, fifth neighbour included. r5 is one of four records at exactly the same
    # distance from r1, and a rebuilt graph used to choose among them by walk order: the index
    # cut to `k` before the storage layer's tie-break by id could see the others, so in memory
    # mode the answer changed after a rebuild.
    assert nearest(fresh_table, query_at("r1")) == ORDER_FROM_R1
    for record in CORPUS:
        assert nearest(fresh_table, record["vector"], k=1) == [record["id"]]
    assert stats(fresh_table)["record_count"] == len(CORPUS)


def test_rebuilding_an_intact_index_changes_no_answer(fresh_table):
    post(f"/tables/{fresh_table}/index/rebuild")
    assert nearest(fresh_table, query_at("r1")) == ORDER_FROM_R1


def test_a_record_written_after_the_drop_is_found_before_and_after_the_rebuild(fresh_table):
    call("DELETE", f"/tables/{fresh_table}/index")
    late = [0.0, 0.0, 0.5, 0.5]  # nearer to itself than to r7 or r8, which it lies between
    post(f"/tables/{fresh_table}/records", {"records": [{"id": "late", "vector": late}]})
    assert nearest(fresh_table, late, k=1) == ["late"]

    post(f"/tables/{fresh_table}/index/rebuild")
    assert nearest(fresh_table, late, k=1) == ["late"]
    assert stats(fresh_table)["record_count"] == len(CORPUS) + 1


@pytest.mark.parametrize(
    "method, path",
    [
        ("DELETE", "/tables/definitely_not_a_table/index"),
        ("POST", "/tables/definitely_not_a_table/index/rebuild"),
    ],
)
def test_index_operations_on_a_table_that_does_not_exist_are_a_404(method, path):
    assert call(method, path, expect_ok=False).status_code == 404


# ---- Bulk insert ----------------------------------------------------------------------------


def test_a_bulk_insert_stores_the_records(empty_table):
    data = post(f"/tables/{empty_table}/records/bulk", {"records": CORPUS}).json()["data"]

    assert data["count"] == len(CORPUS)
    assert sorted(data["inserted_ids"]) == ALL_IDS
    assert "mode" not in data, "the response still echoes a mode the endpoint no longer has"
    assert stats(empty_table)["record_count"] == len(CORPUS)
    for record in CORPUS:
        assert nearest(empty_table, record["vector"], k=1) == [record["id"]]
    stored = get(f"/tables/{empty_table}/records/r3").json()["data"]
    assert payload_of(stored) == BY_ID["r3"]["payload"]


@pytest.mark.parametrize("mode", FORMER_BULK_MODES + ["warp_speed"])
def test_a_mode_is_refused_now_that_the_endpoint_has_none(empty_table, mode):
    """Ten names for one code path were removed from the server, and for a while a body still
    carrying one was accepted and ignored -- which is how every SDK went on sending `mode`
    through a release that had deleted it. A field the server does not have is now an error
    naming it, so a client saying something the server does not do finds out."""
    response = post(
        f"/tables/{empty_table}/records/bulk",
        {"records": CORPUS, "mode": mode},
        expect_ok=False,
    )
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == "INVALID_INPUT"
    assert "mode" in error["message"], error
    assert stats(empty_table)["record_count"] == 0, "the refused batch was stored anyway"


def test_a_wrong_dimension_is_refused(empty_table):
    response = post(
        f"/tables/{empty_table}/records/bulk",
        {"records": [{"id": "bad", "vector": [1.0, 2.0]}]},
        expect_ok=False,
    )
    assert response.status_code == 400, response.text
    assert get(f"/tables/{empty_table}/records/bad", expect_ok=False).status_code == 404


def test_an_empty_batch_is_a_400(empty_table):
    response = post(f"/tables/{empty_table}/records/bulk", {"records": []}, expect_ok=False)
    assert response.status_code == 400, response.text


def test_bulk_insert_into_a_table_that_does_not_exist_is_a_404():
    response = post(
        "/tables/definitely_not_a_table/records/bulk", {"records": CORPUS}, expect_ok=False
    )
    assert response.status_code == 404, response.text
