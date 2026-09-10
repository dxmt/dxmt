"""Insert, read, update, delete -- and the shapes the caller gets back.

One thing here is worth knowing before reading the assertions: the same record has two shapes
depending on which endpoint returns it. A search hit carries `payload` as an object; a
`GET /records/{id}` carries `payload_json` as a *string*. The Python SDK hides this in
`_normalize_record`, so a caller who uses the SDK never finds out, and a caller who uses HTTP
directly finds out the hard way. These tests pin both shapes rather than pretend one of them
away -- if they are ever unified, this file is where the decision surfaces.
"""

import json

from .helpers.client import delete, get, post, put, succeeded
from .helpers.corpus import BY_ID, CORPUS, create_table, ids_of, insert_corpus, query_at


def payload_of(record):
    """A record from `GET /records/{id}`, whose payload arrives as a JSON string."""
    raw = record.get("payload_json")
    return json.loads(raw) if isinstance(raw, str) else record.get("payload")


def test_every_inserted_record_can_be_read_back(table):
    for record in CORPUS:
        got = get(f"/tables/{table}/records/{record['id']}").json()["data"]
        assert got["id"] == record["id"]
        assert payload_of(got) == record["payload"]


def test_insert_reports_what_it_inserted(fresh_table):
    body = {"records": [{"id": "new1", "vector": [0.0, 0.0, 1.0, 0.0], "payload": {"rank": 99}}]}
    data = post(f"/tables/{fresh_table}/records", body).json()["data"]
    assert data["count"] == 1
    assert data["inserted_ids"] == ["new1"]


def test_a_repeated_id_replaces_rather_than_duplicates(fresh_table):
    """Inserting an id that already exists is an upsert, not an error and not a second copy."""
    body = {"records": [{"id": "r1", "vector": [0.0, 0.0, 0.0, 1.0], "payload": {"rank": 42}}]}
    post(f"/tables/{fresh_table}/records", body)

    got = get(f"/tables/{fresh_table}/records/r1").json()["data"]
    assert payload_of(got) == {"rank": 42}

    # And the table did not grow: searching for everything still returns eight records.
    hits = post(
        f"/tables/{fresh_table}/search", {"vector": query_at("r1"), "top_k": 50}
    ).json()["data"]["hits"]
    assert len(hits) == len(CORPUS)


def test_update_replaces_the_payload_it_is_given(fresh_table):
    """`PUT` replaces; the fields not mentioned are gone. Worth pinning because "update" reads
    like a merge, and a caller who assumes merge silently loses the other fields."""
    put(f"/tables/{fresh_table}/records/r1", {"payload": {"rank": 100}})
    got = get(f"/tables/{fresh_table}/records/r1").json()["data"]
    assert payload_of(got) == {"rank": 100}, "city and tag should be gone, not merged"


def test_update_bumps_the_version(fresh_table):
    data = put(f"/tables/{fresh_table}/records/r1", {"payload": {"rank": 100}}).json()["data"]
    assert data["record"]["version"] == 2


def test_an_updated_vector_changes_who_is_nearest(fresh_table):
    """The index has to see the update, not just the record store. Moving `r8` onto `r1`'s
    position must make it `r1`'s nearest neighbour after itself."""
    put(f"/tables/{fresh_table}/records/r8", {"vector": list(BY_ID["r1"]["vector"])})
    hits = post(
        f"/tables/{fresh_table}/search", {"vector": query_at("r1"), "top_k": 2}
    ).json()["data"]["hits"]
    assert set(ids_of(hits)) == {"r1", "r8"}


def test_delete_removes_from_both_the_store_and_the_index(fresh_table):
    data = delete(f"/tables/{fresh_table}/records", body={"ids": ["r1", "r2"]}).json()["data"]
    assert data["deleted_count"] == 2
    assert data["requested_count"] == 2

    assert not succeeded(get(f"/tables/{fresh_table}/records/r1", expect_ok=False))

    # The deleted records must not come back from a search either -- an index that keeps a
    # tombstoned node reachable returns ids whose records no longer exist.
    hits = post(
        f"/tables/{fresh_table}/search", {"vector": query_at("r1"), "top_k": 50}
    ).json()["data"]["hits"]
    assert "r1" not in ids_of(hits)
    assert "r2" not in ids_of(hits)
    assert len(hits) == len(CORPUS) - 2


def test_deleting_something_that_is_not_there_reports_the_difference(fresh_table):
    data = delete(f"/tables/{fresh_table}/records", body={"ids": ["r1", "nope"]}).json()["data"]
    assert data["requested_count"] == 2
    assert data["deleted_count"] == 1


def test_reading_a_record_that_does_not_exist_is_a_404(table):
    response = get(f"/tables/{table}/records/nope", expect_ok=False)
    assert response.status_code == 404


def test_a_batch_insert_is_all_present_afterwards(empty_table):
    """Eight records in one request, then every one readable. A batch that reported success
    while dropping part of itself would pass a test that only checked the response."""
    insert_corpus(empty_table)
    for record in CORPUS:
        got = get(f"/tables/{empty_table}/records/{record['id']}").json()["data"]
        assert got["id"] == record["id"]
