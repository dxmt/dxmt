"""The administrative routes: schema changes, dropping an index, snapshots and backups.

These are mounted under `/api/v1` and, until this file, were the part of the router with no
functional coverage at all -- `PATCH /tables/{t}`, `DELETE /tables/{t}/index`,
`POST /tables/{t}/batch_search`, `POST /snapshot` and `POST /backup`. The storage tests reach
them in-process, which cannot catch a route that stopped being mounted or a handler that renames
a field on the way out.

What is deliberately not asserted here:

* **That a backup restores.** `tests/backup_restore_test.rs` proves that in-process, where it
  can stop the server and swap the WAL directory out. Over HTTP the honest assertion is that the
  backup is written, describes itself, and refuses the destinations that would corrupt it.
* **That a snapshot shortens the WAL.** The log is truncated after the snapshot is read back,
  and the size of a file this suite cannot see is not an HTTP contract.
"""

import json
import os
import shutil
import uuid

import pytest

from .helpers.client import call, get, post
from .helpers.corpus import BY_ID, CORPUS, ORDER_FROM_R1, ids_of, query_at

#: The *server* writes the backup, so the destination has to be a path it can write: outside
#: `DB_WAL_DIR`, which `backup_to` rejects as an overlapping destination, and *inside*
#: `DB_BACKUP_ROOT`, which bounds where a request may have the database copied to at all.
#:
#: This was `/tmp/dxmt_api_backup` and passed because the root defaulted to the WAL directory's
#: parent, which is `/tmp` under this script. Reading the root the server was actually given is
#: the difference between testing the rule and happening not to trip over it.
BACKUP_ROOT = os.environ.get("DXMT_BACKUP_ROOT")

#: Set by `scripts/api_tests.sh`. Absent when attaching to a server someone else started.
WAL_DIR = os.environ.get("DXMT_WAL_DIR")


def describe(name):
    return get(f"/tables/{name}").json()["data"]


def alter(name, body, expect_ok=True):
    return call("PATCH", f"/tables/{name}", body, expect_ok=expect_ok)


def search(name, record_id="r1", k=5, path="search"):
    body = {"vector": query_at(record_id), "top_k": k}
    return ids_of(post(f"/tables/{name}/{path}", body).json()["data"]["hits"])


def payload_of(name, record_id):
    """`GET /records/{id}` carries the payload as a JSON string; see test_crud.py."""
    record = get(f"/tables/{name}/records/{record_id}").json()["data"]
    raw = record.get("payload_json")
    return json.loads(raw) if isinstance(raw, str) else record.get("payload")


# ---- PATCH /tables/{t} — schema evolution ----------------------------------------------------


def test_altering_a_table_bumps_its_schema_version(fresh_table):
    before = describe(fresh_table)["table"]["schema_version"]

    after = alter(fresh_table, {"description": "now documented"}).json()["data"]

    assert after["schema_version"] == before + 1
    assert after["description"] == "now documented"


def test_adding_a_payload_field_declares_it_without_touching_records(fresh_table):
    field = {"name": "tier", "field_type": "string", "nullable": True}

    table = alter(fresh_table, {"add_fields": [field]}).json()["data"]

    assert [f["name"] for f in table["payload_schema"]] == ["tier"]
    # The declaration is metadata. Every record is still there, still answering as before.
    assert describe(fresh_table)["record_count"] == len(CORPUS)
    assert search(fresh_table) == ORDER_FROM_R1
    assert payload_of(fresh_table, "r1") == BY_ID["r1"]["payload"]


def test_dropping_a_payload_field_keeps_the_data_it_described(fresh_table):
    """`drop_fields` removes the declaration, not the values -- records keep their payloads."""
    alter(fresh_table, {"add_fields": [{"name": "city", "field_type": "string"}]})

    table = alter(fresh_table, {"drop_fields": ["city"]}).json()["data"]

    assert table["payload_schema"] in (None, []), table["payload_schema"]
    assert payload_of(fresh_table, "r1") == BY_ID["r1"]["payload"], (
        "dropping the declaration deleted the data it described"
    )


def test_a_declared_field_survives_a_further_alter(fresh_table):
    """Two alters in a row: the second must not silently discard what the first declared."""
    alter(fresh_table, {"add_fields": [{"name": "tier", "field_type": "string"}]})

    table = alter(fresh_table, {"description": "unrelated change"}).json()["data"]

    assert [f["name"] for f in table["payload_schema"]] == ["tier"]
    assert table["schema_version"] >= 2


def test_altering_a_table_that_does_not_exist_is_a_404():
    assert alter("definitely_not_a_table", {"description": "x"}, expect_ok=False).status_code == 404


# `DELETE /tables/{t}/index` is covered in test_table_ops.py, and more strictly than a second
# copy here would be: it asserts the table stops naming the index it no longer has
# (`index_type == "brute_force"`), that brute force reproduces the index's answer for every
# record, that a rebuild afterwards really rebuilds, and that a record written after the drop is
# found both before and after it.


# ---- POST /tables/{t}/batch_search -----------------------------------------------------------


def test_batch_search_answers_exactly_as_search_does(table):
    """The route is an alias -- `server.rs` points it at `search_vectors`.

    Pinned because an alias that quietly starts answering differently is drift no unit test
    sees: both paths claim to answer "what is nearest to this vector".
    """
    assert search(table, path="batch_search") == search(table, path="search") == ORDER_FROM_R1


# ---- POST /snapshot --------------------------------------------------------------------------


def test_a_snapshot_reports_how_many_tables_it_wrote(table):
    data = post("/snapshot").json()["data"]

    assert data["tables"] >= 1, data
    assert str(data["tables"]) in data["message"]


def test_a_snapshot_leaves_the_data_readable(fresh_table):
    """Snapshotting must not be observable from the API except by finishing."""
    before = search(fresh_table)

    post("/snapshot")

    assert describe(fresh_table)["record_count"] == len(CORPUS)
    assert search(fresh_table) == before


# ---- POST /backup ----------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("DXMT_BACKUP_ROOT"),
    reason="DXMT_BACKUP_ROOT is only set when this suite starts the server",
)
def test_a_backup_is_written_and_describes_itself(table):
    dest = os.path.join(BACKUP_ROOT or "", f"out_{uuid.uuid4().hex[:8]}")
    try:
        manifest = post("/backup", {"path": dest}).json()["data"]

        assert manifest["tables"] >= 1
        assert manifest["files"], "a backup that copied no files is not a backup"
        assert manifest["bytes"] > 0
        assert "DB_WAL_DIR" in manifest["restore"], "the manifest must say how to restore it"
        assert os.path.isdir(dest), f"the server reported a backup it did not write: {dest}"
    finally:
        shutil.rmtree(dest, ignore_errors=True)


@pytest.mark.skipif(not WAL_DIR, reason="DXMT_WAL_DIR is only set when this suite starts the server")
def test_a_backup_refuses_a_destination_inside_the_wal_directory():
    """Source and destination must not overlap.

    A backup written under `DB_WAL_DIR` makes a later recursive copy self-referential, so it is
    refused before anything is written rather than left half-done.
    """
    response = post("/backup", {"path": os.path.join(WAL_DIR, "inside")}, expect_ok=False)

    assert response.status_code == 400, response.text
    assert not os.path.exists(os.path.join(WAL_DIR, "inside")), "it wrote before refusing"


def test_a_backup_without_a_path_is_refused():
    assert post("/backup", {}, expect_ok=False).status_code in (400, 422)
