"""`records/list` and the vector field, which used to mean two different things.

The handler returned whatever the record's own `vector` field held. For an in-memory table that
is the vector; once the vectors live in a row file — the default as soon as there is a WAL
directory — the record's field is empty and the API returned `[]`. So the same request answered
differently depending on `DXMT_VECTOR_STORE`, a server setting the caller cannot see, and `[]` was
indistinguishable from "this record has no vector". Meanwhile `GET /records/{id}` and `POST
/search` with `with_vector` both read through the row file and always answered with the vector.

Three routes that return a record's vector, and one of them disagreed with the other two
depending on how the server was started.

`with_vector` makes it one thing: off by default (a page of 100 rows at 768 dimensions is over a
megabyte of floats that almost no caller listing records wants), and when asked for, the vector
— from wherever it actually lives.

This file is why the suite runs twice. Every assertion here is about the two storage modes
agreeing, and a single-mode run would pass while the bug was live.
"""

import os

import pytest

from .helpers.client import get, post
from .helpers.corpus import BY_ID, CORPUS, DIM

STORAGE_MODE = os.environ.get("DXMT_STORAGE_MODE", "memory")


def listed(table, **body):
    return post(f"/tables/{table}/records/list", {"limit": 50, **body}).json()["data"]["records"]


def test_vectors_are_omitted_by_default(table):
    """The default is the cheap answer, and it is the same answer in both storage modes."""
    for record in listed(table):
        assert record["vector"] == [], (
            f"listing returned a vector nobody asked for, in {STORAGE_MODE} mode: "
            f"{record['id']} carries {len(record['vector'])} floats"
        )


def test_with_vector_returns_the_real_vector_in_either_storage_mode(table):
    """The assertion the bug would have failed: in disk mode this used to be `[]`."""
    records = {r["id"]: r for r in listed(table, with_vector=True)}

    for expected in CORPUS:
        got = records[expected["id"]]["vector"]
        assert len(got) == DIM, (
            f"{expected['id']} came back with {len(got)} floats instead of {DIM} in "
            f"{STORAGE_MODE} mode — this is the storage-dependent answer the flag exists to fix"
        )
        assert got == pytest.approx(expected["vector"], abs=1e-6), (
            f"{expected['id']}'s vector does not round-trip: {got} != {expected['vector']}"
        )


def test_the_three_routes_that_return_a_vector_agree(table):
    """`records/list`, `GET records/{id}` and `search` with `with_vector` describe one record."""
    record_id = "r3"
    expected = BY_ID[record_id]["vector"]

    from_list = {r["id"]: r for r in listed(table, with_vector=True)}[record_id]["vector"]
    from_get = get(f"/tables/{table}/records/{record_id}").json()["data"]["vector"]
    hits = post(
        f"/tables/{table}/search",
        {"vector": expected, "top_k": 1, "with_vector": True},
    ).json()["data"]["hits"]

    assert from_list == pytest.approx(expected, abs=1e-6)
    assert from_get == pytest.approx(expected, abs=1e-6)
    assert hits[0]["vector"] == pytest.approx(expected, abs=1e-6), (
        "search's with_vector disagrees with the other two"
    )


def test_asking_for_vectors_does_not_change_which_records_come_back(table):
    """The flag is about what each record carries, not about which ones are selected."""
    without = [r["id"] for r in listed(table)]
    with_them = [r["id"] for r in listed(table, with_vector=True)]
    assert without == with_them


def test_the_flag_survives_paging_and_filtering(table):
    page = post(
        f"/tables/{table}/records/list",
        {
            "limit": 2,
            "offset": 1,
            "with_vector": True,
            "filter": {"must": [{"key": "city", "operation": {"type": "Equal", "value": "Paris"}}]},
        },
    ).json()["data"]

    assert page["total"] == 2, "Paris holds r7 and r8"
    assert len(page["records"]) == 1, "offset 1 of 2 leaves one"
    assert len(page["records"][0]["vector"]) == DIM
