"""`IsEmpty` and `IsNull`, over HTTP.

"Which records are missing this field" had no expression at all. `NotEqual` answers a different
question — a record that has the field with some other value satisfies it too — so the only way
to find the rows a pipeline failed to enrich was to page the whole table and check by hand.

The three states this has to keep apart are the ones the rest of the index deliberately blurs:
a field that is absent, a field whose value is `null`, and a field holding an empty array. They
are all `IsEmpty`; only the second is `IsNull`. An array that *contains* a null is neither.
"""

import uuid

import pytest

from .helpers.client import delete, post, put


@pytest.fixture(scope="module")
def presence():
    name = f"presence_{uuid.uuid4().hex[:8]}"
    post("/tables", {"name": name, "dimension": 2})
    post(
        f"/tables/{name}/records",
        {
            "records": [
                {"id": "absent", "vector": [1.0, 0.0], "payload": {"other": 1}},
                {"id": "null", "vector": [0.9, 0.1], "payload": {"maybe": None}},
                {"id": "empty_array", "vector": [0.8, 0.2], "payload": {"maybe": []}},
                {"id": "value", "vector": [0.7, 0.3], "payload": {"maybe": "here"}},
                {"id": "array_with_null", "vector": [0.6, 0.4], "payload": {"maybe": [None]}},
                {"id": "object", "vector": [0.5, 0.5], "payload": {"maybe": {"nested": 1}}},
            ]
        },
    )
    yield name
    delete(f"/tables/{name}", expect_ok=False)


def cond(key, op):
    return {"must": [{"key": key, "operation": {"type": op}}]}


def listed(table, filt):
    body = {"limit": 100, "filter": filt}
    records = post(f"/tables/{table}/records/list", body).json()["data"]["records"]
    return sorted(r["id"] for r in records)


def searched(table, filt):
    body = {"vector": [1.0, 0.0], "top_k": 100, "filter": filt}
    hits = post(f"/tables/{table}/search", body).json()["data"]["hits"]
    return sorted(h["id"] for h in hits)


def counted(table, filt):
    return post(f"/tables/{table}/records/count", {"filter": filt}).json()["data"]["count"]


def test_is_empty_is_absent_null_or_an_empty_array(presence):
    expected = ["absent", "empty_array", "null"]
    assert listed(presence, cond("maybe", "IsEmpty")) == expected
    assert searched(presence, cond("maybe", "IsEmpty")) == expected
    assert counted(presence, cond("maybe", "IsEmpty")) == len(expected)


def test_is_null_is_only_the_null(presence):
    # An array holding a null counts: the condition asks about a value in the field, and one of
    # the elements is null -- the same rule every other operator follows on an array.
    expected = ["array_with_null", "null"]
    assert listed(presence, cond("maybe", "IsNull")) == expected
    assert searched(presence, cond("maybe", "IsNull")) == expected


def test_a_field_nothing_carries_is_empty_everywhere(presence):
    everything = sorted(["absent", "array_with_null", "empty_array", "null", "object", "value"])
    assert listed(presence, cond("no_such_field", "IsEmpty")) == everything
    assert listed(presence, cond("no_such_field", "IsNull")) == []


def test_must_not_is_empty_is_the_records_that_have_something(presence):
    have = {"must_not": [{"key": "maybe", "operation": {"type": "IsEmpty"}}]}
    assert listed(presence, have) == ["array_with_null", "object", "value"]


def test_it_follows_an_update(presence):
    name = f"{presence}_update"
    post("/tables", {"name": name, "dimension": 2})
    try:
        post(
            f"/tables/{name}/records",
            {"records": [{"id": "r1", "vector": [1.0, 0.0], "payload": {"maybe": None}}]},
        )
        assert listed(name, cond("maybe", "IsEmpty")) == ["r1"]
        # Filling the field in has to take the record out of the empty set. An index that
        # marked "this record is empty here" on write and never unmarked it would answer this
        # with r1 forever, and nothing else would notice.
        put(
            f"/tables/{name}/records/r1",
            {"vector": [1.0, 0.0], "payload": {"maybe": "filled"}},
        )
        assert listed(name, cond("maybe", "IsEmpty")) == []
        assert listed(name, cond("maybe", "IsNull")) == []
        # And back again.
        put(f"/tables/{name}/records/r1", {"vector": [1.0, 0.0], "payload": {"maybe": None}})
        assert listed(name, cond("maybe", "IsEmpty")) == ["r1"]
    finally:
        delete(f"/tables/{name}", expect_ok=False)
