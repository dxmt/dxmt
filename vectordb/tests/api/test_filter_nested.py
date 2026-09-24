"""Filters on a nested payload field, over HTTP.

A payload is arbitrary JSON and people nest it — `{"meta": {"lang": "en"}}` is what a document
store hands you. Until this, the only filterable names were the top-level keys: `meta` held an
object, an object matches nothing scalar, and a filter on `meta.lang` matched nothing at all.
Not an error, a wrong answer — the empty page that this codebase keeps having to stop returning.

Two things are worth checking over HTTP rather than only in the agreement test. That every read
path answers a nested filter the same way, because `search`, `records/list` and `records/count`
reach the index by different routes; and that a payload which really holds a key with a dot in
it is still filterable by the name it was written under, since the same string now has two
possible readings.
"""

import uuid

import pytest

from .helpers.client import delete, get, post


@pytest.fixture(scope="module")
def nested():
    name = f"nested_{uuid.uuid4().hex[:8]}"
    post("/tables", {"name": name, "dimension": 2})
    post(
        f"/tables/{name}/records",
        {
            "records": [
                {
                    "id": "r1",
                    "vector": [1.0, 0.0],
                    "payload": {
                        "meta": {"lang": "en", "score": 10, "tags": ["a", "b"]},
                        "top": "x",
                    },
                },
                {
                    "id": "r2",
                    "vector": [0.0, 1.0],
                    "payload": {"meta": {"lang": "zh", "score": 20}, "top": "y"},
                },
                {
                    "id": "r3",
                    "vector": [0.7, 0.7],
                    "payload": {
                        "meta": {"lang": "en", "score": 30, "deep": {"level": "three"}},
                        # Written with a dot in the key, on a record that also has `meta`.
                        "meta.lang": "written-not-derived",
                    },
                },
                {"id": "r4", "vector": [0.5, 0.5], "payload": {"top": "z"}},
            ]
        },
    )
    yield name
    delete(f"/tables/{name}", expect_ok=False)


def cond(key, op, value):
    return {"must": [{"key": key, "operation": {"type": op, "value": value}}]}


def counted(table, filt):
    return post(f"/tables/{table}/records/count", {"filter": filt}).json()["data"]["count"]


def listed(table, filt):
    body = {"limit": 100, "filter": filt}
    records = post(f"/tables/{table}/records/list", body).json()["data"]["records"]
    return sorted(r["id"] for r in records)


def searched(table, filt):
    body = {"vector": [1.0, 0.0], "top_k": 100, "filter": filt}
    hits = post(f"/tables/{table}/search", body).json()["data"]["hits"]
    return sorted(h["id"] for h in hits)


CASES = [
    # (name, filter, the ids it must return)
    # r1 only, not r3: r3 also carries a literal `"meta.lang"` key, and a key written with a
    # dot wins over the path reading of the same string. So r3's `meta.lang` is
    # "written-not-derived", and its nested `{"meta": {"lang": "en"}}` is not reachable under
    # that name at all. Writing both is a payload asking for trouble; what matters is that the
    # index and the record-by-record matcher pick the same one of the two.
    ("a nested string", cond("meta.lang", "Equal", "en"), ["r1"]),
    ("a nested number", cond("meta.score", "GreaterThanOrEqual", 20), ["r2", "r3"]),
    # One level matches any element, the same as at the top level.
    ("an array under a path", cond("meta.tags", "Equal", "b"), ["r1"]),
    ("two levels down", cond("meta.deep.level", "Equal", "three"), ["r3"]),
    # r3 holds `"meta.lang": "written-not-derived"` as a key. The literal wins, so this is the
    # one record that matches, and `meta.lang == "en"` above does *not* return it twice or
    # merge the two values.
    ("a key written with a dot", cond("meta.lang", "Equal", "written-not-derived"), ["r3"]),
    ("a path nothing has", cond("meta.nope", "Equal", "anything"), []),
    ("the top level still", cond("top", "Equal", "x"), ["r1"]),
]


@pytest.mark.parametrize("name,filt,expected", CASES, ids=[c[0] for c in CASES])
def test_every_read_path_answers_a_nested_filter_the_same_way(nested, name, filt, expected):
    assert listed(nested, filt) == expected, f"records/list: {name}"
    assert searched(nested, filt) == expected, f"search: {name}"
    assert counted(nested, filt) == len(expected), f"records/count: {name}"


def test_a_nested_path_is_reported_as_an_indexed_field(nested):
    report = get(f"/tables/{nested}/payload_index").json()["data"]
    fields = {f["field"] for f in report["fields"]}
    # The paths are what the index files, so this is where a reader finds out that nesting is
    # indexed path by path -- and what it costs.
    assert {"meta.lang", "meta.score", "meta.tags", "meta.deep.level"} <= fields, fields


def test_excluding_a_parent_excludes_the_paths_under_it(nested):
    name = f"{nested}_policy"
    post(
        "/tables",
        {"name": name, "dimension": 2, "payload_index": {"except": ["meta"]}},
    )
    try:
        post(
            f"/tables/{name}/records",
            {
                "records": [
                    {
                        "id": "r1",
                        "vector": [1.0, 0.0],
                        "payload": {"meta": {"lang": "en"}, "top": "x"},
                    }
                ]
            },
        )
        # An excluded name covers its subtree: excluding `meta` and still indexing `meta.lang`
        # would make the exclusion useless against the field that actually costs something.
        refused = post(
            f"/tables/{name}/records/count",
            {"filter": cond("meta.lang", "Equal", "en")},
            expect_ok=False,
        )
        assert refused.status_code == 400, refused.text[:300]
        assert "meta.lang" in refused.text, refused.text[:300]
        assert counted(name, cond("top", "Equal", "x")) == 1
    finally:
        delete(f"/tables/{name}", expect_ok=False)
