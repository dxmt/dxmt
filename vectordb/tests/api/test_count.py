"""`records/count`, over HTTP.

"How many of these are there" had no answer short of paging every match and counting them
client-side: table stats carry a count for the table and nothing for a condition. So the
property that matters is not that the endpoint returns *a* number, it is that the number is the
one a caller would have got the long way — the same filter, through `records/list`, counted by
hand. A count computed from a different index than the one that answers the query is a count
that can be right today and wrong after a delete.
"""

import sys
from pathlib import Path

import pytest

import json

from .helpers.client import call, post, succeeded

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semantics import matches  # noqa: E402


def payload_of(record):
    """`records/list` returns the payload as a JSON string, not an object."""
    raw = record.get("payload_json")
    return json.loads(raw) if raw else (record.get("payload") or {})


def count(table, filter=None, expect_ok=True):
    body = {} if filter is None else {"filter": filter}
    return post(f"/tables/{table}/records/count", body, expect_ok=expect_ok)


def listed(table, filter=None):
    body = {"limit": 1000}
    if filter is not None:
        body["filter"] = filter
    return post(f"/tables/{table}/records/list", body).json()["data"]["records"]


def test_an_unfiltered_count_is_the_table(table):
    assert count(table).json()["data"]["count"] == len(listed(table))


FILTERS = [
    {"must": [{"key": "city", "operation": {"type": "Equal", "value": "London"}}]},
    {"must": [{"key": "rank", "operation": {"type": "GreaterThanOrEqual", "value": 3}}]},
    {"must_not": [{"key": "city", "operation": {"type": "Equal", "value": "London"}}]},
    {"should": [
        {"key": "city", "operation": {"type": "Equal", "value": "London"}},
        {"key": "city", "operation": {"type": "Equal", "value": "Moscow"}},
    ]},
    {"must": [{"key": "city", "operation": {"type": "StartsWith", "value": "L"}}]},
]


@pytest.mark.parametrize("filter", FILTERS, ids=range(len(FILTERS)))
def test_a_filtered_count_agrees_with_the_records_that_filter_returns(table, filter):
    counted = count(table, filter).json()["data"]["count"]
    fetched = listed(table, filter)
    assert counted == len(fetched), (
        f"count said {counted} and the list returned {len(fetched)}; one of the two reads this "
        f"filter differently, and a caller cannot tell which"
    )


def test_it_follows_a_delete(empty_table):
    # A count that comes from an index the deletes do not reach is right until the first
    # delete. This is the cheapest way to notice that.
    records = [
        {"id": f"c{i}", "vector": [1.0, float(i), 0.0, 0.0], "payload": {"keep": i % 2 == 0}}
        for i in range(10)
    ]
    post(f"/tables/{empty_table}/records", {"records": records})
    keep = {"must": [{"key": "keep", "operation": {"type": "Equal", "value": True}}]}
    assert count(empty_table).json()["data"]["count"] == 10
    assert count(empty_table, keep).json()["data"]["count"] == 5

    call("DELETE", f"/tables/{empty_table}/records", {"ids": ["c0", "c2"]})
    assert count(empty_table).json()["data"]["count"] == 8
    assert count(empty_table, keep).json()["data"]["count"] == 3


def test_a_filter_it_cannot_answer_is_refused_not_guessed(table):
    # The same refusal `search` and `records/list` give: a count that silently answered zero
    # for a filter the server cannot evaluate would be indistinguishable from an empty match.
    response = count(
        table,
        {"must": [{"key": "tags", "operation": {"type": "Equal", "value": ["a", "b"]}}]},
        expect_ok=False,
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "INVALID_INPUT"


def test_a_table_that_does_not_exist_is_a_404():
    response = count("definitely_not_a_table", expect_ok=False)
    assert response.status_code == 404, response.text
    assert not succeeded(response)


def test_the_model_agrees(table):
    """The same answer the shared filter model gives, over the records the table holds."""
    records = listed(table)
    for filter in FILTERS:
        expected = sum(
            1
            for record in records
            if matches(record["id"], payload_of(record), filter)
        )
        assert count(table, filter).json()["data"]["count"] == expected, filter
