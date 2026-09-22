"""Paging `records/list` by cursor, over HTTP.

`tests/list_cursor_test.rs` checks the exactly-once property while writing between pages; this
pins the wire shape a client depends on: `next_cursor` on every page but the last, the pages
joining up into the whole list, and the two ways to ask wrongly being refused rather than guessed.
"""

from .helpers.client import post, succeeded


def page(table, **body):
    return post(f"/tables/{table}/records/list", body, expect_ok=False)


def test_following_next_cursor_walks_the_whole_list_in_order(table):
    everything = [r["id"] for r in page(table, limit=1000).json()["data"]["records"]]
    walked, cursor, pages = [], None, 0
    while True:
        data = page(table, limit=3, **({"cursor": cursor} if cursor else {})).json()["data"]
        walked += [r["id"] for r in data["records"]]
        pages += 1
        cursor = data["next_cursor"]
        if cursor is None:
            break
        assert pages < 100, "the cursor is not advancing"
    assert walked == everything
    assert data["records"], "the last page should carry records, not be an empty page after them"


def test_a_filtered_scroll_walks_the_matches(table):
    f = {"must": [{"key": "city", "operation": {"type": "Equal", "value": "London"}}]}
    everything = [r["id"] for r in page(table, limit=1000, filter=f).json()["data"]["records"]]
    first = page(table, limit=1, filter=f).json()["data"]
    second = page(table, limit=5, filter=f, cursor=first["next_cursor"]).json()["data"]
    assert [r["id"] for r in first["records"] + second["records"]] == everything
    assert second["next_cursor"] is None


def test_cursor_and_offset_together_are_refused(table):
    first = page(table, limit=2).json()["data"]
    response = page(table, limit=2, offset=2, cursor=first["next_cursor"])
    assert response.status_code == 400 and response.json()["error"]["code"] == "INVALID_INPUT"


def test_a_cursor_the_server_did_not_issue_is_refused(table):
    for bad in ["not-a-cursor", "0000", "zz" * 10]:
        response = page(table, limit=2, cursor=bad)
        assert not succeeded(response) and response.status_code == 400, f"{bad!r}: {response.status_code}"
