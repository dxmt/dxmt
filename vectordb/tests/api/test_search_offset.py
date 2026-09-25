"""`offset` on a search: page two without fetching page one twice.

Paging a search meant asking for `2 * top_k` and throwing half of it away on the client — over
the wire, with the payloads attached. The property that matters is that the pages *tile*: every
hit appears once, in one page, in the same order the unpaged search would have put it in. A
paging scheme that repeats or skips a row under no concurrent writes is worse than none, because
the caller cannot see it happening.

What this deliberately does not promise is that a deep offset is cheap. The index is asked for
`offset + top_k` and the prefix is dropped, because an approximate index cannot start in the
middle of a ranking it has not computed.
"""

import math
import uuid

import pytest

from .helpers.client import delete, post


@pytest.fixture(scope="module")
def paged():
    name = f"paged_{uuid.uuid4().hex[:8]}"
    post("/tables", {"name": name, "dimension": 2})
    # Twenty records on an arc, so the ranking against [1, 0] is r0, r1, r2, ... exactly.
    post(f"/tables/{name}/records", {"records": [
        {"id": f"r{i}", "vector": [math.cos(i * 0.05), math.sin(i * 0.05)], "payload": {"i": i}}
        for i in range(20)
    ]})
    yield name
    delete(f"/tables/{name}", expect_ok=False)


def ids(table, **body):
    body.setdefault("vector", [1.0, 0.0])
    hits = post(f"/tables/{table}/search", body).json()["data"]["hits"]
    return [h["id"] for h in hits]


def test_pages_tile_the_ranking(paged):
    whole = ids(paged, top_k=15)
    pages = [ids(paged, top_k=5, offset=n) for n in (0, 5, 10)]
    assert [hit for page in pages for hit in page] == whole, pages


def test_an_offset_past_the_end_is_an_empty_page(paged):
    assert ids(paged, top_k=5, offset=100) == []
    # And the one straddling the end is short rather than wrong.
    assert ids(paged, top_k=5, offset=18) == ["r18", "r19"]


def test_it_applies_to_the_exact_mode_too(paged):
    # `exact` skips the whole approximate machinery -- a separate code path, and one where an
    # offset applied in the wrong place would be invisible on an ANN result and obvious here.
    assert ids(paged, top_k=3, offset=3, search_mode="exact") == ["r3", "r4", "r5"]


def test_it_composes_with_a_filter(paged):
    # The offset is taken after the filter, not before: skipping rows the filter was going to
    # remove would make page two start in a different place than page one ended.
    below = {"must": [{"key": "i", "operation": {"type": "LessThan", "value": 6}}]}
    first = ids(paged, top_k=3, filter=below)
    second = ids(paged, top_k=3, offset=3, filter=below)
    assert first == ["r0", "r1", "r2"], first
    assert second == ["r3", "r4", "r5"], second


def test_it_composes_with_a_score_threshold(paged):
    # Same reasoning for the floor: what it removes must not count against the offset.
    # cos(0.05 i) against [1, 0], so this floor keeps the first handful and no more.
    floored = {"score_threshold": 0.95}
    kept = ids(paged, top_k=10, **floored)
    assert len(kept) > 2, kept
    assert ids(paged, top_k=10, offset=2, **floored) == kept[2:]
