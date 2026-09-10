"""Filters on a numeric field.

Split from the string filters on purpose. Since `aed5b89` a numeric field is a sorted
`Vec<(OrderedFloat, RecordId)>` and a string field is still an inverted `HashMap` per distinct
value, so `rank` and `city` take different code paths through `payload_index.rs`. A suite that
only ever filters on one of them covers half of filtering, and the half it covers is whichever
one the author happened to pick.

The corpus has `rank` 1..8 on records `r1`..`r8`, so the expected answer to any range is
arithmetic.
"""

from .helpers.client import post
from .helpers.corpus import ids_of, query_at
from .helpers.filters import cond, must, must_not


def search(table, filt, k=8, vector=None):
    body = {"vector": vector or query_at("r1"), "top_k": k}
    if filt is not None:
        body["filter"] = filt
    return post(f"/tables/{table}/search", body).json()["data"]["hits"]


def ranks(hits):
    return sorted(h["payload"]["rank"] for h in hits)


def test_greater_than_or_equal(table):
    assert ranks(search(table, must(cond("rank", "GreaterThanOrEqual", 6)))) == [6, 7, 8]


def test_greater_than_excludes_the_boundary(table):
    assert ranks(search(table, must(cond("rank", "GreaterThan", 6)))) == [7, 8]


def test_less_than_or_equal(table):
    assert ranks(search(table, must(cond("rank", "LessThanOrEqual", 2)))) == [1, 2]


def test_less_than_excludes_the_boundary(table):
    assert ranks(search(table, must(cond("rank", "LessThan", 2)))) == [1]


def test_a_closed_range_is_the_intersection(table):
    """Two conditions on one key. This is the shape that reads as satisfiable on each half and
    is empty only jointly, which is why `cannot_match_anything` groups conditions by key."""
    filt = must(
        cond("rank", "GreaterThanOrEqual", 3),
        cond("rank", "LessThanOrEqual", 5),
    )
    assert ranks(search(table, filt)) == [3, 4, 5]


def test_a_range_with_nothing_in_it_returns_nothing(table):
    """`rank` is an integer 1..8, so nothing lies strictly between 4 and 5."""
    filt = must(
        cond("rank", "GreaterThan", 4),
        cond("rank", "LessThan", 5),
    )
    assert search(table, filt) == []


def test_a_range_beyond_every_value_returns_nothing(table):
    assert search(table, must(cond("rank", "GreaterThan", 100))) == []
    assert search(table, must(cond("rank", "LessThan", 0))) == []


def test_equality_on_an_integer(table):
    """The sorted index stores integers and floats under one scaled key, so an `Equal` arrives
    as a float and a comparison by variant would match nothing at all."""
    hits = search(table, must(cond("rank", "Equal", 4)))
    assert ids_of(hits) == ["r4"]


def test_in_and_not_in(table):
    assert ranks(search(table, must(cond("rank", "In", [2, 5, 7])))) == [2, 5, 7]
    assert ranks(search(table, must(cond("rank", "NotIn", [2, 5, 7])))) == [1, 3, 4, 6, 8]


def test_not_equal_keeps_everything_else(table):
    assert ranks(search(table, must(cond("rank", "NotEqual", 4)))) == [1, 2, 3, 5, 6, 7, 8]


def test_must_not_removes(table):
    assert ranks(search(table, must_not(cond("rank", "GreaterThanOrEqual", 4)))) == [1, 2, 3]


def test_a_filter_on_a_field_no_record_has_returns_nothing(table):
    """Not an error and not everything: no record carries `missing`, so nothing can match a
    condition that needs a value for it."""
    assert search(table, must(cond("missing", "GreaterThanOrEqual", 0))) == []


def test_a_filtered_search_still_orders_by_distance(table):
    """The filter decides membership; the query decides order. Both matter, and a filter path
    that returned its matches in index order rather than distance order would pass a test that
    only checked the set."""
    hits = search(table, must(cond("rank", "GreaterThanOrEqual", 3)), k=8)
    assert ids_of(hits) == ["r3", "r4", "r5", "r6", "r7", "r8"]


def test_top_k_applies_after_the_filter(table):
    hits = search(table, must(cond("rank", "GreaterThanOrEqual", 3)), k=2)
    assert ids_of(hits) == ["r3", "r4"]
