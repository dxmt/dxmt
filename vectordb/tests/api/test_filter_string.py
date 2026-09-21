"""Filters on a string field.

The other half of filtering. A string value stays in `value_to_ids`, the inverted map keyed by
distinct value, while numbers moved to a sorted array (`aed5b89`) -- so equality here is a hash
lookup and the substring tests walk every distinct value, neither of which the numeric tests
touch. The corpus gives `city` four values across eight records (London, Moscow, Berlin, Paris,
two each) and `tag` three (alpha 3, beta 3, gamma 2).
"""

from .helpers.client import post
from .helpers.corpus import ids_of, query_at
from .helpers.filters import cond, must, must_not, should


def search(table, filt, k=8):
    body = {"vector": query_at("r1"), "top_k": k}
    if filt is not None:
        body["filter"] = filt
    return post(f"/tables/{table}/search", body).json()["data"]["hits"]


def cities(hits):
    return sorted(h["payload"]["city"] for h in hits)


def test_equality(table):
    assert ids_of(search(table, must(cond("city", "Equal", "London")))) == ["r1", "r2"]


def test_equality_on_an_absent_value_returns_nothing(table):
    assert search(table, must(cond("city", "Equal", "Atlantis"))) == []


def test_not_equal_keeps_the_rest(table):
    hits = search(table, must(cond("city", "NotEqual", "London")))
    assert ids_of(hits) == ["r3", "r4", "r5", "r6", "r7", "r8"]


def test_in_and_not_in(table):
    assert cities(search(table, must(cond("city", "In", ["London", "Paris"])))) == [
        "London", "London", "Paris", "Paris",
    ]
    assert cities(search(table, must(cond("city", "NotIn", ["London", "Paris"])))) == [
        "Berlin", "Berlin", "Moscow", "Moscow",
    ]


def test_contains(table):
    """A substring test can only be satisfied by a string, so a record whose value is a number
    can never match one -- which is why `Contains` is allowed to take part in the index's
    emptiness proof."""
    assert ids_of(search(table, must(cond("city", "Contains", "osco")))) == ["r3", "r4"]
    assert search(table, must(cond("city", "Contains", "zzz"))) == []


def test_starts_with_and_ends_with(table):
    assert ids_of(search(table, must(cond("city", "StartsWith", "Ber")))) == ["r5", "r6"]
    assert ids_of(search(table, must(cond("city", "EndsWith", "ris")))) == ["r7", "r8"]
    # Case matters, and silently lower-casing would be a change in behaviour worth catching.
    assert search(table, must(cond("city", "StartsWith", "ber"))) == []


def test_two_conditions_on_different_keys_intersect(table):
    """`alpha` is r1, r3, r6; London is r1, r2. Only r1 is both."""
    filt = must(cond("tag", "Equal", "alpha"), cond("city", "Equal", "London"))
    assert ids_of(search(table, filt)) == ["r1"]


def test_two_conditions_that_cannot_both_hold(table):
    """Empty across two keys, which no single field's values can disprove -- the index cannot
    shortcut this one, so the search has to discover it."""
    filt = must(cond("city", "Equal", "London"), cond("city", "Equal", "Paris"))
    assert search(table, filt) == []


def test_should_is_a_union(table):
    filt = should(cond("city", "Equal", "London"), cond("city", "Equal", "Paris"))
    assert ids_of(search(table, filt)) == ["r1", "r2", "r7", "r8"]


def test_must_not_removes(table):
    hits = search(table, must_not(cond("tag", "Equal", "alpha")))
    assert ids_of(hits) == ["r2", "r4", "r5", "r7", "r8"]


def test_a_string_condition_on_a_numeric_field_matches_nothing(table):
    """`rank` holds numbers; asking whether it starts with "1" must not match rank 1. A field
    that stored numbers as strings, or a comparison that stringified before testing, would."""
    assert search(table, must(cond("rank", "StartsWith", "1"))) == []


def test_a_filtered_search_still_orders_by_distance(table):
    hits = search(table, must(cond("tag", "Equal", "alpha")))
    assert ids_of(hits) == ["r1", "r3", "r6"]


# A whole array or object as the target of an equality. The payload index keys scalars only, so
# which endpoint answered decided the result: `records/list` and a selective search returned
# nothing while a broad search returned the right records. It is refused now, the same way from
# every endpoint, rather than answered differently by each.
COMPOSITE_TARGETS = [
    ("Equal", ["alpha", "beta"]),
    ("NotEqual", {"a": 1}),
    ("In", [["alpha"]]),
    ("NotIn", [{"a": 1}]),
]


def test_an_array_or_object_target_is_refused_everywhere(table):
    from .helpers.client import succeeded

    for op, value in COMPOSITE_TARGETS:
        filt = must(cond("tag", op, value))
        endpoints = {
            "search": (f"/tables/{table}/search", {"vector": query_at("r1"), "top_k": 8, "filter": filt}),
            "exact search": (
                f"/tables/{table}/search",
                {"vector": query_at("r1"), "top_k": 8, "filter": filt, "search_mode": "exact"},
            ),
            "records/list": (f"/tables/{table}/records/list", {"limit": 8, "filter": filt}),
        }
        for name, (path, body) in endpoints.items():
            response = post(path, body, expect_ok=False)
            assert not succeeded(response), f"{name} accepted {op} {value!r}"
            assert response.status_code == 400, f"{name}: {response.status_code} {response.text}"
            error = response.json()["error"]
            assert error["code"] == "INVALID_INPUT", error
            assert "tag" in error["message"], f"the message should name the field: {error}"


def test_a_scalar_target_in_the_same_operations_still_works(table):
    # The refusal is about the target's shape, not the operations: these are the same four.
    assert sorted(ids_of(search(table, must(cond("tag", "In", ["gamma"]))))) == ["r4", "r7"]
    assert search(table, must(cond("tag", "NotIn", ["alpha", "beta", "gamma"]))) == []
