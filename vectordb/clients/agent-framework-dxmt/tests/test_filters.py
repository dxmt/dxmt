"""Portable filters to the server's shape. No server needed."""

from __future__ import annotations

import pytest
from agent_framework import Filter, FilterGroup

from agent_framework_dxmt._filters import translate


def cond(key, type_, value=...):
    operation = {"type": type_} if value is ... else {"type": type_, "value": value}
    return {"key": key, "operation": operation}


def test_none_is_no_filter(definition):
    assert translate(None, definition) is None


@pytest.mark.parametrize(
    ("f", "server"),
    [
        (Filter("text", "eq", "x"), {"must": [cond("body", "Equal", "x")]}),
        (Filter("text", "ne", "x"), {"must": [cond("body", "NotEqual", "x")]}),
        (Filter("number", "gt", 1), {"must": [cond("price", "GreaterThan", 1)]}),
        (Filter("number", "gte", 1), {"must": [cond("price", "GreaterThanOrEqual", 1)]}),
        (Filter("number", "lt", 1), {"must": [cond("price", "LessThan", 1)]}),
        (Filter("number", "lte", 1), {"must": [cond("price", "LessThanOrEqual", 1)]}),
        (
            Filter("number", "between", (1, 2)),
            {"must": [cond("price", "GreaterThanOrEqual", 1), cond("price", "LessThanOrEqual", 2)]},
        ),
        (Filter("integer", "in", [1, 2]), {"must": [cond("integer", "In", [1, 2])]}),
        (Filter("integer", "not_in", [1, 2]), {"must": [cond("integer", "NotIn", [1, 2])]}),
        (Filter("flag", "is_null"), {"must": [cond("flag", "IsNull")]}),
        (Filter("flag", "is_not_null"), {"must_not": [cond("flag", "IsEmpty")]}),
        (Filter("flag", "exists"), {"must_not": [cond("flag", "IsEmpty")]}),
        (Filter("tags", "contains", "a"), {"must": [cond("tags", "Equal", "a")]}),
        (Filter("tags", "contains_any", ["a", "b"]), {"must": [cond("tags", "In", ["a", "b"])]}),
        (
            Filter("tags", "contains_all", ["a", "b"]),
            {"must": [cond("tags", "Equal", "a"), cond("tags", "Equal", "b")]},
        ),
        (Filter("text", "starts_with", "he"), {"must": [cond("body", "StartsWith", "he")]}),
        (Filter("text", "ends_with", "ld"), {"must": [cond("body", "EndsWith", "ld")]}),
        (Filter("text", "contains_text", "lo w"), {"must": [cond("body", "Contains", "lo w")]}),
        (Filter("id", "eq", 7), {"must": [cond("id", "Equal", 7)]}),
        (Filter("id", "in", [7, 8]), {"must": [cond("id", "In", [7, 8])]}),
    ],
)
def test_leaves_use_storage_names(definition, f, server):
    assert translate(f, definition) == server


def test_and_or_not_groups(definition):
    both = FilterGroup("and", [Filter("number", "gt", 1), Filter("flag", "eq", True)])
    assert translate(both, definition) == {"must": [cond("price", "GreaterThan", 1), cond("flag", "Equal", True)]}
    either = FilterGroup("or", [Filter("integer", "eq", 1), Filter("integer", "eq", 2)])
    assert translate(either, definition) == {"should": [cond("integer", "Equal", 1), cond("integer", "Equal", 2)]}
    negated = FilterGroup("not", [Filter("flag", "eq", True)])
    assert translate(negated, definition) == {"must_not": [cond("flag", "Equal", True)]}
    mixed = FilterGroup("and", [Filter("number", "gt", 1), either, negated])
    assert translate(mixed, definition) == {
        "must": [cond("price", "GreaterThan", 1)],
        "should": [cond("integer", "Equal", 1), cond("integer", "Equal", 2)],
        "must_not": [cond("flag", "Equal", True)],
    }


@pytest.mark.parametrize(
    "f",
    [
        FilterGroup(
            "or",
            [FilterGroup("and", [Filter("number", "gt", 1), Filter("flag", "eq", True)]), Filter("integer", "eq", 1)],
        ),
        FilterGroup(
            "and",
            [
                FilterGroup("or", [Filter("integer", "eq", 1), Filter("integer", "eq", 2)]),
                FilterGroup("or", [Filter("flag", "eq", True), Filter("flag", "eq", False)]),
            ],
        ),
        FilterGroup("not", [FilterGroup("and", [Filter("number", "gt", 1), Filter("flag", "eq", True)])]),
        Filter("embedding", "eq", [1, 2, 3]),
        Filter("missing", "eq", 1),
        Filter("text", "custom.op", 1),
    ],
)
def test_shapes_without_a_flat_form_are_refused(definition, f):
    with pytest.raises(NotImplementedError):
        translate(f, definition)
