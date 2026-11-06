"""Which Haystack filters get a server form, and what it is. No server needed."""

import pytest
from haystack.errors import FilterError

from haystack_integrations.document_stores.dxmt.filters import translate


def cond(key, type_, value=...):
    op = {"type": type_} if value is ... else {"type": type_, "value": value}
    return {"key": key, "operation": op}


def leaf(field, op, value):
    return {"field": field, "operator": op, "value": value}


def as_dict(filters):
    f = translate(filters)
    return None if f is None else f.to_dict()


class TestLeaves:
    def test_equality_on_meta_is_the_payload_path(self):
        assert as_dict(leaf("meta.page", "==", "100")) == {"must": [cond("meta.page", "Equal", "100")]}

    def test_a_bare_key_is_meta(self):
        assert as_dict(leaf("page", "==", "100")) == {"must": [cond("meta.page", "Equal", "100")]}

    def test_content_and_id_are_themselves(self):
        assert as_dict(leaf("content", "==", "x"))["must"][0]["key"] == "content"
        assert as_dict(leaf("id", "==", "x"))["must"][0]["key"] == "id"

    def test_not_equal(self):
        assert as_dict(leaf("meta.n", "!=", 2)) == {"must": [cond("meta.n", "NotEqual", 2)]}

    @pytest.mark.parametrize("op,type_", [(">", "GreaterThan"), (">=", "GreaterThanOrEqual"), ("<", "LessThan"), ("<=", "LessThanOrEqual")])
    def test_numeric_ranges(self, op, type_):
        assert as_dict(leaf("meta.n", op, 1.5)) == {"must": [cond("meta.n", type_, 1.5)]}

    def test_in(self):
        assert as_dict(leaf("meta.c", "in", ["a", "b"])) == {"must": [cond("meta.c", "In", ["a", "b"])]}

    def test_not_in_counts_a_missing_field_as_not_in(self):
        # Haystack: `not any(equal)` is True for a document without the field; the server's
        # NotIn is not, so the disjunction with IsEmpty is what makes them agree.
        assert as_dict(leaf("meta.c", "not in", ["a"])) == {
            "should": [cond("meta.c", "NotIn", ["a"]), cond("meta.c", "IsEmpty")]
        }


class TestNoServerForm:
    @pytest.mark.parametrize(
        "f",
        [
            leaf("meta.n", "==", None),
            leaf("meta.n", "!=", None),
            leaf("meta.d", ">", "1972-12-11T19:54:58"),
            leaf("meta.d", "==", "1972-12-11T19:54:58"),
            leaf("meta.n", ">", "100"),
            leaf("meta.n", ">", [1, 2]),
            leaf("meta.n", "==", [1, 2]),
            {"operator": "NOT", "conditions": [leaf("meta.a", "==", 1), leaf("meta.b", "==", 2)]},
            {"operator": "OR", "conditions": [
                {"operator": "AND", "conditions": [leaf("meta.a", "==", 1), leaf("meta.b", "==", 2)]},
                leaf("meta.c", "==", 3),
            ]},
            {"operator": "AND", "conditions": [
                leaf("meta.a", "not in", [1]), leaf("meta.b", "not in", [2]),
            ]},
        ],
    )
    def test_goes_to_the_evaluator(self, f):
        assert translate(f) is None

    def test_empty_is_no_filter(self):
        assert translate(None) is None and translate({}) is None


class TestLogic:
    def test_and(self):
        f = {"operator": "AND", "conditions": [leaf("meta.a", "==", 1), leaf("meta.b", ">", 2)]}
        assert as_dict(f) == {"must": [cond("meta.a", "Equal", 1), cond("meta.b", "GreaterThan", 2)]}

    def test_or(self):
        f = {"operator": "OR", "conditions": [leaf("meta.a", "==", 1), leaf("meta.b", "==", 2)]}
        assert as_dict(f) == {"should": [cond("meta.a", "Equal", 1), cond("meta.b", "Equal", 2)]}

    def test_not_of_one_condition(self):
        f = {"operator": "NOT", "conditions": [leaf("meta.a", "==", 1)]}
        assert as_dict(f) == {"must_not": [cond("meta.a", "Equal", 1)]}

    def test_and_with_a_negation_and_an_or_group(self):
        f = {"operator": "AND", "conditions": [
            leaf("meta.a", "==", 1),
            {"operator": "NOT", "conditions": [leaf("meta.b", "==", 2)]},
            {"operator": "OR", "conditions": [leaf("meta.c", "==", 3), leaf("meta.c", "==", 4)]},
        ]}
        assert as_dict(f) == {
            "must": [cond("meta.a", "Equal", 1)],
            "must_not": [cond("meta.b", "Equal", 2)],
            "should": [cond("meta.c", "Equal", 3), cond("meta.c", "Equal", 4)],
        }


class TestMalformed:
    @pytest.mark.parametrize(
        "f",
        [
            {"field": "meta.a", "value": 1},
            {"field": "meta.a", "operator": "=="},
            {"operator": "AND"},
            {"conditions": []},
            {"field": "meta.a", "operator": "in", "value": "not-a-list"},
            {"field": "meta.a", "operator": "~=", "value": 1},
        ],
    )
    def test_raises_filter_error(self, f):
        with pytest.raises(FilterError):
            translate(f)
