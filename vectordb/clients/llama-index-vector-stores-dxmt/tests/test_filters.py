"""The translation from LlamaIndex ``MetadataFilters`` to the server's filter shape.

Each case states the LlamaIndex meaning (``build_metadata_filter_fn`` in llama-index-core is the
reference) and the exact server filter that means the same, or the ``ValueError`` for a shape
the server cannot take. Nothing here is approximated: the translation this tests replaced
turned ``ALL`` into ``IN``, ``NOT`` into ``AND`` and dropped ``IS_EMPTY``, three queries that
ran and returned the wrong rows.
"""

import pytest
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.dxmt.utils import convert_llamaindex_filters_to_dxmt


def one(key, value, op):
    return convert_llamaindex_filters_to_dxmt(
        MetadataFilters(filters=[MetadataFilter(key=key, value=value, operator=op)])
    )


def cond(key, type_, value=...):
    operation = {"type": type_} if value is ... else {"type": type_, "value": value}
    return {"key": key, "operation": operation}


class TestOneOperator:
    @pytest.mark.parametrize(
        ("op", "value", "server"),
        [
            (FilterOperator.EQ, "AI", cond("k", "Equal", "AI")),
            (FilterOperator.NE, "draft", cond("k", "NotEqual", "draft")),
            (FilterOperator.GT, 0.8, cond("k", "GreaterThan", 0.8)),
            (FilterOperator.GTE, 2020, cond("k", "GreaterThanOrEqual", 2020)),
            (FilterOperator.LT, 10, cond("k", "LessThan", 10)),
            (FilterOperator.LTE, 10, cond("k", "LessThanOrEqual", 10)),
            (FilterOperator.IN, ["a", "b"], cond("k", "In", ["a", "b"])),
            (FilterOperator.NIN, ["a", "b"], cond("k", "NotIn", ["a", "b"])),
            (FilterOperator.EQ, "2024", cond("k", "Equal", "2024")),
        ],
    )
    def test_direct_mappings(self, op, value, server):
        assert one("k", value, op) == {"must": [server]}

    def test_in_with_one_value_is_still_a_list(self):
        assert one("k", "a", FilterOperator.IN) == {"must": [cond("k", "In", ["a"])]}

    def test_contains_is_equal_because_an_array_field_matches_on_any_element(self):
        # LlamaIndex: "metadata array contains value" -- `value in metadata_value`.
        assert one("tags", "rag", FilterOperator.CONTAINS) == {"must": [cond("tags", "Equal", "rag")]}

    def test_any_is_in(self):
        assert one("tags", ["a", "b"], FilterOperator.ANY) == {"must": [cond("tags", "In", ["a", "b"])]}

    def test_all_is_one_equal_per_value_all_of_which_must_hold(self):
        # The old translation sent `In`, which matches a record carrying *either* tag.
        assert one("tags", ["a", "b"], FilterOperator.ALL) == {
            "must": [cond("tags", "Equal", "a"), cond("tags", "Equal", "b")]
        }

    def test_all_with_no_values_is_refused(self):
        with pytest.raises(ValueError, match="ALL"):
            one("tags", [], FilterOperator.ALL)

    def test_text_match_is_a_substring_test(self):
        assert one("body", "machine", FilterOperator.TEXT_MATCH) == {
            "must": [cond("body", "Contains", "machine")]
        }

    def test_text_match_insensitive_has_no_server_form(self):
        with pytest.raises(ValueError, match="TEXT_MATCH_INSENSITIVE"):
            one("body", "machine", FilterOperator.TEXT_MATCH_INSENSITIVE)

    def test_is_empty(self):
        # Used to be dropped, so the filter silently widened to every record.
        assert one("k", None, FilterOperator.IS_EMPTY) == {"must": [cond("k", "IsEmpty")]}

    def test_comparing_against_a_list_with_eq_is_refused(self):
        with pytest.raises(ValueError, match="Equal"):
            one("tags", ["a", "b"], FilterOperator.EQ)


class TestConditions:
    A = MetadataFilter(key="a", value=1, operator=FilterOperator.EQ)
    B = MetadataFilter(key="b", value=2, operator=FilterOperator.EQ)
    C = MetadataFilter(key="c", value=3, operator=FilterOperator.EQ)

    def test_and(self):
        f = MetadataFilters(filters=[self.A, self.B], condition=FilterCondition.AND)
        assert convert_llamaindex_filters_to_dxmt(f) == {
            "must": [cond("a", "Equal", 1), cond("b", "Equal", 2)]
        }

    def test_and_is_the_default(self):
        f = MetadataFilters(filters=[self.A, self.B])
        assert convert_llamaindex_filters_to_dxmt(f)["must"] == [
            cond("a", "Equal", 1),
            cond("b", "Equal", 2),
        ]

    def test_or(self):
        f = MetadataFilters(filters=[self.A, self.B], condition=FilterCondition.OR)
        assert convert_llamaindex_filters_to_dxmt(f) == {
            "should": [cond("a", "Equal", 1), cond("b", "Equal", 2)]
        }

    def test_not_means_none_of_these_match(self):
        # LlamaIndex: "Match when none of the filters match" -- not (a or b).
        f = MetadataFilters(filters=[self.A, self.B], condition=FilterCondition.NOT)
        assert convert_llamaindex_filters_to_dxmt(f) == {
            "must_not": [cond("a", "Equal", 1), cond("b", "Equal", 2)]
        }

    def test_empty_and_none(self):
        assert convert_llamaindex_filters_to_dxmt(MetadataFilters(filters=[])) == {}
        assert convert_llamaindex_filters_to_dxmt(None) == {}

    def test_a_range_is_two_conditions_on_one_key(self):
        f = MetadataFilters(
            filters=[
                MetadataFilter(key="year", value=2020, operator=FilterOperator.GTE),
                MetadataFilter(key="year", value=2023, operator=FilterOperator.LTE),
            ]
        )
        assert convert_llamaindex_filters_to_dxmt(f) == {
            "must": [cond("year", "GreaterThanOrEqual", 2020), cond("year", "LessThanOrEqual", 2023)]
        }


class TestNesting:
    A = MetadataFilter(key="a", value=1, operator=FilterOperator.EQ)
    B = MetadataFilter(key="b", value=2, operator=FilterOperator.EQ)
    C = MetadataFilter(key="c", value=3, operator=FilterOperator.EQ)
    D = MetadataFilter(key="d", value=4, operator=FilterOperator.EQ)

    def test_and_of_and_flattens(self):
        inner = MetadataFilters(filters=[self.A, self.B], condition=FilterCondition.AND)
        outer = MetadataFilters(filters=[inner, self.C], condition=FilterCondition.AND)
        assert convert_llamaindex_filters_to_dxmt(outer) == {
            "must": [cond("a", "Equal", 1), cond("b", "Equal", 2), cond("c", "Equal", 3)]
        }

    def test_or_of_or_flattens(self):
        inner = MetadataFilters(filters=[self.A, self.B], condition=FilterCondition.OR)
        outer = MetadataFilters(filters=[inner, self.C], condition=FilterCondition.OR)
        assert convert_llamaindex_filters_to_dxmt(outer) == {
            "should": [cond("a", "Equal", 1), cond("b", "Equal", 2), cond("c", "Equal", 3)]
        }

    def test_not_inside_and_is_must_not(self):
        negated = MetadataFilters(filters=[self.B], condition=FilterCondition.NOT)
        outer = MetadataFilters(filters=[self.A, negated], condition=FilterCondition.AND)
        assert convert_llamaindex_filters_to_dxmt(outer) == {
            "must": [cond("a", "Equal", 1)],
            "must_not": [cond("b", "Equal", 2)],
        }

    def test_and_with_one_or_group_is_must_plus_should(self):
        either = MetadataFilters(filters=[self.B, self.C], condition=FilterCondition.OR)
        outer = MetadataFilters(filters=[self.A, either], condition=FilterCondition.AND)
        assert convert_llamaindex_filters_to_dxmt(outer) == {
            "must": [cond("a", "Equal", 1)],
            "should": [cond("b", "Equal", 2), cond("c", "Equal", 3)],
        }

    def test_or_of_and_groups_is_refused(self):
        # (a and b) or (c and d): the old translation nested `{"must": [...]}` inside
        # `should`, a shape the server rejects -- or, in the SDK's permissive days, read
        # as something else entirely.
        g1 = MetadataFilters(filters=[self.A, self.B], condition=FilterCondition.AND)
        g2 = MetadataFilters(filters=[self.C, self.D], condition=FilterCondition.AND)
        outer = MetadataFilters(filters=[g1, g2], condition=FilterCondition.OR)
        with pytest.raises(ValueError, match="OR"):
            convert_llamaindex_filters_to_dxmt(outer)

    def test_and_of_two_or_groups_is_refused(self):
        g1 = MetadataFilters(filters=[self.A, self.B], condition=FilterCondition.OR)
        g2 = MetadataFilters(filters=[self.C, self.D], condition=FilterCondition.OR)
        outer = MetadataFilters(filters=[g1, g2], condition=FilterCondition.AND)
        with pytest.raises(ValueError, match="OR"):
            convert_llamaindex_filters_to_dxmt(outer)

    def test_not_of_an_all_is_refused(self):
        # not (tags has a and tags has b) has no flat form; refused rather than guessed.
        every = MetadataFilter(key="tags", value=["a", "b"], operator=FilterOperator.ALL)
        outer = MetadataFilters(filters=[every], condition=FilterCondition.NOT)
        with pytest.raises(ValueError, match="NOT"):
            convert_llamaindex_filters_to_dxmt(outer)

    def test_an_empty_nested_group_is_skipped(self):
        outer = MetadataFilters(filters=[MetadataFilters(filters=[]), self.A])
        assert convert_llamaindex_filters_to_dxmt(outer) == {"must": [cond("a", "Equal", 1)]}
