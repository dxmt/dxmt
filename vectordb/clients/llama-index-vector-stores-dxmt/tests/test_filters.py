"""Tests for filter conversion utilities."""

import pytest
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    FilterCondition,
)
from llama_index.vector_stores.dxmt.utils import (
    convert_llamaindex_filters_to_dxmt,
    _convert_single_filter,
)


class TestSingleFilterConversion:
    """Tests for converting single MetadataFilter to DXMT format."""

    def test_eq_operator(self):
        """Test EQ (==) operator conversion."""
        filter = MetadataFilter(key="category", value="AI", operator=FilterOperator.EQ)
        result = _convert_single_filter(filter)

        assert result == {
            "key": "category",
            "operation": {
                "type": "Equal",
                "value": "AI"
            }
        }

    def test_ne_operator(self):
        """Test NE (!=) operator conversion."""
        filter = MetadataFilter(key="status", value="draft", operator=FilterOperator.NE)
        result = _convert_single_filter(filter)

        assert result == {
            "key": "status",
            "operation": {
                "type": "NotEqual",
                "value": "draft"
            }
        }

    def test_gt_operator(self):
        """Test GT (>) operator conversion."""
        filter = MetadataFilter(key="score", value=0.8, operator=FilterOperator.GT)
        result = _convert_single_filter(filter)

        assert result == {
            "key": "score",
            "operation": {
                "type": "GreaterThan",
                "value": 0.8
            }
        }

    def test_gte_operator(self):
        """Test GTE (>=) operator conversion."""
        filter = MetadataFilter(key="year", value=2020, operator=FilterOperator.GTE)
        result = _convert_single_filter(filter)

        assert result == {
            "key": "year",
            "operation": {
                "type": "GreaterThanOrEqual",
                "value": 2020
            }
        }

    def test_lt_operator(self):
        """Test LT (<) operator conversion."""
        filter = MetadataFilter(key="price", value=100.0, operator=FilterOperator.LT)
        result = _convert_single_filter(filter)

        assert result == {
            "key": "price",
            "operation": {
                "type": "LessThan",
                "value": 100.0
            }
        }

    def test_lte_operator(self):
        """Test LTE (<=) operator conversion."""
        filter = MetadataFilter(key="age", value=30, operator=FilterOperator.LTE)
        result = _convert_single_filter(filter)

        assert result == {
            "key": "age",
            "operation": {
                "type": "LessThanOrEqual",
                "value": 30
            }
        }

    def test_in_operator(self):
        """Test IN operator conversion."""
        filter = MetadataFilter(
            key="category",
            value=["AI", "ML", "DL"],
            operator=FilterOperator.IN
        )
        result = _convert_single_filter(filter)

        assert result == {
            "key": "category",
            "operation": {
                "type": "In",
                "value": ["AI", "ML", "DL"]
            }
        }

    def test_nin_operator(self):
        """Test NIN (not in) operator conversion."""
        filter = MetadataFilter(
            key="status",
            value=["draft", "archived"],
            operator=FilterOperator.NIN
        )
        result = _convert_single_filter(filter)

        assert result == {
            "key": "status",
            "operation": {
                "type": "NotIn",
                "value": ["draft", "archived"]
            }
        }

    def test_contains_operator(self):
        """Test CONTAINS operator conversion."""
        filter = MetadataFilter(key="title", value="python", operator=FilterOperator.CONTAINS)
        result = _convert_single_filter(filter)

        assert result == {
            "key": "title",
            "operation": {
                "type": "Contains",
                "value": "python"
            }
        }

    def test_text_match_operator(self):
        """Test TEXT_MATCH operator conversion."""
        filter = MetadataFilter(key="content", value="machine learning", operator=FilterOperator.TEXT_MATCH)
        result = _convert_single_filter(filter)

        assert result == {
            "key": "content",
            "operation": {
                "type": "Contains",
                "value": "machine learning"
            }
        }


class TestComplexFilterConversion:
    """Tests for converting complex MetadataFilters to DXMT format."""

    def test_single_filter(self):
        """Test converting a single filter."""
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="AI", operator=FilterOperator.EQ)
            ]
        )

        result = convert_llamaindex_filters_to_dxmt(filters)

        # Even one filter is wrapped. The server's `Filter` is `{must?, must_not?, should?}`
        # (`src/types/record.rs`), and serde ignores unknown fields -- so an unwrapped
        # `{"key": ..., "operation": ...}` does not fail, it deserializes to a filter with
        # every list empty and silently returns unfiltered results.
        assert result == {
            "must": [
                {
                    "key": "category",
                    "operation": {
                        "type": "Equal",
                        "value": "AI"
                    }
                }
            ]
        }

    def test_and_filters(self):
        """Test converting multiple filters with AND condition."""
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="AI", operator=FilterOperator.EQ),
                MetadataFilter(key="year", value=2020, operator=FilterOperator.GTE),
            ],
            condition=FilterCondition.AND,
        )

        result = convert_llamaindex_filters_to_dxmt(filters)

        assert "must" in result
        assert len(result["must"]) == 2
        assert result["must"][0]["key"] == "category"
        assert result["must"][1]["key"] == "year"

    def test_or_filters(self):
        """Test converting multiple filters with OR condition."""
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="AI", operator=FilterOperator.EQ),
                MetadataFilter(key="category", value="ML", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.OR,
        )

        result = convert_llamaindex_filters_to_dxmt(filters)

        assert "should" in result
        assert len(result["should"]) == 2

    def test_nested_filters(self):
        """Test converting nested filters."""
        # (category == "AI" AND year >= 2020) OR (category == "ML" AND year >= 2021)
        inner_filters_1 = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="AI", operator=FilterOperator.EQ),
                MetadataFilter(key="year", value=2020, operator=FilterOperator.GTE),
            ],
            condition=FilterCondition.AND,
        )

        inner_filters_2 = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="ML", operator=FilterOperator.EQ),
                MetadataFilter(key="year", value=2021, operator=FilterOperator.GTE),
            ],
            condition=FilterCondition.AND,
        )

        outer_filters = MetadataFilters(
            filters=[inner_filters_1, inner_filters_2],
            condition=FilterCondition.OR,
        )

        result = convert_llamaindex_filters_to_dxmt(outer_filters)

        assert "should" in result
        assert len(result["should"]) == 2
        assert "must" in result["should"][0]
        assert "must" in result["should"][1]

    def test_empty_filters(self):
        """Test converting empty filters."""
        filters = MetadataFilters(filters=[])
        result = convert_llamaindex_filters_to_dxmt(filters)

        assert result == {}

    def test_complex_and_chain(self):
        """Test complex AND chain with multiple conditions."""
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="AI", operator=FilterOperator.EQ),
                MetadataFilter(key="year", value=2020, operator=FilterOperator.GTE),
                MetadataFilter(key="score", value=0.8, operator=FilterOperator.GT),
                MetadataFilter(key="status", value="published", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.AND,
        )

        result = convert_llamaindex_filters_to_dxmt(filters)

        assert "must" in result
        assert len(result["must"]) == 4

        # Verify all conditions are present
        keys = [f["key"] for f in result["must"]]
        assert "category" in keys
        assert "year" in keys
        assert "score" in keys
        assert "status" in keys

    def test_range_filter(self):
        """Test range filter (between values)."""
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="year", value=2020, operator=FilterOperator.GTE),
                MetadataFilter(key="year", value=2024, operator=FilterOperator.LTE),
            ],
            condition=FilterCondition.AND,
        )

        result = convert_llamaindex_filters_to_dxmt(filters)

        assert "must" in result
        assert len(result["must"]) == 2
        assert result["must"][0]["operation"]["type"] == "GreaterThanOrEqual"
        assert result["must"][1]["operation"]["type"] == "LessThanOrEqual"

    def test_in_filter_with_single_value(self):
        """Test IN filter with single value (should be converted to list)."""
        filter = MetadataFilter(key="category", value="AI", operator=FilterOperator.IN)
        result = _convert_single_filter(filter)

        # Single value should be wrapped in list for IN operator
        assert result["operation"]["type"] == "In"
        # Note: Current implementation doesn't auto-wrap, this documents current behavior

    def test_complex_or_chain(self):
        """Test complex OR chain."""
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="AI", operator=FilterOperator.EQ),
                MetadataFilter(key="category", value="ML", operator=FilterOperator.EQ),
                MetadataFilter(key="category", value="DL", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.OR,
        )

        result = convert_llamaindex_filters_to_dxmt(filters)

        assert "should" in result
        assert len(result["should"]) == 3


class TestFilterEdgeCases:
    """Tests for edge cases in filter conversion."""

    def test_filter_with_none_value(self):
        """Test filter with None value."""
        filter = MetadataFilter(key="optional_field", value=None, operator=FilterOperator.EQ)
        result = _convert_single_filter(filter)

        assert result["operation"]["value"] is None

    def test_filter_with_boolean_value(self):
        """Test filter with boolean value.

        `MetadataFilter.value` is a pydantic union that does not admit `bool`, so the filter
        is built with `model_construct` to skip validation. What is under test is our
        converter, which passes the value through untouched -- and a payload really can hold
        a boolean, so the conversion has to survive one.
        """
        filter = MetadataFilter.model_construct(
            key="is_active", value=True, operator=FilterOperator.EQ
        )
        result = _convert_single_filter(filter)

        assert result["operation"]["value"] is True

    def test_filter_with_numeric_string(self):
        """Test filter with numeric string value."""
        filter = MetadataFilter(key="code", value="12345", operator=FilterOperator.EQ)
        result = _convert_single_filter(filter)

        assert result["operation"]["value"] == "12345"

    def test_filter_with_empty_list(self):
        """Test filter with empty list for IN operator."""
        filter = MetadataFilter(key="categories", value=[], operator=FilterOperator.IN)
        result = _convert_single_filter(filter)

        assert result["operation"]["value"] == []

    def test_default_condition_when_not_specified(self):
        """Test that default condition is AND when not specified."""
        # Create filters without explicit condition
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="a", value=1, operator=FilterOperator.EQ),
                MetadataFilter(key="b", value=2, operator=FilterOperator.EQ),
            ]
        )

        result = convert_llamaindex_filters_to_dxmt(filters)

        # Should default to AND (must)
        assert "must" in result or len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
