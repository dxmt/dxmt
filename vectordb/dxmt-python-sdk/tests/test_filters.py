"""Unit tests for filter conversion utilities."""

import pytest
from dxmt.filters import convert_simple_filter_to_dxmt, _map_operator, _parse_condition


class TestOperatorMapping:
    """Test MongoDB operator to DXMT operation type mapping."""

    def test_map_supported_operators(self):
        """Test mapping of supported operators."""
        assert _map_operator("$eq") == "Equal"
        assert _map_operator("$ne") == "NotEqual"
        assert _map_operator("$gt") == "GreaterThan"
        assert _map_operator("$gte") == "GreaterThanOrEqual"
        assert _map_operator("$lt") == "LessThan"
        assert _map_operator("$lte") == "LessThanOrEqual"
        assert _map_operator("$in") == "In"
        assert _map_operator("$nin") == "NotIn"
        assert _map_operator("$contains") == "Contains"
        assert _map_operator("$startswith") == "StartsWith"
        assert _map_operator("$endswith") == "EndsWith"

    def test_map_unsupported_operator(self):
        """Test that unsupported operators raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported operator"):
            _map_operator("$invalid")


class TestParseCondition:
    """Test condition parsing logic."""

    def test_parse_simple_equality(self):
        """Test parsing simple equality condition."""
        result = _parse_condition({"category": "AI"})
        assert len(result) == 1
        assert result[0] == {
            "key": "category",
            "operation": {"type": "Equal", "value": "AI"}
        }

    def test_parse_single_operator(self):
        """Test parsing condition with single operator."""
        result = _parse_condition({"year": {"$gte": 2022}})
        assert len(result) == 1
        assert result[0] == {
            "key": "year",
            "operation": {"type": "GreaterThanOrEqual", "value": 2022}
        }

    def test_parse_multiple_operators(self):
        """Test parsing condition with multiple operators on same field."""
        result = _parse_condition({"year": {"$gte": 2020, "$lte": 2023}})
        assert len(result) == 2
        # Order may vary, so check both conditions exist
        keys = [r["key"] for r in result]
        types = [r["operation"]["type"] for r in result]
        assert all(k == "year" for k in keys)
        assert "GreaterThanOrEqual" in types
        assert "LessThanOrEqual" in types

    def test_parse_multiple_fields(self):
        """Test parsing multiple field conditions."""
        result = _parse_condition({"year": 2022, "rating": {"$gte": 8.0}})
        assert len(result) == 2
        # Check both conditions exist
        field_map = {r["key"]: r for r in result}
        assert "year" in field_map
        assert "rating" in field_map
        assert field_map["year"]["operation"]["type"] == "Equal"
        assert field_map["rating"]["operation"]["type"] == "GreaterThanOrEqual"


class TestConvertSimpleFilterToDXMT:
    """Test main filter conversion function."""

    def test_convert_none_filter(self):
        """Test that None input returns None."""
        assert convert_simple_filter_to_dxmt(None) is None

    def test_convert_empty_filter(self):
        """Test that empty dict returns None."""
        assert convert_simple_filter_to_dxmt({}) is None

    def test_convert_simple_equality_filter(self):
        """Test conversion of simple equality filter."""
        result = convert_simple_filter_to_dxmt({"category": "AI", "score": 0.8})
        assert "must" in result
        assert len(result["must"]) == 2
        # Check both conditions
        field_map = {c["key"]: c for c in result["must"]}
        assert field_map["category"]["operation"] == {"type": "Equal", "value": "AI"}
        assert field_map["score"]["operation"] == {"type": "Equal", "value": 0.8}

    def test_convert_operator_filter(self):
        """Test conversion of operator-based filter."""
        result = convert_simple_filter_to_dxmt({
            "year": {"$gte": 2022},
            "rating": {"$lt": 9.0}
        })
        assert "must" in result
        assert len(result["must"]) == 2
        field_map = {c["key"]: c for c in result["must"]}
        assert field_map["year"]["operation"]["type"] == "GreaterThanOrEqual"
        assert field_map["year"]["operation"]["value"] == 2022
        assert field_map["rating"]["operation"]["type"] == "LessThan"
        assert field_map["rating"]["operation"]["value"] == 9.0

    def test_convert_and_operator(self):
        """Test conversion of $and operator."""
        result = convert_simple_filter_to_dxmt({
            "$and": [
                {"year": {"$gte": 2022}},
                {"rating": {"$gte": 8.0}}
            ]
        })
        assert "must" in result
        assert len(result["must"]) == 2
        field_map = {c["key"]: c for c in result["must"]}
        assert "year" in field_map
        assert "rating" in field_map
        assert field_map["year"]["operation"]["value"] == 2022
        assert field_map["rating"]["operation"]["value"] == 8.0

    def test_convert_or_operator(self):
        """Test conversion of $or operator."""
        result = convert_simple_filter_to_dxmt({
            "$or": [
                {"category": "AI"},
                {"category": "ML"}
            ]
        })
        assert "should" in result
        assert len(result["should"]) == 2
        values = [c["operation"]["value"] for c in result["should"]]
        assert "AI" in values
        assert "ML" in values

    def test_convert_not_operator(self):
        """Test conversion of $not operator."""
        result = convert_simple_filter_to_dxmt({
            "$not": [
                {"status": "deleted"}
            ]
        })
        assert "must_not" in result
        assert len(result["must_not"]) == 1
        assert result["must_not"][0]["key"] == "status"
        assert result["must_not"][0]["operation"]["value"] == "deleted"

    def test_convert_dxmt_native_format_passthrough(self):
        """Test that DXMT native format is passed through unchanged."""
        dxmt_filter = {
            "must": [
                {"key": "category", "operation": {"type": "Equal", "value": "AI"}}
            ]
        }
        result = convert_simple_filter_to_dxmt(dxmt_filter)
        assert result == dxmt_filter

    def test_convert_complex_filter(self):
        """Test conversion of complex real-world filter."""
        result = convert_simple_filter_to_dxmt({
            "$and": [
                {"year": {"$gte": 2020, "$lte": 2023}},
                {"rating": {"$gte": 8.0}},
                {"category": {"$in": ["AI", "ML", "DL"]}}
            ]
        })
        assert "must" in result
        # Should have 4 conditions: year >= 2020, year <= 2023, rating >= 8.0, category in [...]
        assert len(result["must"]) == 4

        # Check all conditions exist
        keys = [c["key"] for c in result["must"]]
        assert keys.count("year") == 2
        assert keys.count("rating") == 1
        assert keys.count("category") == 1

    def test_convert_in_operator(self):
        """Test conversion of $in operator."""
        result = convert_simple_filter_to_dxmt({
            "category": {"$in": ["AI", "ML", "DL"]}
        })
        assert "must" in result
        assert len(result["must"]) == 1
        assert result["must"][0]["operation"]["type"] == "In"
        assert result["must"][0]["operation"]["value"] == ["AI", "ML", "DL"]

    def test_convert_string_operators(self):
        """Test conversion of string operators."""
        result = convert_simple_filter_to_dxmt({
            "$and": [
                {"title": {"$contains": "Space"}},
                {"genre": {"$startswith": "sci"}},
                {"tags": {"$endswith": "action"}}
            ]
        })
        assert "must" in result
        assert len(result["must"]) == 3
        field_map = {c["key"]: c for c in result["must"]}
        assert field_map["title"]["operation"]["type"] == "Contains"
        assert field_map["genre"]["operation"]["type"] == "StartsWith"
        assert field_map["tags"]["operation"]["type"] == "EndsWith"


class TestRealWorldScenarios:
    """Test real-world filter scenarios from examples."""

    def test_recommendation_system_filter(self):
        """Test the filter from recommendation_system.py Example 3."""
        # This is the filter that was failing
        result = convert_simple_filter_to_dxmt({
            "$and": [
                {"year": {"$gte": 2022}},
                {"rating": {"$gte": 8.0}},
            ]
        })

        assert result is not None
        assert "must" in result
        assert len(result["must"]) == 2

        # Verify both conditions
        field_map = {c["key"]: c for c in result["must"]}
        assert field_map["year"]["operation"]["type"] == "GreaterThanOrEqual"
        assert field_map["year"]["operation"]["value"] == 2022
        assert field_map["rating"]["operation"]["type"] == "GreaterThanOrEqual"
        assert field_map["rating"]["operation"]["value"] == 8.0

    def test_duplicate_detection_filter(self):
        """Test filter from duplicate_detection.py."""
        result = convert_simple_filter_to_dxmt({
            "category": "technology"
        })

        assert result is not None
        assert "must" in result
        assert len(result["must"]) == 1
        assert result["must"][0]["key"] == "category"
        assert result["must"][0]["operation"]["value"] == "technology"

    def test_filtered_search_range_filter(self):
        """Test range filter scenario."""
        result = convert_simple_filter_to_dxmt({
            "$and": [
                {"price": {"$gte": 100, "$lte": 500}},
                {"in_stock": True}
            ]
        })

        assert result is not None
        assert "must" in result
        # Should have 3 conditions: price >= 100, price <= 500, in_stock = true
        assert len(result["must"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestMatchesFilter:
    """Client-side evaluation of a filter in DXMT's own shape.

    This exists only for the paths that must list records and filter them here, because the
    listing endpoint takes no filter. **The server is the authority**; this is a second reading
    of the same conditions and therefore a second chance to drift, which is why every operator
    is pinned rather than the handful the current callers happen to use.
    """

    from dxmt.filters import matches_filter

    RECORD = {"payload": {"year": 2023, "title": "vector search", "tag": "ai"}}

    def check(self, op_type, key, value):
        from dxmt.filters import matches_filter

        return matches_filter(
            self.RECORD, {"must": [{"key": key, "operation": {"type": op_type, "value": value}}]}
        )

    def test_no_filter_matches_everything(self):
        from dxmt.filters import matches_filter

        assert matches_filter(self.RECORD, None)
        assert matches_filter(self.RECORD, {})

    def test_equality(self):
        assert self.check("Equal", "year", 2023)
        assert not self.check("Equal", "year", 2022)
        assert self.check("NotEqual", "year", 2022)

    def test_comparisons(self):
        assert self.check("GreaterThanOrEqual", "year", 2023)
        assert self.check("GreaterThan", "year", 2022)
        assert not self.check("GreaterThan", "year", 2023)
        assert self.check("LessThanOrEqual", "year", 2023)
        assert not self.check("LessThan", "year", 2023)

    def test_a_comparison_against_a_non_number_is_false_not_an_error(self):
        assert not self.check("GreaterThan", "title", 5)

    def test_membership(self):
        assert self.check("In", "tag", ["ai", "ml"])
        assert not self.check("In", "tag", ["ml"])
        assert self.check("NotIn", "tag", ["ml"])
        assert not self.check("NotIn", "tag", ["ai"])

    def test_string_tests(self):
        assert self.check("Contains", "title", "search")
        assert not self.check("Contains", "title", "audio")
        assert self.check("StartsWith", "title", "vector")
        assert self.check("EndsWith", "title", "search")
        # A string test against a number matches nothing, the same way the server reads it.
        assert not self.check("StartsWith", "year", "20")

    def test_a_missing_field_matches_nothing(self):
        assert not self.check("Equal", "absent", 1)
        assert not self.check("GreaterThan", "absent", 0)

    def test_must_not_and_should(self):
        from dxmt.filters import matches_filter

        year_is = {"key": "year", "operation": {"type": "Equal", "value": 2023}}
        tag_is_ml = {"key": "tag", "operation": {"type": "Equal", "value": "ml"}}

        assert not matches_filter(self.RECORD, {"must_not": [year_is]})
        assert matches_filter(self.RECORD, {"must_not": [tag_is_ml]})
        assert matches_filter(self.RECORD, {"should": [year_is, tag_is_ml]})
        assert not matches_filter(self.RECORD, {"should": [tag_is_ml]})

    def test_a_record_from_a_single_read_is_understood_too(self):
        """`GET /records/{id}` returns `payload_json` as a string, not `payload` as an object.
        A filter evaluator that only knew one shape would silently match nothing on records
        that came from the other endpoint."""
        from dxmt.filters import matches_filter

        record = {"payload_json": '{"year": 2023}'}
        assert matches_filter(
            record, {"must": [{"key": "year", "operation": {"type": "Equal", "value": 2023}}]}
        )
