"""Unit tests for filter conversion utilities."""

import pytest
from dxmt import Field
from dxmt.filters import convert_simple_filter_to_dxmt, _map_operator


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
        result = convert_simple_filter_to_dxmt({"category": "AI"})["must"]
        assert len(result) == 1
        assert result[0] == {
            "key": "category",
            "operation": {"type": "Equal", "value": "AI"}
        }

    def test_parse_single_operator(self):
        """Test parsing condition with single operator."""
        result = convert_simple_filter_to_dxmt({"year": {"$gte": 2022}})["must"]
        assert len(result) == 1
        assert result[0] == {
            "key": "year",
            "operation": {"type": "GreaterThanOrEqual", "value": 2022}
        }

    def test_parse_multiple_operators(self):
        """Test parsing condition with multiple operators on same field."""
        result = convert_simple_filter_to_dxmt({"year": {"$gte": 2020, "$lte": 2023}})["must"]
        assert len(result) == 2
        # Order may vary, so check both conditions exist
        keys = [r["key"] for r in result]
        types = [r["operation"]["type"] for r in result]
        assert all(k == "year" for k in keys)
        assert "GreaterThanOrEqual" in types
        assert "LessThanOrEqual" in types

    def test_parse_multiple_fields(self):
        """Test parsing multiple field conditions."""
        result = convert_simple_filter_to_dxmt({"year": 2022, "rating": {"$gte": 8.0}})["must"]
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


def must(*conds):
    return {"must": list(conds)}


def cond(key, op_type, value):
    return {"key": key, "operation": {"type": op_type, "value": value}}


class TestTranslationIsExactOrRefused:
    """Each of these used to translate into a different query that ran and returned the wrong
    records. Now each translates exactly or raises."""

    def test_an_and_inside_an_or_is_refused_not_flattened(self):
        # Was {"should": [a, b, c]}: (a and b) or c read as a or b or c.
        with pytest.raises(ValueError, match="OR"):
            convert_simple_filter_to_dxmt({"$or": [{"a": 1, "b": 2}, {"c": 3}]})
        with pytest.raises(ValueError, match="OR"):
            convert_simple_filter_to_dxmt({"$or": [{"$and": [{"a": 1}, {"b": 2}]}, {"c": 3}]})

    def test_a_key_beside_a_logical_operator_is_kept(self):
        # Was {"must": [a]}: "lang" silently dropped.
        assert convert_simple_filter_to_dxmt({"$and": [{"a": 1}], "lang": "en"}) == must(
            cond("a", "Equal", 1), cond("lang", "Equal", "en")
        )
        assert convert_simple_filter_to_dxmt({"$or": [{"a": 1}, {"b": 2}], "lang": "en"}) == {
            "must": [cond("lang", "Equal", "en")],
            "should": [cond("a", "Equal", 1), cond("b", "Equal", 2)],
        }

    def test_the_native_shape_with_stray_keys_is_refused(self):
        with pytest.raises(ValueError, match="lang"):
            convert_simple_filter_to_dxmt({"must": [], "lang": "en"})

    def test_an_unknown_top_level_operator_is_refused(self):
        # Was a condition on a field named "$where".
        with pytest.raises(ValueError, match=r"\$where"):
            convert_simple_filter_to_dxmt({"$where": "x"})

    def test_nesting_that_has_a_flat_form_is_accepted(self):
        assert convert_simple_filter_to_dxmt(
            {"$and": [{"$and": [{"a": 1}]}, {"$or": [{"b": 2}, {"$or": [{"c": 3}]}]}]}
        ) == {"must": [cond("a", "Equal", 1)], "should": [cond("b", "Equal", 2), cond("c", "Equal", 3)]}

    def test_two_or_groups_are_refused(self):
        with pytest.raises(ValueError, match="two OR groups"):
            convert_simple_filter_to_dxmt({"$and": [{"$or": [{"a": 1}, {"b": 1}]}, {"$or": [{"c": 1}, {"d": 1}]}]})

    def test_nor_and_the_list_form_of_not(self):
        expected = {"must_not": [cond("a", "Equal", 1), cond("b", "Equal", 2)]}
        assert convert_simple_filter_to_dxmt({"$nor": [{"a": 1}, {"b": 2}]}) == expected
        assert convert_simple_filter_to_dxmt({"$not": [{"a": 1}, {"b": 2}]}) == expected
        # not (a and b) is not "none of them".
        with pytest.raises(ValueError, match="NOT"):
            convert_simple_filter_to_dxmt({"$nor": [{"a": 1, "b": 2}]})

    def test_values_the_server_would_refuse_are_refused_here(self):
        with pytest.raises(ValueError, match="any\n?\s*element|element"):
            convert_simple_filter_to_dxmt({"tags": ["a", "b"]})
        with pytest.raises(ValueError, match="object"):
            convert_simple_filter_to_dxmt({"meta": {"lang": "en"}})
        with pytest.raises(ValueError, match="list"):
            convert_simple_filter_to_dxmt({"tag": {"$in": "ai"}})
        with pytest.raises(ValueError, match="empty"):
            convert_simple_filter_to_dxmt({"tag": {}})
        # The server orders numbers only: a string bound would match nothing.
        with pytest.raises(ValueError, match="numbers only"):
            convert_simple_filter_to_dxmt({"date": {"$gte": "2024-01-01"}})
        with pytest.raises(ValueError, match="numbers only"):
            Field("flag") > True


class TestField:
    def test_every_operation(self):
        cases = [
            (Field("y") == 1, "Equal", 1),
            (Field("y") != 1, "NotEqual", 1),
            (Field("y") > 1, "GreaterThan", 1),
            (Field("y") >= 1, "GreaterThanOrEqual", 1),
            (Field("y") < 1, "LessThan", 1),
            (Field("y") <= 1, "LessThanOrEqual", 1),
            (Field("y").is_in([1, 2]), "In", [1, 2]),
            (Field("y").not_in((1, 2)), "NotIn", [1, 2]),
            (Field("y").contains("a"), "Contains", "a"),
            (Field("y").starts_with("a"), "StartsWith", "a"),
            (Field("y").ends_with("a"), "EndsWith", "a"),
        ]
        for built, op_type, value in cases:
            assert built.to_dict() == must(cond("y", op_type, value))

    def test_combinations(self):
        f = (Field("lang") == "en") & Field("year").between(2020, 2023) & ~Field("tags").is_in(["spam"])
        assert f.to_dict() == {
            "must": [
                cond("lang", "Equal", "en"),
                cond("year", "GreaterThanOrEqual", 2020),
                cond("year", "LessThanOrEqual", 2023),
            ],
            "must_not": [cond("tags", "In", ["spam"])],
        }
        g = ((Field("a") == 1) | (Field("b") == 2) | (Field("c") == 3)) & (Field("d") == 4)
        assert g.to_dict() == {
            "must": [cond("d", "Equal", 4)],
            "should": [cond("a", "Equal", 1), cond("b", "Equal", 2), cond("c", "Equal", 3)],
        }
        assert (~((Field("a") == 1) | (Field("b") == 2))).to_dict() == {
            "must_not": [cond("a", "Equal", 1), cond("b", "Equal", 2)]
        }

    def test_what_has_no_flat_form_is_refused(self):
        a, b, c, d = (Field(k) == 1 for k in "abcd")
        with pytest.raises(ValueError):
            (a & b) | c
        with pytest.raises(ValueError):
            (a | b) & (c | d)
        with pytest.raises(ValueError):
            ~(a & b)
        with pytest.raises(ValueError):
            ~~a

    def test_python_boolean_operators_are_refused(self):
        # Each of these calls bool() on a filter and would keep only one side.
        with pytest.raises(TypeError, match="between"):
            2020 <= Field("y") <= 2023
        with pytest.raises(TypeError):
            (Field("a") == 1) and (Field("b") == 2)
        with pytest.raises(TypeError):
            not (Field("a") == 1)

    def test_a_composite_value_is_refused_when_built(self):
        with pytest.raises(ValueError, match="element"):
            Field("tags") == ["a"]

    def test_accepted_wherever_a_dict_is(self):
        f = (Field("a") == 1) | (Field("b") == 2)
        assert convert_simple_filter_to_dxmt(f) == f.to_dict()
