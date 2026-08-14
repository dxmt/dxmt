#!/usr/bin/env python3
"""Debug script to investigate filter conversion issues"""

import sys
import json
sys.path.insert(0, '/localdisk/code/dxmt/vectordb/dxmt-python-sdk')

from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    FilterCondition,
)
from llama_index.vector_stores.dxmt.utils import convert_llamaindex_filters_to_dxmt

print("=" * 70)
print("Filter Conversion Debug")
print("=" * 70)

# Test 3 - FAILING: Simple equality filter
print("\n[TEST 3 - FAILING] difficulty == 'Beginner'")
filters_test3 = MetadataFilters(
    filters=[
        MetadataFilter(key="difficulty", value="Beginner", operator=FilterOperator.EQ)
    ]
)
dxmt_filter_3 = convert_llamaindex_filters_to_dxmt(filters_test3)
print(f"LlamaIndex filter: {filters_test3}")
print(f"DXMT filter: {json.dumps(dxmt_filter_3, indent=2)}")

# Test 4 - FAILING: Double condition AND
print("\n[TEST 4 - FAILING] year >= 2000 AND popularity >= 80")
filters_test4 = MetadataFilters(
    filters=[
        MetadataFilter(key="year", value=2000, operator=FilterOperator.GTE),
        MetadataFilter(key="popularity", value=80, operator=FilterOperator.GTE),
    ],
    condition=FilterCondition.AND,
)
dxmt_filter_4 = convert_llamaindex_filters_to_dxmt(filters_test4)
print(f"LlamaIndex filter: {filters_test4}")
print(f"DXMT filter: {json.dumps(dxmt_filter_4, indent=2)}")

# Test 6 - WORKING: Complex 3-condition AND
print("\n[TEST 6 - WORKING] category == 'Programming' AND year < 2000 AND popularity > 70")
filters_test6 = MetadataFilters(
    filters=[
        MetadataFilter(key="category", value="Programming", operator=FilterOperator.EQ),
        MetadataFilter(key="year", value=2000, operator=FilterOperator.LT),
        MetadataFilter(key="popularity", value=70, operator=FilterOperator.GT),
    ],
    condition=FilterCondition.AND,
)
dxmt_filter_6 = convert_llamaindex_filters_to_dxmt(filters_test6)
print(f"LlamaIndex filter: {filters_test6}")
print(f"DXMT filter: {json.dumps(dxmt_filter_6, indent=2)}")

# Test 7 - WORKING: OR condition with difficulty
print("\n[TEST 7 - WORKING] difficulty == 'Beginner' OR popularity >= 90")
filters_test7 = MetadataFilters(
    filters=[
        MetadataFilter(key="difficulty", value="Beginner", operator=FilterOperator.EQ),
        MetadataFilter(key="popularity", value=90, operator=FilterOperator.GTE),
    ],
    condition=FilterCondition.OR,
)
dxmt_filter_7 = convert_llamaindex_filters_to_dxmt(filters_test7)
print(f"LlamaIndex filter: {filters_test7}")
print(f"DXMT filter: {json.dumps(dxmt_filter_7, indent=2)}")

# Test 5 - FAILING: IN operator
print("\n[TEST 5 - FAILING] difficulty IN ['Intermediate', 'Advanced']")
filters_test5 = MetadataFilters(
    filters=[
        MetadataFilter(
            key="difficulty",
            value=["Intermediate", "Advanced"],
            operator=FilterOperator.IN
        )
    ]
)
dxmt_filter_5 = convert_llamaindex_filters_to_dxmt(filters_test5)
print(f"LlamaIndex filter: {filters_test5}")
print(f"DXMT filter: {json.dumps(dxmt_filter_5, indent=2)}")

print("\n" + "=" * 70)
print("Analysis:")
print("=" * 70)
print("\nLet's check if single filter returns dict vs list with 'must'")
print(f"Test 3 (single filter) type: {type(dxmt_filter_3)}, keys: {dxmt_filter_3.keys() if isinstance(dxmt_filter_3, dict) else 'N/A'}")
print(f"Test 4 (double AND) type: {type(dxmt_filter_4)}, keys: {dxmt_filter_4.keys() if isinstance(dxmt_filter_4, dict) else 'N/A'}")
print(f"Test 6 (triple AND) type: {type(dxmt_filter_6)}, keys: {dxmt_filter_6.keys() if isinstance(dxmt_filter_6, dict) else 'N/A'}")
print(f"Test 7 (double OR) type: {type(dxmt_filter_7)}, keys: {dxmt_filter_7.keys() if isinstance(dxmt_filter_7, dict) else 'N/A'}")
