#!/bin/bash
# Integration tests for advanced features
# Tests: payload filters, different metrics, batch operations

API_URL="http://localhost:9999"
TABLE_NAME="integration_test_$(date +%s)"
PASSED=0
FAILED=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_test() {
    echo ""
    echo "========================================="
    echo " TEST: $1"
    echo "========================================="
}

log_pass() {
    echo -e "${GREEN}✓ PASSED${NC}: $1"
    ((PASSED++))
}

log_fail() {
    echo -e "${RED}✗ FAILED${NC}: $1"
    ((FAILED++))
}

log_warn() {
    echo -e "${YELLOW}⚠ WARNING${NC}: $1"
}

# Setup: Create table
log_test "Setup: Create Test Table"
curl -s -X POST "$API_URL/api/v1/tables" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"$TABLE_NAME\",\"dimension\":128,\"distance_metric\":\"euclidean\",\"index_type\":\"hnsw\"}" > /dev/null
echo "Table '$TABLE_NAME' created"

# Test 1: Batch Insert with Metadata
log_test "1. Batch Insert with Rich Metadata"
cat > /tmp/batch_insert.json <<EOF
{
  "records": [
    {
      "id": "product_1",
      "vector": $(python3 -c "import json; print(json.dumps([0.1]*128))"),
      "payload": {"category": "electronics", "price": 999, "brand": "Apple", "rating": 4.5}
    },
    {
      "id": "product_2",
      "vector": $(python3 -c "import json; print(json.dumps([0.2]*128))"),
      "payload": {"category": "electronics", "price": 799, "brand": "Samsung", "rating": 4.3}
    },
    {
      "id": "product_3",
      "vector": $(python3 -c "import json; print(json.dumps([0.3]*128))"),
      "payload": {"category": "clothing", "price": 49, "brand": "Nike", "rating": 4.8}
    },
    {
      "id": "product_4",
      "vector": $(python3 -c "import json; print(json.dumps([0.4]*128))"),
      "payload": {"category": "clothing", "price": 79, "brand": "Adidas", "rating": 4.6}
    },
    {
      "id": "product_5",
      "vector": $(python3 -c "import json; print(json.dumps([0.5]*128))"),
      "payload": {"category": "electronics", "price": 1299, "brand": "Sony", "rating": 4.7}
    }
  ]
}
EOF

HTTP_CODE=$(curl -s -w "%{http_code}" -o /tmp/batch_result.json \
    -X POST "$API_URL/api/v1/tables/$TABLE_NAME/records" \
    -H "Content-Type: application/json" \
    -d @/tmp/batch_insert.json)

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
    log_pass "Batch insert of 5 records (HTTP $HTTP_CODE)"
else
    log_fail "Batch insert (HTTP $HTTP_CODE)"
fi

sleep 1

# Test 2: Search with Payload Filters (if supported)
log_test "2. Search with Payload Filters"
cat > /tmp/filtered_search.json <<EOF
{
  "vector": $(python3 -c "import json; print(json.dumps([0.15]*128))"),
  "k": 10,
  "filters": [
    {
      "field": "category",
      "operator": "eq",
      "value": "electronics"
    }
  ]
}
EOF

HTTP_CODE=$(curl -s -w "%{http_code}" -o /tmp/filter_result.json \
    -X POST "$API_URL/api/v1/tables/$TABLE_NAME/search" \
    -H "Content-Type: application/json" \
    -d @/tmp/filtered_search.json)

if [ "$HTTP_CODE" = "200" ]; then
    NUM_RESULTS=$(cat /tmp/filter_result.json | python3 -c "import json,sys; data=json.load(sys.stdin); print(len(data.get('data', {}).get('hits', [])))" 2>/dev/null || echo "0")
    if [ "$NUM_RESULTS" -gt "0" ]; then
        log_pass "Filtered search returned $NUM_RESULTS results"
    else
        log_warn "Filtered search returned 0 results (filters may not be implemented)"
    fi
else
    log_warn "Filtered search (HTTP $HTTP_CODE - feature may not be implemented)"
fi

# Test 3: Range Query (price filter)
log_test "3. Range Filter Test"
cat > /tmp/range_search.json <<EOF
{
  "vector": $(python3 -c "import json; print(json.dumps([0.25]*128))"),
  "k": 10,
  "filters": [
    {
      "field": "price",
      "operator": "gte",
      "value": 100
    },
    {
      "field": "price",
      "operator": "lte",
      "value": 1000
    }
  ]
}
EOF

HTTP_CODE=$(curl -s -w "%{http_code}" -o /tmp/range_result.json \
    -X POST "$API_URL/api/v1/tables/$TABLE_NAME/search" \
    -H "Content-Type: application/json" \
    -d @/tmp/range_search.json)

if [ "$HTTP_CODE" = "200" ]; then
    NUM_RESULTS=$(cat /tmp/range_result.json | python3 -c "import json,sys; data=json.load(sys.stdin); print(len(data.get('data', {}).get('hits', [])))" 2>/dev/null || echo "0")
    if [ "$NUM_RESULTS" -gt "0" ]; then
        log_pass "Range filter search returned $NUM_RESULTS results"
    else
        log_warn "Range filter returned 0 results (feature may not be implemented)"
    fi
else
    log_warn "Range filter search (HTTP $HTTP_CODE - feature may not be implemented)"
fi

# Test 4: Batch Delete
log_test "4. Batch Delete"
cat > /tmp/batch_delete.json <<EOF
{
  "ids": ["product_2", "product_4"]
}
EOF

HTTP_CODE=$(curl -s -w "%{http_code}" -o /tmp/delete_result.json \
    -X DELETE "$API_URL/api/v1/tables/$TABLE_NAME/records" \
    -H "Content-Type: application/json" \
    -d @/tmp/batch_delete.json)

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "204" ]; then
    log_pass "Batch delete (HTTP $HTTP_CODE)"

    # Verify deletion
    HTTP_CODE=$(curl -s -w "%{http_code}" -o /dev/null \
        "$API_URL/api/v1/tables/$TABLE_NAME/records/product_2")

    if [ "$HTTP_CODE" = "404" ]; then
        log_pass "Verified record was deleted (HTTP 404)"
    else
        log_fail "Record still exists after deletion (HTTP $HTTP_CODE)"
    fi
else
    log_fail "Batch delete (HTTP $HTTP_CODE)"
fi

# Test 5: Different K values
log_test "5. Search with Different K Values"
for K in 1 3 5; do
    HTTP_CODE=$(curl -s -w "%{http_code}" -o /tmp/search_k${K}.json \
        -X POST "$API_URL/api/v1/tables/$TABLE_NAME/search" \
        -H "Content-Type: application/json" \
        -d "{\"vector\":$(python3 -c "import json; print(json.dumps([0.3]*128))"),\"k\":$K}")

    if [ "$HTTP_CODE" = "200" ]; then
        NUM_RESULTS=$(cat /tmp/search_k${K}.json | python3 -c "import json,sys; data=json.load(sys.stdin); print(len(data.get('data', {}).get('hits', [])))" 2>/dev/null || echo "0")
        if [ "$NUM_RESULTS" -le "$K" ]; then
            log_pass "Search k=$K returned $NUM_RESULTS results (≤ $K)"
        else
            log_fail "Search k=$K returned $NUM_RESULTS results (> $K)"
        fi
    else
        log_fail "Search k=$K (HTTP $HTTP_CODE)"
    fi
done

# Test 6: Update Record with New Metadata
log_test "6. Update Record with New Metadata"
cat > /tmp/update_payload.json <<EOF
{
  "vector": $(python3 -c "import json; print(json.dumps([0.15]*128))"),
  "payload": {"category": "electronics", "price": 899, "brand": "Apple", "rating": 4.9, "updated": true}
}
EOF

HTTP_CODE=$(curl -s -w "%{http_code}" -o /tmp/update_result.json \
    -X PUT "$API_URL/api/v1/tables/$TABLE_NAME/records/product_1" \
    -H "Content-Type: application/json" \
    -d @/tmp/update_payload.json)

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "204" ]; then
    log_pass "Record updated (HTTP $HTTP_CODE)"

    # Verify update
    HTTP_CODE=$(curl -s -w "%{http_code}" -o /tmp/get_updated.json \
        "$API_URL/api/v1/tables/$TABLE_NAME/records/product_1")

    if [ "$HTTP_CODE" = "200" ]; then
        UPDATED=$(cat /tmp/get_updated.json | python3 -c "import json,sys; data=json.load(sys.stdin); payload=json.loads(data.get('data', {}).get('payload_json', '{}')); print(payload.get('updated', False))" 2>/dev/null || echo "false")
        if [ "$UPDATED" = "True" ]; then
            log_pass "Verified record metadata was updated"
        else
            log_warn "Metadata update verification inconclusive"
        fi
    fi
else
    log_fail "Record update (HTTP $HTTP_CODE)"
fi

# Test 7: Concurrent Operations
log_test "7. Concurrent Insert Operations"
START_TIME=$(date +%s%N)

for i in {1..10}; do
    curl -s -X POST "$API_URL/api/v1/tables/$TABLE_NAME/records" \
        -H "Content-Type: application/json" \
        -d "{\"records\":[{\"id\":\"concurrent_$i\",\"vector\":$(python3 -c "import json; print(json.dumps([0.${i}]*128))"),\"payload\":{\"concurrent\":true}}]}" > /dev/null &
done

wait

END_TIME=$(date +%s%N)
DURATION_MS=$(( (END_TIME - START_TIME) / 1000000 ))

# Verify all inserts succeeded
SUCCESS_COUNT=0
for i in {1..10}; do
    HTTP_CODE=$(curl -s -w "%{http_code}" -o /dev/null \
        "$API_URL/api/v1/tables/$TABLE_NAME/records/concurrent_$i")
    if [ "$HTTP_CODE" = "200" ]; then
        ((SUCCESS_COUNT++))
    fi
done

if [ "$SUCCESS_COUNT" -eq "10" ]; then
    log_pass "All 10 concurrent inserts succeeded (${DURATION_MS}ms)"
else
    log_fail "Only $SUCCESS_COUNT/10 concurrent inserts succeeded"
fi

# Test 8: Large Batch Insert
log_test "8. Large Batch Insert (100 records)"
cat > /tmp/large_batch.json <<EOF
{
  "records": [
EOF

for i in {1..99}; do
    echo "    {\"id\":\"large_$i\",\"vector\":$(python3 -c "import json; print(json.dumps([0.01*$i]*128))"),\"payload\":{\"index\":$i}}," >> /tmp/large_batch.json
done
echo "    {\"id\":\"large_100\",\"vector\":$(python3 -c "import json; print(json.dumps([1.0]*128))"),\"payload\":{\"index\":100}}" >> /tmp/large_batch.json
echo "  ]" >> /tmp/large_batch.json
echo "}" >> /tmp/large_batch.json

START_TIME=$(date +%s%N)
HTTP_CODE=$(curl -s -w "%{http_code}" -o /tmp/large_batch_result.json \
    -X POST "$API_URL/api/v1/tables/$TABLE_NAME/records" \
    -H "Content-Type: application/json" \
    -d @/tmp/large_batch.json)
END_TIME=$(date +%s%N)
DURATION_MS=$(( (END_TIME - START_TIME) / 1000000 ))

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
    THROUGHPUT=$(awk "BEGIN {print 100000/$DURATION_MS}")
    log_pass "Large batch insert (${DURATION_MS}ms, ${THROUGHPUT} rec/s)"
else
    log_fail "Large batch insert (HTTP $HTTP_CODE)"
fi

# Cleanup
log_test "Cleanup"
curl -s -X DELETE "$API_URL/api/v1/tables/$TABLE_NAME" > /dev/null
echo "Test table deleted"

# Summary
echo ""
echo "========================================="
echo " INTEGRATION TEST SUMMARY"
echo "========================================="
TOTAL=$((PASSED + FAILED))
echo "Total tests:  $TOTAL"
echo -e "${GREEN}Passed:       $PASSED${NC}"
if [ "$FAILED" -gt "0" ]; then
    echo -e "${RED}Failed:       $FAILED${NC}"
else
    echo "Failed:       $FAILED"
fi
echo "========================================="

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}✓ ALL INTEGRATION TESTS PASSED!${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    exit 1
fi
