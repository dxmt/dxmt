#!/bin/bash
# Quick functional test using curl
# Tests basic CRUD and search functionality

API_URL="http://localhost:9999"
TABLE_NAME="test_table_$(date +%s)"
PASSED=0
FAILED=0

# Colors
GREEN='\033[0.32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_test() {
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

# Test 1: Server Health
log_test "Server Health Check"
RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/health_response.json "$API_URL/status")
HTTP_CODE="${RESPONSE: -3}"

if [ "$HTTP_CODE" = "200" ]; then
    log_pass "Server is healthy (HTTP $HTTP_CODE)"
    cat /tmp/health_response.json | python3 -m json.tool 2>/dev/null || cat /tmp/health_response.json
else
    log_fail "Server health check (HTTP $HTTP_CODE)"
fi

# Test 2: Create Table
log_test "Create Table"
cat > /tmp/create_table.json <<EOF
{
  "name": "$TABLE_NAME",
  "dimension": 128,
  "distance_metric": "euclidean",
  "index_type": "hnsw"
}
EOF

RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/create_table_response.json \
    -X POST "$API_URL/api/v1/tables" \
    -H "Content-Type: application/json" \
    -d @/tmp/create_table.json)
HTTP_CODE="${RESPONSE: -3}"

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
    log_pass "Table created (HTTP $HTTP_CODE)"
else
    log_fail "Table creation (HTTP $HTTP_CODE)"
    cat /tmp/create_table_response.json
fi

# Test 3: List Tables
log_test "List Tables"
RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/list_tables.json "$API_URL/api/v1/tables")
HTTP_CODE="${RESPONSE: -3}"

if [ "$HTTP_CODE" = "200" ]; then
    log_pass "Listed tables (HTTP $HTTP_CODE)"
else
    log_fail "List tables (HTTP $HTTP_CODE)"
fi

# Test 4: Insert Records
log_test "Insert Records"
cat > /tmp/insert_records.json <<EOF
{
  "records": [
    {
      "id": "vec1",
      "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
      "payload": {"type": "test", "index": 1}
    },
    {
      "id": "vec2",
      "vector": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
      "payload": {"type": "test", "index": 2}
    },
    {
      "id": "vec3",
      "vector": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      "payload": {"type": "test", "index": 3}
    }
  ]
}
EOF

RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/insert_response.json \
    -X POST "$API_URL/api/v1/tables/$TABLE_NAME/records" \
    -H "Content-Type: application/json" \
    -d @/tmp/insert_records.json)
HTTP_CODE="${RESPONSE: -3}"

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
    log_pass "Records inserted (HTTP $HTTP_CODE)"
else
    log_fail "Record insertion (HTTP $HTTP_CODE)"
    cat /tmp/insert_response.json
fi

# Test 5: Get Single Record
log_test "Get Single Record"
RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/get_record.json \
    "$API_URL/api/v1/tables/$TABLE_NAME/records/vec1")
HTTP_CODE="${RESPONSE: -3}"

if [ "$HTTP_CODE" = "200" ]; then
    log_pass "Retrieved record (HTTP $HTTP_CODE)"
else
    log_fail "Get record (HTTP $HTTP_CODE)"
fi

# Wait for indexing
echo "Waiting 2 seconds for indexing..."
sleep 2

# Test 6: Vector Search
log_test "Vector Search"
cat > /tmp/search.json <<EOF
{
  "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
  "k": 3
}
EOF

START_TIME=$(date +%s%N)
RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/search_response.json \
    -X POST "$API_URL/api/v1/tables/$TABLE_NAME/search" \
    -H "Content-Type: application/json" \
    -d @/tmp/search.json)
END_TIME=$(date +%s%N)
HTTP_CODE="${RESPONSE: -3}"
LATENCY_MS=$(( (END_TIME - START_TIME) / 1000000 ))

if [ "$HTTP_CODE" = "200" ]; then
    log_pass "Search completed in ${LATENCY_MS}ms (HTTP $HTTP_CODE)"
    echo "Search results:"
    cat /tmp/search_response.json | python3 -m json.tool 2>/dev/null || cat /tmp/search_response.json
else
    log_fail "Search (HTTP $HTTP_CODE)"
    cat /tmp/search_response.json
fi

# Test 7: Update Record
log_test "Update Record"
cat > /tmp/update_record.json <<EOF
{
  "vector": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
  "payload": {"type": "updated", "index": 999}
}
EOF

RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/update_response.json \
    -X PUT "$API_URL/api/v1/tables/$TABLE_NAME/records/vec1" \
    -H "Content-Type: application/json" \
    -d @/tmp/update_record.json)
HTTP_CODE="${RESPONSE: -3}"

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "204" ]; then
    log_pass "Record updated (HTTP $HTTP_CODE)"
else
    log_fail "Record update (HTTP $HTTP_CODE)"
    cat /tmp/update_response.json
fi

# Test 8: Delete Record
log_test "Delete Record"
cat > /tmp/delete_records.json <<EOF
{
  "ids": ["vec2"]
}
EOF

RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/delete_response.json \
    -X DELETE "$API_URL/api/v1/tables/$TABLE_NAME/records" \
    -H "Content-Type: application/json" \
    -d @/tmp/delete_records.json)
HTTP_CODE="${RESPONSE: -3}"

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "204" ]; then
    log_pass "Record deleted (HTTP $HTTP_CODE)"
else
    log_fail "Record deletion (HTTP $HTTP_CODE)"
    cat /tmp/delete_response.json
fi

# Test 9: Delete Table
log_test "Delete Table"
RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/delete_table.json \
    -X DELETE "$API_URL/api/v1/tables/$TABLE_NAME")
HTTP_CODE="${RESPONSE: -3}"

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "204" ]; then
    log_pass "Table deleted (HTTP $HTTP_CODE)"
else
    log_fail "Table deletion (HTTP $HTTP_CODE)"
    cat /tmp/delete_table.json
fi

# Summary
echo ""
echo "========================================="
echo " TEST SUMMARY"
echo "========================================="
TOTAL=$((PASSED + FAILED))
echo -e "Total tests:  $TOTAL"
echo -e "${GREEN}Passed:       $PASSED${NC}"
if [ "$FAILED" -gt 0 ]; then
    echo -e "${RED}Failed:       $FAILED${NC}"
else
    echo -e "Failed:       $FAILED"
fi
echo "========================================="

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED!${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    exit 1
fi
