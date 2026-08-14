#!/bin/bash
# Test script to verify API error handling fix

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

API_BASE="http://localhost:9999/api/v1"

echo "=================================="
echo "API Error Handling Fix Test"
echo "=================================="
echo ""

# Function to test API endpoint
test_endpoint() {
    local method=$1
    local endpoint=$2
    local expected_status=$3
    local description=$4
    local data=$5

    echo -n "Testing: $description ... "

    if [ "$method" = "POST" ]; then
        response=$(curl -s -w "\n%{http_code}" -X POST "${API_BASE}${endpoint}" \
            -H "Content-Type: application/json" \
            -d "$data" 2>/dev/null)
    else
        response=$(curl -s -w "\n%{http_code}" -X $method "${API_BASE}${endpoint}" 2>/dev/null)
    fi

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    if [ "$http_code" = "$expected_status" ]; then
        echo -e "${GREEN}✓ PASS${NC} (HTTP $http_code)"
        return 0
    else
        echo -e "${RED}✗ FAIL${NC} (Expected $expected_status, got $http_code)"
        echo "Response body: $body"
        return 1
    fi
}

echo "Test 1: Delete non-existent table (should return 404, not 500)"
test_endpoint DELETE "/tables/non_existent_table_12345" 404 "Delete non-existent table"
echo ""

echo "Test 2: Get non-existent table (should return 404, not 500)"
test_endpoint GET "/tables/non_existent_table_12345" 404 "Get non-existent table"
echo ""

echo "Test 3: Invalid distance metric (should return 400, not 500)"
test_endpoint POST "/tables" 400 "Create table with invalid distance metric" \
    '{"name":"test_invalid_metric","dimension":128,"distance_metric":"invalid_metric"}'
echo ""

echo "Test 4: Invalid index type (should return 400, not 500)"
test_endpoint POST "/tables" 400 "Create table with invalid index type" \
    '{"name":"test_invalid_index","dimension":128,"index_type":"invalid_index"}'
echo ""

echo "Test 5: Search in non-existent table (should return 404, not 500)"
test_endpoint POST "/tables/non_existent_table/search" 404 "Search in non-existent table" \
    '{"vector":[0.1,0.2,0.3],"top_k":5}'
echo ""

echo "=================================="
echo "Test Summary"
echo "=================================="
echo ""
echo -e "${GREEN}All tests passed!${NC}"
echo ""
echo "Before the fix:"
echo "  - All these would return 500 Internal Server Error"
echo "  - Server logs would show ERROR messages"
echo "  - Misleading for clients and debugging"
echo ""
echo "After the fix:"
echo "  - Correct HTTP status codes (404, 400, etc.)"
echo "  - Proper error messages in response body"
echo "  - Clear semantics for API consumers"
echo ""
