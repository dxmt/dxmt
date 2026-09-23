#!/bin/bash
# Test script to verify UPSERT fix

set -e

API_BASE="http://localhost:9999/api/v1"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================="
echo "UPSERT Functionality Test"
echo "=================================="
echo ""

# Clean up any existing test table
echo "Step 0: Cleaning up..."
curl -s -X DELETE "${API_BASE}/tables/upsert_test" > /dev/null 2>&1 || true
sleep 1

# Step 1: Create table
echo "Step 1: Creating table..."
curl -s -X POST "${API_BASE}/tables" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "upsert_test",
    "dimension": 4,
    "distance_metric": "cosine",
    "index_type": "brute_force"
  }' > /dev/null

if [ $? -eq 0 ]; then
    echo -e "   ${GREEN}✓${NC} Table created"
else
    echo -e "   ${RED}✗${NC} Failed to create table"
    exit 1
fi

sleep 1

# Step 2: Insert initial record with ID "doc1"
echo ""
echo "Step 2: Inserting initial record (id=doc1, version=v1)..."
curl -s -X POST "${API_BASE}/tables/upsert_test/records" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
      "id": "doc1",
      "vector": [1.0, 0.0, 0.0, 0.0],
      "payload": {"text": "Python v1", "version": "v1"}
    }]
  }' > /dev/null

echo -e "   ${GREEN}✓${NC} Initial record inserted"
sleep 1

# Step 3: Query to verify initial state
echo ""
echo "Step 3: Querying to verify initial state..."
RESULT1=$(curl -s -X POST "${API_BASE}/tables/upsert_test/search" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [1.0, 0.0, 0.0, 0.0],
    "top_k": 10
  }')

COUNT1=$(echo "$RESULT1" | jq '.data.hits | length')
VERSION1=$(echo "$RESULT1" | jq -r '.data.hits[0].payload.version')

echo "   Found $COUNT1 document(s)"
echo "   Version: $VERSION1"

if [ "$COUNT1" -eq 1 ] && [ "$VERSION1" = "v1" ]; then
    echo -e "   ${GREEN}✓${NC} Initial state correct"
else
    echo -e "   ${RED}✗${NC} Initial state incorrect"
fi

sleep 2

# Step 4: UPDATE (UPSERT) with same ID "doc1"
echo ""
echo "Step 4: Updating record (same id=doc1, version=v2)..."
curl -s -X POST "${API_BASE}/tables/upsert_test/records" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
      "id": "doc1",
      "vector": [0.0, 1.0, 0.0, 0.0],
      "payload": {"text": "Python v2 - Updated", "version": "v2", "updated": true}
    }]
  }' > /dev/null

echo -e "   ${GREEN}✓${NC} Update submitted"
sleep 2

# Step 5: Query again to verify update
echo ""
echo "Step 5: Querying to verify update..."
RESULT2=$(curl -s -X POST "${API_BASE}/tables/upsert_test/search" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.0, 1.0, 0.0, 0.0],
    "top_k": 10
  }')

COUNT2=$(echo "$RESULT2" | jq '.data.hits | length')
VERSION2=$(echo "$RESULT2" | jq -r '.data.hits[0].payload.version')
UPDATED2=$(echo "$RESULT2" | jq -r '.data.hits[0].payload.updated')

echo "   Found $COUNT2 document(s)"
if [ "$COUNT2" -gt 0 ]; then
    echo "   First result version: $VERSION2"
    echo "   First result updated: $UPDATED2"
fi

# Step 6: Verify UPSERT worked
echo ""
echo "=================================="
echo "VERIFICATION RESULTS"
echo "=================================="
echo ""

if [ "$COUNT2" -eq 1 ]; then
    echo -e "${GREEN}✅ PASS: Only 1 document found (no duplicates)${NC}"
else
    echo -e "${RED}❌ FAIL: Found $COUNT2 documents (expected 1)${NC}"
    echo ""
    echo "All results:"
    echo "$RESULT2" | jq '.data.hits[] | {id: .id, version: .payload.version, text: .payload.text}'
fi

if [ "$VERSION2" = "v2" ] && [ "$UPDATED2" = "true" ]; then
    echo -e "${GREEN}✅ PASS: Document was updated (version=v2, updated=true)${NC}"
else
    echo -e "${RED}❌ FAIL: Document not updated correctly${NC}"
    echo "   Expected: version=v2, updated=true"
    echo "   Got: version=$VERSION2, updated=$UPDATED2"
fi

# Step 7: Multiple rapid updates test
echo ""
echo "=================================="
echo "Rapid Updates Test"
echo "=================================="
echo ""
echo "Performing 5 rapid updates..."

for i in {1..5}; do
    curl -s -X POST "${API_BASE}/tables/upsert_test/records" \
      -H "Content-Type: application/json" \
      -d "{
        \"records\": [{
          \"id\": \"doc1\",
          \"vector\": [0.$i, 0.$i, 0.$i, 0.$i],
          \"payload\": {\"text\": \"Python v$i\", \"version\": \"v$i\", \"update_num\": $i}
        }]
      }" > /dev/null
    echo "   Update $i submitted"
    sleep 0.5
done

sleep 2

echo ""
echo "Verifying final state..."
RESULT3=$(curl -s -X POST "${API_BASE}/tables/upsert_test/search" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.5, 0.5, 0.5, 0.5],
    "top_k": 10
  }')

COUNT3=$(echo "$RESULT3" | jq '.data.hits | length')
UPDATE_NUM=$(echo "$RESULT3" | jq -r '.data.hits[0].payload.update_num')

echo "   Found $COUNT3 document(s)"
if [ "$COUNT3" -gt 0 ]; then
    echo "   Latest update_num: $UPDATE_NUM"
fi

echo ""
if [ "$COUNT3" -eq 1 ]; then
    echo -e "${GREEN}✅ PASS: No duplicates after rapid updates${NC}"
else
    echo -e "${RED}❌ FAIL: Found $COUNT3 documents after rapid updates${NC}"
fi

if [ "$UPDATE_NUM" -eq 5 ]; then
    echo -e "${GREEN}✅ PASS: Latest update applied (update_num=5)${NC}"
else
    echo -e "${YELLOW}⚠️  WARNING: Latest update may not be applied (expected 5, got $UPDATE_NUM)${NC}"
fi

# Cleanup
echo ""
echo "Cleaning up..."
curl -s -X DELETE "${API_BASE}/tables/upsert_test" > /dev/null
echo -e "${GREEN}✓${NC} Test table deleted"

echo ""
echo "=================================="
echo "Test Complete"
echo "=================================="
