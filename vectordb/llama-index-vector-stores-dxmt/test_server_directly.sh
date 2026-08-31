#!/bin/bash
# Test DXMT server filtering directly

TABLE="direct_test_$(date +%s)"

echo "=== Testing DXMT Server Filters Directly ==="
echo

# Create table
echo "1. Creating table: $TABLE"
curl -s -X POST "http://localhost:9999/tables" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "'"$TABLE"'",
    "dimension": 3,
    "distance_metric": "cosine"
  }' | jq -r '.status'
echo

# Insert records
echo "2. Inserting 3 records..."
curl -s -X POST "http://localhost:9999/tables/$TABLE/records/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "id": "doc1",
        "vector": [0.1, 0.2, 0.3],
        "payload": {"name": "Rust", "popularity": 85}
      },
      {
        "id": "doc2",
        "vector": [0.2, 0.3, 0.4],
        "payload": {"name": "Go", "popularity": 80}
      },
      {
        "id": "doc3",
        "vector": [0.3, 0.4, 0.5],
        "payload": {"name": "Python", "popularity": 70}
      }
    ]
  }' | jq -r '.status'
echo

# Search without filter
echo "3. Search without filter (should return 3):"
curl -s -X POST "http://localhost:9999/tables/$TABLE/search" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.2, 0.3, 0.4],
    "limit": 5
  }' | jq -r '.data.hits | length'
echo

# Search with filter: popularity >= 80
echo "4. Search with filter popularity >= 80 (should return 2):"
curl -s -X POST "http://localhost:9999/tables/$TABLE/search" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.2, 0.3, 0.4],
    "limit": 5,
    "filter": {
      "must": [
        {
          "key": "popularity",
          "operation": {"type": "GreaterThanOrEqual", "value": 80}
        }
      ]
    }
  }' | jq -r '.data.hits | length'

echo "  Full response:"
curl -s -X POST "http://localhost:9999/tables/$TABLE/search" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.2, 0.3, 0.4],
    "limit": 5,
    "filter": {
      "must": [
        {
          "key": "popularity",
          "operation": {"type": "GreaterThanOrEqual", "value": 80}
        }
      ]
    }
  }' | jq '.data.hits[] | {id: .id, name: .payload.name, popularity: .payload.popularity}'
echo

# Cleanup
echo "5. Cleaning up..."
curl -s -X DELETE "http://localhost:9999/tables/$TABLE" | jq -r '.status'
echo

echo "=== Test Complete ==="
