#!/bin/bash
set -x
BASE="http://localhost:9998"

curl -s "$BASE/status" | jq .data.status

curl -s -X POST "$BASE/tables" \
  -H "Content-Type: application/json" \
  -d '{"name":"tbl","dimension":3}' | jq .

curl -s "$BASE/tables" | jq .

curl -s -X POST "$BASE/tables/tbl/records" \
  -H "Content-Type: application/json" \
  -d '{"id":"r1","vector":[1,0,0]}' | jq .

curl -s "$BASE/tables/tbl/records/r1" | jq .

curl -s -X POST "$BASE/tables/tbl/search" \
  -H "Content-Type: application/json" \
  -d '{"vector":[1,0,0],"top_k":5}' | jq .
