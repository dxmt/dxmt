#!/bin/bash
# Performance benchmark script
# Tests throughput (VPS), search QPS, and latency

API_URL="http://localhost:9999"
TABLE_NAME="perf_test_$(date +%s)"
NUM_VECTORS=1000
NUM_QUERIES=100
BATCH_SIZE=50

echo "========================================"
echo " VectorDB Performance Benchmark"
echo "========================================"
echo "Configuration:"
echo "  Vectors: $NUM_VECTORS"
echo "  Dimension: 128"
echo "  Batch size: $BATCH_SIZE"
echo "  Queries: $NUM_QUERIES"
echo ""

# Create table
echo "Creating table..."
curl -s -X POST "$API_URL/api/v1/tables" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"$TABLE_NAME\",\"dimension\":128,\"distance_metric\":\"euclidean\",\"index_type\":\"hnsw\"}" > /dev/null

if [ $? -ne 0 ]; then
    echo "Failed to create table"
    exit 1
fi
echo "✓ Table created"

# Function to generate a random vector
gen_vector() {
    echo -n "["
    for i in $(seq 1 127); do
        echo -n "0.$(($RANDOM % 10)),"
    done
    echo -n "0.$(($RANDOM % 10))]"
}

# Generate insertion payload
echo "Preparing test data..."
cat > /tmp/perf_vectors.json <<EOF
{
  "records": [
EOF

for i in $(seq 1 $((NUM_VECTORS - 1))); do
    echo "    {\"id\":\"vec_$i\",\"vector\":$(gen_vector),\"payload\":{\"index\":$i}}," >> /tmp/perf_vectors.json
done
echo "    {\"id\":\"vec_$NUM_VECTORS\",\"vector\":$(gen_vector),\"payload\":{\"index\":$NUM_VECTORS}}" >> /tmp/perf_vectors.json
echo "  ]" >> /tmp/perf_vectors.json
echo "}" >> /tmp/perf_vectors.json

echo "✓ Test data prepared"

# Benchmark 1: Insertion throughput
echo ""
echo "========================================"
echo " BENCHMARK 1: Insertion Throughput"
echo "========================================"

START_TIME=$(date +%s%N)
RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/insert_result.json \
    -X POST "$API_URL/api/v1/tables/$TABLE_NAME/records" \
    -H "Content-Type: application/json" \
    -d @/tmp/perf_vectors.json)
END_TIME=$(date +%s%N)
HTTP_CODE="${RESPONSE: -3}"

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
    DURATION_MS=$(( (END_TIME - START_TIME) / 1000000 ))
    DURATION_S=$(awk "BEGIN {print $DURATION_MS/1000}")
    THROUGHPUT=$(awk "BEGIN {print $NUM_VECTORS/$DURATION_S}")

    echo "✓ Inserted $NUM_VECTORS vectors"
    echo "  Duration: ${DURATION_MS}ms (${DURATION_S}s)"
    echo "  Throughput: ${THROUGHPUT} vectors/sec"
else
    echo "✗ Insertion failed (HTTP $HTTP_CODE)"
    cat /tmp/insert_result.json
    curl -s -X DELETE "$API_URL/api/v1/tables/$TABLE_NAME" > /dev/null
    exit 1
fi

# Wait for indexing
echo ""
echo "Waiting 3 seconds for indexing..."
sleep 3

# Benchmark 2: Search QPS and Latency
echo ""
echo "========================================"
echo " BENCHMARK 2: Search QPS & Latency"
echo "========================================"

# Generate query vector
QUERY_VEC=$(gen_vector)

# Warmup queries
echo "Running warmup queries..."
for i in {1..10}; do
    curl -s -X POST "$API_URL/api/v1/tables/$TABLE_NAME/search" \
        -H "Content-Type: application/json" \
        -d "{\"vector\":$QUERY_VEC,\"k\":10}" > /dev/null
done

# Run benchmark queries
echo "Running $NUM_QUERIES benchmark queries..."
> /tmp/latencies.txt

START_TIME=$(date +%s%N)
for i in $(seq 1 $NUM_QUERIES); do
    QUERY_START=$(date +%s%N)
    curl -s -X POST "$API_URL/api/v1/tables/$TABLE_NAME/search" \
        -H "Content-Type: application/json" \
        -d "{\"vector\":$QUERY_VEC,\"k\":10}" > /dev/null
    QUERY_END=$(date +%s%N)
    LATENCY_MS=$(( (QUERY_END - QUERY_START) / 1000000 ))
    echo $LATENCY_MS >> /tmp/latencies.txt
done
END_TIME=$(date +%s%N)

# Calculate metrics
TOTAL_DURATION_MS=$(( (END_TIME - START_TIME) / 1000000 ))
TOTAL_DURATION_S=$(awk "BEGIN {print $TOTAL_DURATION_MS/1000}")
QPS=$(awk "BEGIN {print $NUM_QUERIES/$TOTAL_DURATION_S}")

# Calculate latency statistics
sort -n /tmp/latencies.txt > /tmp/latencies_sorted.txt
AVG_LATENCY=$(awk '{s+=$1} END {print s/NR}' /tmp/latencies_sorted.txt)
P50_LATENCY=$(awk 'NR=='$((NUM_QUERIES/2))' {print $1}' /tmp/latencies_sorted.txt)
P95_LATENCY=$(awk 'NR=='$((NUM_QUERIES*95/100))' {print $1}' /tmp/latencies_sorted.txt)
P99_LATENCY=$(awk 'NR=='$((NUM_QUERIES*99/100))' {print $1}' /tmp/latencies_sorted.txt)
MIN_LATENCY=$(head -1 /tmp/latencies_sorted.txt)
MAX_LATENCY=$(tail -1 /tmp/latencies_sorted.txt)

echo "✓ Completed $NUM_QUERIES queries"
echo ""
echo "Throughput:"
echo "  QPS: $QPS queries/sec"
echo ""
echo "Latency (ms):"
echo "  Min: ${MIN_LATENCY}ms"
echo "  Avg: ${AVG_LATENCY}ms"
echo "  P50: ${P50_LATENCY}ms"
echo "  P95: ${P95_LATENCY}ms"
echo "  P99: ${P99_LATENCY}ms"
echo "  Max: ${MAX_LATENCY}ms"

# Benchmark 3: Concurrent Search
echo ""
echo "========================================"
echo " BENCHMARK 3: Concurrent Search"
echo "========================================"

CONCURRENT_QUERIES=50
NUM_THREADS=10

echo "Running $CONCURRENT_QUERIES concurrent queries ($NUM_THREADS threads)..."

# Create search script
cat > /tmp/search_worker.sh <<'WORKER_EOF'
#!/bin/bash
API_URL="$1"
TABLE_NAME="$2"
QUERY_VEC="$3"
NUM_QUERIES="$4"

for i in $(seq 1 $NUM_QUERIES); do
    curl -s -X POST "$API_URL/api/v1/tables/$TABLE_NAME/search" \
        -H "Content-Type: application/json" \
        -d "{\"vector\":$QUERY_VEC,\"k\":10}" > /dev/null
done
WORKER_EOF

chmod +x /tmp/search_worker.sh

START_TIME=$(date +%s%N)

# Launch workers
QUERIES_PER_THREAD=$((CONCURRENT_QUERIES / NUM_THREADS))
for i in $(seq 1 $NUM_THREADS); do
    /tmp/search_worker.sh "$API_URL" "$TABLE_NAME" "$QUERY_VEC" "$QUERIES_PER_THREAD" &
done

# Wait for all workers
wait

END_TIME=$(date +%s%N)

DURATION_MS=$(( (END_TIME - START_TIME) / 1000000 ))
DURATION_S=$(awk "BEGIN {print $DURATION_MS/1000}")
CONCURRENT_QPS=$(awk "BEGIN {print $CONCURRENT_QUERIES/$DURATION_S}")

echo "✓ Completed $CONCURRENT_QUERIES concurrent queries"
echo "  Duration: ${DURATION_MS}ms (${DURATION_S}s)"
echo "  Concurrent QPS: $CONCURRENT_QPS queries/sec"

# Cleanup
echo ""
echo "Cleaning up..."
curl -s -X DELETE "$API_URL/api/v1/tables/$TABLE_NAME" > /dev/null
echo "✓ Test table deleted"

# Summary
echo ""
echo "========================================"
echo " PERFORMANCE SUMMARY"
echo "========================================"
echo "Insertion:"
echo "  Throughput: ${THROUGHPUT} vectors/sec"
echo ""
echo "Search (Sequential):"
echo "  QPS: $QPS queries/sec"
echo "  P95 Latency: ${P95_LATENCY}ms"
echo ""
echo "Search (Concurrent):"
echo "  QPS: $CONCURRENT_QPS queries/sec"
echo "  Threads: $NUM_THREADS"
echo "========================================"

# Save results
cat > /tmp/perf_results.json <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "configuration": {
    "num_vectors": $NUM_VECTORS,
    "dimension": 128,
    "num_queries": $NUM_QUERIES,
    "concurrent_threads": $NUM_THREADS
  },
  "results": {
    "insertion": {
      "throughput_vps": $THROUGHPUT,
      "duration_ms": $DURATION_MS
    },
    "search_sequential": {
      "qps": $QPS,
      "avg_latency_ms": $AVG_LATENCY,
      "p50_latency_ms": $P50_LATENCY,
      "p95_latency_ms": $P95_LATENCY,
      "p99_latency_ms": $P99_LATENCY
    },
    "search_concurrent": {
      "qps": $CONCURRENT_QPS,
      "num_queries": $CONCURRENT_QUERIES,
      "num_threads": $NUM_THREADS
    }
  }
}
EOF

echo ""
echo "Results saved to /tmp/perf_results.json"
