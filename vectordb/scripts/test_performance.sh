#!/bin/bash
# Quick performance test for single-node optimizations

set -e

echo "==================================="
echo "DXMT VectorDB Performance Test"
echo "==================================="
echo ""

# Build in release mode
echo "Building in release mode..."
cargo build --release 2>&1 | tail -1

# Start the server in background
echo "Starting VectorDB server..."
./target/release/vectordb &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server failed to start"
    exit 1
fi

echo "Server started (PID: $SERVER_PID)"
echo ""

# Function to cleanup
cleanup() {
    echo ""
    echo "Stopping server..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
}

trap cleanup EXIT

# Run performance tests
echo "Running performance tests..."
echo ""

# Test 1: Create table
echo "Test 1: Creating table..."
curl -s -X POST http://localhost:9999/tables \
  -H "Content-Type: application/json" \
  -d '{
    "name": "perf_test",
    "dimension": 128,
    "distance_metric": "cosine"
  }' | jq .

echo ""

# Test 2: Batch insert (small)
echo "Test 2: Small batch insert (100 vectors)..."
python3 <<'PYTHON'
import requests
import time
import random

def generate_vector(dim=128):
    return [random.random() for _ in range(dim)]

def test_batch_insert(batch_size):
    records = []
    for i in range(batch_size):
        records.append({
            "id": f"vec_{i}",
            "vector": generate_vector(128),
            "payload": {"index": i}
        })

    start = time.time()
    response = requests.post(
        "http://localhost:9999/tables/perf_test/records",
        json=records
    )
    elapsed = time.time() - start

    if response.status_code == 200:
        throughput = batch_size / elapsed
        print(f"✓ Inserted {batch_size} vectors in {elapsed:.3f}s ({throughput:.0f} vec/s)")
    else:
        print(f"✗ Failed: {response.text}")

# Small batch
test_batch_insert(100)
PYTHON

echo ""

# Test 3: Batch insert (medium)
echo "Test 3: Medium batch insert (1000 vectors)..."
python3 <<'PYTHON'
import requests
import time
import random

def generate_vector(dim=128):
    return [random.random() for _ in range(dim)]

records = []
for i in range(100, 1100):
    records.append({
        "id": f"vec_{i}",
        "vector": generate_vector(128),
        "payload": {"index": i}
    })

start = time.time()
response = requests.post(
    "http://localhost:9999/tables/perf_test/records",
    json=records
)
elapsed = time.time() - start

if response.status_code == 200:
    throughput = 1000 / elapsed
    print(f"✓ Inserted 1000 vectors in {elapsed:.3f}s ({throughput:.0f} vec/s)")
else:
    print(f"✗ Failed: {response.text}")
PYTHON

echo ""

# Test 4: Search performance
echo "Test 4: Search performance (100 queries)..."
python3 <<'PYTHON'
import requests
import time
import random

def generate_vector(dim=128):
    return [random.random() for _ in range(dim)]

# Run 100 search queries
total_time = 0
num_queries = 100

for i in range(num_queries):
    query_vector = generate_vector(128)

    start = time.time()
    response = requests.post(
        "http://localhost:9999/tables/perf_test/search",
        json={
            "vector": query_vector,
            "top_k": 10
        }
    )
    elapsed = time.time() - start
    total_time += elapsed

    if response.status_code != 200:
        print(f"✗ Query {i} failed: {response.text}")
        break

if response.status_code == 200:
    avg_latency = (total_time / num_queries) * 1000
    qps = num_queries / total_time
    print(f"✓ Completed {num_queries} queries")
    print(f"  Average latency: {avg_latency:.2f}ms")
    print(f"  Throughput: {qps:.0f} QPS")
PYTHON

echo ""
echo "==================================="
echo "Performance test completed!"
echo "==================================="

# Cleanup is handled by trap
