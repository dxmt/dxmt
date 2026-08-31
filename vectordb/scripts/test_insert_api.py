#!/usr/bin/env python3
import requests
import time
import random

BASE_URL = "http://localhost:9999/api/v1"

# Create table
response = requests.post(
    f"{BASE_URL}/tables",
    json={
        "name": "insert_test",
        "dimension": 128,
        "distance_metric": "cosine"
    }
)
print(f"Create table: {response.status_code}")

# Test single insert
single_record = {
    "id": "test1",
    "vector": [random.random() for _ in range(128)],
    "payload": {"test": 1}
}

response = requests.post(
    f"{BASE_URL}/tables/insert_test/records",
    json={"records": [single_record]}  # BatchInsertRequest format
)
print(f"Single insert: {response.status_code} - {response.text[:200]}")

# Test batch insert
batch_records = [
    {
        "id": f"batch_{i}",
        "vector": [random.random() for _ in range(128)],
        "payload": {"index": i}
    }
    for i in range(1000)
]

start = time.time()
response = requests.post(
    f"{BASE_URL}/tables/insert_test/records",
    json={"records": batch_records}  # BatchInsertRequest format
)
elapsed = time.time() - start

if response.status_code == 200:
    throughput = 1000 / elapsed
    print(f"✓ Batch insert: 1000 vectors in {elapsed:.3f}s ({throughput:.0f} VPS)")
else:
    print(f"✗ Batch insert failed: {response.status_code} - {response.text[:200]}")

# Test search
query_vector = [random.random() for _ in range(128)]
start = time.time()
response = requests.post(
    f"{BASE_URL}/tables/insert_test/search",
    json={
        "vector": query_vector,
        "top_k": 10
    }
)
elapsed = time.time() - start

if response.status_code == 200:
    print(f"✓ Search: {elapsed*1000:.2f}ms latency")
else:
    print(f"✗ Search failed: {response.status_code}")
