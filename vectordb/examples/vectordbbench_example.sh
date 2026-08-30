#!/bin/bash
#
# Example script for running VectorDBBench with DXMT
#
# This script demonstrates various ways to benchmark DXMT using VectorDBBench.
# Make sure DXMT server is running before executing this script.
#

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}DXMT VectorDBBench Example${NC}"
echo -e "${BLUE}================================${NC}\n"

# Check if DXMT server is running
echo -e "${YELLOW}Checking DXMT server...${NC}"
if curl -s http://localhost:9999/health > /dev/null; then
    echo -e "${GREEN}✓ DXMT server is running${NC}\n"
else
    echo -e "${YELLOW}⚠ DXMT server not detected. Please start it with:${NC}"
    echo -e "  cargo run --release -- serve --host 0.0.0.0 --port 9999\n"
    exit 1
fi

# Check if vectordbbench is installed
if ! command -v vectordbbench &> /dev/null; then
    echo -e "${YELLOW}⚠ vectordbbench not found. Please install it with:${NC}"
    echo -e "  cd reference/VectorDBBench && pip install -e .\n"
    exit 1
fi

echo -e "${GREEN}✓ vectordbbench is installed${NC}\n"

# Example 1: Quick test with small dataset
echo -e "${BLUE}Example 1: Quick Test (50K vectors, 1536 dim)${NC}"
echo -e "This runs a fast test with a small dataset.\n"
cat << EOF
vectordbbench dxmtauto \\
  --host localhost \\
  --port 9999 \\
  --db-label "dxmt-quick-test" \\
  --case-type Performance1536D50K \\
  --num-concurrency 1,5,10 \\
  --k 100
EOF
echo ""
read -p "Press Enter to run this example (or Ctrl+C to skip)..."
vectordbbench dxmtauto \
  --host localhost \
  --port 9999 \
  --db-label "dxmt-quick-test" \
  --case-type Performance1536D50K \
  --num-concurrency 1,5,10 \
  --k 100
echo -e "${GREEN}✓ Example 1 completed${NC}\n"

# Example 2: HNSW with medium dataset
echo -e "${BLUE}Example 2: HNSW Index (1M vectors, 768 dim)${NC}"
echo -e "This tests HNSW index with standard parameters.\n"
cat << EOF
vectordbbench dxmthnsw \\
  --host localhost \\
  --port 9999 \\
  --db-label "dxmt-hnsw-1m" \\
  --case-type Performance768D1M \\
  --m 16 \\
  --ef-construction 200 \\
  --ef-search 100 \\
  --num-concurrency 1,10,20 \\
  --k 100
EOF
echo ""
read -p "Press Enter to run this example (or Ctrl+C to skip)..."
vectordbbench dxmthnsw \
  --host localhost \
  --port 9999 \
  --db-label "dxmt-hnsw-1m" \
  --case-type Performance768D1M \
  --m 16 \
  --ef-construction 200 \
  --ef-search 100 \
  --num-concurrency 1,10,20 \
  --k 100
echo -e "${GREEN}✓ Example 2 completed${NC}\n"

# Example 3: HNSW-PQ for memory efficiency
echo -e "${BLUE}Example 3: HNSW-PQ Index (Memory Efficient)${NC}"
echo -e "This tests HNSW with Product Quantization for reduced memory usage.\n"
cat << EOF
vectordbbench dxmthnswpq \\
  --host localhost \\
  --port 9999 \\
  --db-label "dxmt-hnsw-pq" \\
  --case-type Performance768D1M \\
  --m 16 \\
  --ef-construction 200 \\
  --ef-search 100 \\
  --pq-m 8 \\
  --pq-nbits 8 \\
  --num-concurrency 1,10,20
EOF
echo ""
read -p "Press Enter to run this example (or Ctrl+C to skip)..."
vectordbbench dxmthnswpq \
  --host localhost \
  --port 9999 \
  --db-label "dxmt-hnsw-pq" \
  --case-type Performance768D1M \
  --m 16 \
  --ef-construction 200 \
  --ef-search 100 \
  --pq-m 8 \
  --pq-nbits 8 \
  --num-concurrency 1,10,20
echo -e "${GREEN}✓ Example 3 completed${NC}\n"

# Example 4: Batch testing with config file
echo -e "${BLUE}Example 4: Batch Testing with Config File${NC}"
echo -e "This demonstrates running multiple tests from a YAML config.\n"

cat > /tmp/dxmt_batch_test.yaml << EOF
dxmtauto:
  - db_label: dxmt-auto-50k
    host: localhost
    port: 9999
    case_type: Performance1536D50K
    num_concurrency: "1,5,10"
    k: 100

  - db_label: dxmt-auto-500k
    host: localhost
    port: 9999
    case_type: Performance1536D500K
    num_concurrency: "1,10,20"
    k: 100

dxmthnsw:
  - db_label: dxmt-hnsw-optimized
    host: localhost
    port: 9999
    case_type: Performance768D1M
    m: 32
    ef_construction: 256
    ef_search: 128
    num_concurrency: "1,10,20,30"
EOF

echo "Config file created at /tmp/dxmt_batch_test.yaml:"
cat /tmp/dxmt_batch_test.yaml
echo ""
read -p "Press Enter to run batch tests (or Ctrl+C to skip)..."
vectordbbench batchcli --batch-config-file /tmp/dxmt_batch_test.yaml
echo -e "${GREEN}✓ Example 4 completed${NC}\n"

# Example 5: Comparison test
echo -e "${BLUE}Example 5: Parameter Tuning Comparison${NC}"
echo -e "This compares different HNSW parameters to find optimal settings.\n"

cat > /tmp/dxmt_tuning.yaml << EOF
dxmthnsw:
  - db_label: dxmt-ef100
    host: localhost
    port: 9999
    case_type: Performance1536D50K
    m: 16
    ef_construction: 200
    ef_search: 100

  - db_label: dxmt-ef200
    host: localhost
    port: 9999
    case_type: Performance1536D50K
    m: 16
    ef_construction: 200
    ef_search: 200

  - db_label: dxmt-ef400
    host: localhost
    port: 9999
    case_type: Performance1536D50K
    m: 16
    ef_construction: 200
    ef_search: 400
EOF

echo "Config file created at /tmp/dxmt_tuning.yaml:"
cat /tmp/dxmt_tuning.yaml
echo ""
read -p "Press Enter to run parameter tuning tests (or Ctrl+C to skip)..."
vectordbbench batchcli --batch-config-file /tmp/dxmt_tuning.yaml
echo -e "${GREEN}✓ Example 5 completed${NC}\n"

# Summary
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Examples Completed!${NC}"
echo -e "${BLUE}================================${NC}\n"

echo -e "${YELLOW}View results:${NC}"
echo "1. Start web UI: init_bench"
echo "2. Navigate to http://localhost:8501"
echo "3. Click 'Results' to see benchmark results"
echo ""

echo -e "${YELLOW}Next steps:${NC}"
echo "1. Run larger datasets (10M, 100M vectors)"
echo "2. Compare results across configurations"
echo "3. Submit results to VectorDBBench leaderboard"
echo ""

echo -e "${GREEN}All examples completed successfully!${NC}"
