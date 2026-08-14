#!/bin/bash
# WAL 测试套件运行脚本

set -e

echo "================================"
echo "WAL Test Suite"
echo "================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 测试结果目录
RESULTS_DIR="test_results"
mkdir -p "$RESULTS_DIR"

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Test started at: $(date)"
echo ""

# 1. 性能基准测试
echo -e "${YELLOW}=== 1. Performance Benchmark Tests ===${NC}"
echo "Running WAL performance tests..."
echo ""

cargo test --test wal_performance_test -- --nocapture 2>&1 | tee "$RESULTS_DIR/performance_${TIMESTAMP}.log"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ Performance tests PASSED${NC}"
else
    echo -e "${RED}✗ Performance tests FAILED${NC}"
    exit 1
fi

echo ""
echo "---"
echo ""

# 2. 并发压力测试
echo -e "${YELLOW}=== 2. Concurrency Stress Tests ===${NC}"
echo "Running WAL concurrency tests..."
echo ""

cargo test --test wal_concurrency_test -- --nocapture --test-threads=1 2>&1 | tee "$RESULTS_DIR/concurrency_${TIMESTAMP}.log"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ Concurrency tests PASSED${NC}"
else
    echo -e "${RED}✗ Concurrency tests FAILED${NC}"
    exit 1
fi

echo ""
echo "---"
echo ""

# 3. 崩溃恢复测试
echo -e "${YELLOW}=== 3. Crash Recovery Tests ===${NC}"
echo "Running WAL recovery tests..."
echo ""

cargo test --test wal_recovery_test -- --nocapture 2>&1 | tee "$RESULTS_DIR/recovery_${TIMESTAMP}.log"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ Recovery tests PASSED${NC}"
else
    echo -e "${RED}✗ Recovery tests FAILED${NC}"
    exit 1
fi

echo ""
echo "---"
echo ""

# 4. 基本 WAL 单元测试
echo -e "${YELLOW}=== 4. WAL Unit Tests ===${NC}"
echo "Running WAL unit tests..."
echo ""

cargo test --lib wal -- --nocapture 2>&1 | tee "$RESULTS_DIR/unit_${TIMESTAMP}.log"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ Unit tests PASSED${NC}"
else
    echo -e "${RED}✗ Unit tests FAILED${NC}"
    exit 1
fi

echo ""
echo "================================"
echo -e "${GREEN}All WAL tests PASSED!${NC}"
echo "================================"
echo ""

echo "Test completed at: $(date)"
echo ""
echo "Test results saved to: $RESULTS_DIR/"
echo "  - performance_${TIMESTAMP}.log"
echo "  - concurrency_${TIMESTAMP}.log"
echo "  - recovery_${TIMESTAMP}.log"
echo "  - unit_${TIMESTAMP}.log"
echo ""

# 生成简单的汇总报告
echo "Generating summary report..."
cat > "$RESULTS_DIR/summary_${TIMESTAMP}.txt" << EOF
WAL Test Suite Summary
======================
Timestamp: $(date)

Test Results:
✓ Performance Benchmark Tests - PASSED
✓ Concurrency Stress Tests - PASSED
✓ Crash Recovery Tests - PASSED
✓ WAL Unit Tests - PASSED

All tests passed successfully!

Performance Highlights:
- Check performance_${TIMESTAMP}.log for detailed benchmarks
- Check concurrency_${TIMESTAMP}.log for stress test results
- Check recovery_${TIMESTAMP}.log for recovery performance

Next Steps:
1. Review performance metrics in the logs
2. Compare with baseline performance targets
3. Consider additional stress testing under production load
EOF

echo "Summary report saved to: $RESULTS_DIR/summary_${TIMESTAMP}.txt"
echo ""
echo "Done!"
