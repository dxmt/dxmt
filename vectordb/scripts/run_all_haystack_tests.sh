#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║            运行所有 DXMT Haystack 集成测试                           ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

total_tests=0
passed_tests=0
failed_tests=0

# Test 1: Basic BM25 Test
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "测试 1: BM25 检索测试（test_haystack_bm25.py）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
total_tests=$((total_tests+1))
if python3 test_haystack_bm25.py 2>&1 | grep -q "✓ BM25 retrieval test completed successfully"; then
    echo "✅ PASSED"
    passed_tests=$((passed_tests+1))
else
    echo "❌ FAILED"
    failed_tests=$((failed_tests+1))
fi
echo ""

# Test 2: Unit Tests
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "测试 2: 单元测试（test_dxmt_haystack.py）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
total_tests=$((total_tests+1))
if python3 examples/test_dxmt_haystack.py 2>&1 | grep -q "✅ All tests passed"; then
    echo "✅ PASSED"
    passed_tests=$((passed_tests+1))
else
    echo "❌ FAILED"
    failed_tests=$((failed_tests+1))
fi
echo ""

# Test 3: Verification Test
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "测试 3: Bug 修复验证（verify_haystack_fix.py）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
total_tests=$((total_tests+1))
if python3 examples/verify_haystack_fix.py 2>&1 | grep -q "✅ ALL TESTS PASSED"; then
    echo "✅ PASSED"
    passed_tests=$((passed_tests+1))
else
    echo "❌ FAILED"
    failed_tests=$((failed_tests+1))
fi
echo ""

# Test 4: Integration Examples (only Example 3)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "测试 4: 集成示例（dxmt_haystack_rag.py - Example 3）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
total_tests=$((total_tests+1))
if python3 examples/dxmt_haystack_rag.py 2>&1 | grep -q "Found 3 documents:"; then
    echo "✅ PASSED"
    passed_tests=$((passed_tests+1))
else
    echo "❌ FAILED"
    failed_tests=$((failed_tests+1))
fi
echo ""

# Test 5: Protocol Compliance
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "测试 5: Haystack 协议合规性（test_haystack_embedding_filter.py）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
total_tests=$((total_tests+1))
if python3 test_haystack_embedding_filter.py 2>&1 | grep -q "✅ ALL TESTS PASSED"; then
    echo "✅ PASSED"
    passed_tests=$((passed_tests+1))
else
    echo "❌ FAILED"
    failed_tests=$((failed_tests+1))
fi
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                          测试结果总结                                ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "总测试数: $total_tests"
echo "通过: $passed_tests ✅"
echo "失败: $failed_tests ❌"
echo ""

if [ $failed_tests -eq 0 ]; then
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                   🎉 所有测试通过！🎉                                ║"
    echo "║                 DXMT Haystack 集成已就绪                            ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    exit 0
else
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                   ❌ 有测试失败                                      ║"
    echo "║              请检查上面的输出了解详情                                ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    exit 1
fi
