#!/bin/bash
#
# 快速验证 DXMT VectorDBBench 集成
#

set -e

echo "======================================"
echo "DXMT VectorDBBench 验证脚本"
echo "======================================"
echo ""

# 检查命令是否可用
echo "1. 检查 vectordbbench 命令..."
if ! command -v vectordbbench &> /dev/null; then
    echo "❌ vectordbbench 命令未找到"
    echo "The DXMT client is not part of upstream VectorDBBench and is not in this repository -- it lives in a VectorDBBench fork carrying vectordb_bench/backend/clients/dxmt/. Set DXMT_VECTORDBBENCH to that checkout."
    echo "安装: pip install -e \"$DXMT_VECTORDBBENCH\""
    exit 1
fi
echo "✅ vectordbbench 命令可用"
echo ""

# 检查 DXMT 命令是否注册
echo "2. 检查 DXMT 命令..."
if vectordbbench --help 2>&1 | grep -q dxmtauto; then
    echo "✅ DXMT 命令已注册"
    echo ""
    echo "可用命令:"
    vectordbbench --help 2>&1 | grep dxmt
else
    echo "❌ DXMT 命令未注册"
    exit 1
fi
echo ""

# 检查 DXMT 服务器
echo "3. 检查 DXMT 服务器..."
if curl -s http://localhost:9999/health > /dev/null 2>&1; then
    echo "✅ DXMT 服务器正在运行"
else
    echo "⚠️  DXMT 服务器未运行"
    echo "启动命令: cargo run --release -- serve"
fi
echo ""

# 显示使用说明
echo "======================================"
echo "验证完成！"
echo "======================================"
echo ""
echo "快速开始:"
echo ""
echo "# 如果 DXMT 服务器未运行，先启动它:"
echo "cargo run --release -- serve"
echo ""
echo "# 运行快速测试 (约 5 分钟):"
echo "vectordbbench dxmtauto \\"
echo "  --host localhost \\"
echo "  --port 9999 \\"
echo "  --case-type Performance1536D50K \\"
echo "  --num-concurrency 1,5,10"
echo ""
echo "详细文档: VECTORDBBENCH_READY.md"
echo ""
