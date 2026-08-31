#!/bin/bash
# LMDB 持久化自动验证脚本

set -e

echo "=========================================="
echo "  LMDB 持久化自动验证"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
VECTORDB_BIN="./target/release/vectordb"
PORT=9999
PID_FILE="/tmp/vectordb_persist_test.pid"

# 检查 Python 脚本是否存在
if [ ! -f "verify_persistence.py" ]; then
    echo -e "${RED}错误: 未找到 verify_persistence.py${NC}"
    exit 1
fi

# 检查二进制文件是否存在
if [ ! -f "$VECTORDB_BIN" ]; then
    echo -e "${RED}错误: 未找到 $VECTORDB_BIN${NC}"
    echo "请先编译项目: cargo build --release --features storage-lmdb"
    exit 1
fi

echo -e "${YELLOW}步骤 1: 启动 DXMT 服务器 (使用 LMDB 模式)${NC}"
echo ""

# 停止现有的测试服务器
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "停止旧的测试服务器 (PID: $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
    fi
    rm -f "$PID_FILE"
fi

# 启动服务器
echo "启动 DXMT 服务器..."
$VECTORDB_BIN serve --host 127.0.0.1 --port $PORT > /tmp/vectordb_persist_test.log 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"
echo -e "${GREEN}✓ 服务器已启动 (PID: $SERVER_PID)${NC}"

# 等待服务器启动
echo "等待服务器就绪..."
for i in {1..30}; do
    if curl -s "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ 服务器就绪${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ 服务器启动超时${NC}"
        cat /tmp/vectordb_persist_test.log
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
    echo -n "."
done
echo ""

# 阶段 1: 插入数据
echo ""
echo -e "${YELLOW}步骤 2: 插入测试数据${NC}"
echo ""
python3 verify_persistence.py
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ 数据插入失败${NC}"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# 检查 LMDB 文件是否创建
echo ""
echo -e "${YELLOW}步骤 3: 检查 LMDB 文件${NC}"
echo ""
if [ -d "./data/lmdb" ]; then
    echo "LMDB 目录内容:"
    ls -lh ./data/lmdb/
    echo ""
    echo -e "${GREEN}✓ LMDB 文件已创建${NC}"
else
    echo -e "${RED}✗ LMDB 目录不存在${NC}"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# 停止服务器
echo ""
echo -e "${YELLOW}步骤 4: 重启服务器以验证持久化${NC}"
echo ""
echo "停止服务器 (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null || true
sleep 3

# 确认进程已停止
if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "强制停止服务器..."
    kill -9 $SERVER_PID 2>/dev/null || true
    sleep 1
fi
echo -e "${GREEN}✓ 服务器已停止${NC}"

# 重新启动服务器
echo ""
echo "重新启动服务器..."
$VECTORDB_BIN serve --host 127.0.0.1 --port $PORT > /tmp/vectordb_persist_test.log 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"
echo -e "${GREEN}✓ 服务器已重启 (PID: $SERVER_PID)${NC}"

# 等待服务器启动
echo "等待服务器就绪..."
for i in {1..30}; do
    if curl -s "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ 服务器就绪${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ 服务器启动超时${NC}"
        cat /tmp/vectordb_persist_test.log
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
    echo -n "."
done
echo ""

# 阶段 2: 验证持久化
echo ""
echo -e "${YELLOW}步骤 5: 验证数据持久化${NC}"
echo ""
python3 verify_persistence.py --phase2
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ 持久化验证失败${NC}"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# 清理
echo ""
echo -e "${YELLOW}步骤 6: 清理测试服务器${NC}"
echo ""
echo "停止测试服务器 (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null || true
rm -f "$PID_FILE"
echo -e "${GREEN}✓ 测试服务器已停止${NC}"

# 总结
echo ""
echo "=========================================="
echo -e "${GREEN}  ✓ LMDB 持久化验证成功!${NC}"
echo "=========================================="
echo ""
echo "总结:"
echo "  1. ✓ 服务器成功启动 (LMDB 模式)"
echo "  2. ✓ 数据成功插入"
echo "  3. ✓ LMDB 文件已创建"
echo "  4. ✓ 服务器成功重启"
echo "  5. ✓ 重启后数据完整恢复"
echo ""
echo "LMDB 数据目录: ./data/lmdb"
echo ""
echo "如需正式使用 LMDB 持久化，请使用以下命令启动服务器:"
echo "  ./target/release/vectordb serve"
echo ""
echo "注意: .env 文件已配置为使用 LMDB 模式"
echo ""
