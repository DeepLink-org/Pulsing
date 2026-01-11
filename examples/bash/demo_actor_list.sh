#!/usr/bin/env bash
# 演示 pulsing actor list 命令
# 使用简单的 HTTP API 查询，不加入 gossip 集群

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/python:$PYTHONPATH"

# 完全禁用 Rust 日志
export RUST_LOG=off

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "======================================================================"
echo "  Pulsing Actor List - 演示"
echo "======================================================================"
echo ""

# 检查 pyenv
if ! command -v pyenv &> /dev/null; then
    echo "错误: 需要 pyenv"
    exit 1
fi

PYTHON="pyenv exec python"

# 清理可能残留的进程
echo -e "${YELLOW}清理残留进程...${NC}"
pkill -f "pulsing_server" 2>/dev/null || true
sleep 1

# 使用随机端口避免冲突
PORT=$((19000 + RANDOM % 1000))

# 创建一个临时服务端脚本
SERVER_SCRIPT=$(mktemp /tmp/pulsing_server_XXXXXX.py)

cat > "$SERVER_SCRIPT" << EOF
import asyncio
import os
import sys

# 禁用所有日志
os.environ["RUST_LOG"] = "off"

from pulsing.actor import init, remote, get_system


@remote
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count


@remote
class Calculator:
    def add(self, a, b):
        return a + b


async def main():
    await init(addr="127.0.0.1:${PORT}")
    system = get_system()

    await Counter.remote(system, name="counter-1")
    await Counter.remote(system, name="counter-2")
    await Calculator.remote(system, name="calculator")

    # Signal ready
    print("READY", flush=True)

    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
EOF

echo -e "${GREEN}1. 启动 Actor System (127.0.0.1:${PORT})${NC}"

# 启动服务端（后台运行，完全静默）
$PYTHON "$SERVER_SCRIPT" > /dev/null 2>&1 &
SERVER_PID=$!

# 等待服务就绪
echo "   等待服务启动..."
sleep 3

echo ""
echo -e "${GREEN}2. 测试连接单个 endpoint (HTTP API)${NC}"
echo "   命令: pulsing actor list --endpoint 127.0.0.1:${PORT}"
echo ""

$PYTHON -m pulsing.cli actor list --endpoint 127.0.0.1:${PORT}

echo ""
echo -e "${GREEN}3. 显示所有 actors (包括内部)${NC}"
echo "   命令: pulsing actor list --endpoint 127.0.0.1:${PORT} --all_actors True"
echo ""

$PYTHON -m pulsing.cli actor list --endpoint 127.0.0.1:${PORT} --all_actors True

echo ""
echo -e "${GREEN}4. JSON 格式输出${NC}"
echo "   命令: pulsing actor list --endpoint 127.0.0.1:${PORT} --json True"
echo ""

$PYTHON -m pulsing.cli actor list --endpoint 127.0.0.1:${PORT} --json True

echo ""
echo -e "${GREEN}5. 使用 --seeds 查询集群${NC}"
echo "   命令: pulsing actor list --seeds 127.0.0.1:${PORT}"
echo ""

$PYTHON -m pulsing.cli actor list --seeds 127.0.0.1:${PORT}

# 清理
echo ""
echo -e "${GREEN}清理...${NC}"
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
rm -f "$SERVER_SCRIPT"

echo ""
echo "======================================================================"
echo "  演示完成"
echo "======================================================================"
echo ""
echo "用法总结:"
echo ""
echo -e "  ${BLUE}# 查询单个 actor system${NC}"
echo "  pulsing actor list --endpoint 127.0.0.1:8000"
echo ""
echo -e "  ${BLUE}# 查询整个集群${NC}"
echo "  pulsing actor list --seeds 127.0.0.1:8000,127.0.0.1:8001"
echo ""
echo -e "  ${BLUE}# 显示所有 actors (包括内部)${NC}"
echo "  pulsing actor list --endpoint 127.0.0.1:8000 --all_actors True"
echo ""
echo -e "  ${BLUE}# JSON 格式输出${NC}"
echo "  pulsing actor list --endpoint 127.0.0.1:8000 --json True"
echo ""
echo -e "${GREEN}✓ 使用简单 HTTP API，不加入 gossip 集群${NC}"
echo -e "${GREEN}✓ 支持显示完整 Python 元信息: 类名、模块、代码路径、Actor ID${NC}"
echo ""
