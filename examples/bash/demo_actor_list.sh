#!/usr/bin/env bash
# 演示 pulsing actor list 命令

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/python:$PYTHONPATH"

# 设置日志级别为 ERROR，减少刷屏
export RUST_LOG=error

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
pkill -f "127.0.0.1:19001" 2>/dev/null || true
sleep 1

# 使用随机端口避免冲突
PORT=$((19000 + RANDOM % 1000))

# 创建一个临时服务端脚本
SERVER_SCRIPT=$(mktemp /tmp/pulsing_server_XXXXXX.py)

cat > "$SERVER_SCRIPT" << EOF
import asyncio
import os
os.environ["RUST_LOG"] = "error"

from pulsing.actor import init, remote, get_system


@remote
class Counter:
    """A simple counter actor"""
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count += 1
        return self.count
    
    def get(self):
        return self.count


@remote
class Calculator:
    """A calculator actor"""
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b


async def main():
    # Start actor system
    await init(addr="127.0.0.1:${PORT}")
    system = get_system()
    print(f"Actor system started: {system.addr}", flush=True)
    
    # Create some actors
    counter1 = await Counter.remote(system, name="counter-1")
    counter2 = await Counter.remote(system, name="counter-2")
    calc = await Calculator.remote(system, name="calculator")
    
    print("Created actors: counter-1, counter-2, calculator", flush=True)
    print("READY", flush=True)
    
    # Keep running
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
EOF

echo -e "${GREEN}1. 启动 Actor System (127.0.0.1:${PORT})${NC}"
echo ""

# 启动服务端（后台运行）
$PYTHON "$SERVER_SCRIPT" &
SERVER_PID=$!

# 等待服务就绪
echo "   等待服务启动..."
for i in {1..10}; do
    if $PYTHON -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.connect(('127.0.0.1', ${PORT}))
    s.close()
    exit(0)
except:
    exit(1)
" 2>/dev/null; then
        echo "   服务已就绪"
        break
    fi
    sleep 0.5
done
sleep 1

echo ""
echo -e "${GREEN}2. 测试连接单个 endpoint${NC}"
echo "   命令: pulsing actor list --endpoint 127.0.0.1:${PORT}"
echo ""

$PYTHON -m pulsing.cli actor list --endpoint 127.0.0.1:${PORT} 2>/dev/null

echo ""
echo -e "${GREEN}3. 显示所有 actors (包括内部)${NC}"
echo "   命令: pulsing actor list --endpoint 127.0.0.1:${PORT} --all_actors True"
echo ""

$PYTHON -m pulsing.cli actor list --endpoint 127.0.0.1:${PORT} --all_actors True 2>/dev/null

echo ""
echo -e "${GREEN}4. JSON 格式输出${NC}"
echo "   命令: pulsing actor list --endpoint 127.0.0.1:${PORT} --json True"
echo ""

$PYTHON -m pulsing.cli actor list --endpoint 127.0.0.1:${PORT} --json True 2>/dev/null

echo ""
echo -e "${GREEN}5. 使用 --seeds 查询集群${NC}"
echo "   命令: pulsing actor list --seeds 127.0.0.1:${PORT}"
echo ""

$PYTHON -m pulsing.cli actor list --seeds 127.0.0.1:${PORT} 2>/dev/null

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
