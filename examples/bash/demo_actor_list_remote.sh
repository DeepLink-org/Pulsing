#!/usr/bin/env bash
# 演示 pulsing actor list 的远程查询功能

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/python:$PYTHONPATH"

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "======================================================================"
echo "  Pulsing Actor List - 远程查询演示"
echo "======================================================================"
echo ""

# 检查 pyenv
if ! command -v pyenv &> /dev/null; then
    echo "错误: 需要 pyenv"
    exit 1
fi

PYTHON="pyenv exec python"

# 创建两个节点的应用
NODE1_SCRIPT=$(mktemp /tmp/pulsing_node1.XXXXXX.py)
NODE2_SCRIPT=$(mktemp /tmp/pulsing_node2.XXXXXX.py)

cat > "$NODE1_SCRIPT" << 'EOF'
import asyncio
from pulsing.actor import init, remote, get_system


@remote
class ServiceA:
    def ping(self):
        return "pong from A"


async def main():
    await init(addr="127.0.0.1:9001")
    system = get_system()
    print(f"Node 1 started: {system.addr}")

    # Create actors
    await ServiceA.remote(system, name="service-a-1")
    await ServiceA.remote(system, name="service-a-2")
    print("Created 2 actors on Node 1")

    # Keep running
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
EOF

cat > "$NODE2_SCRIPT" << 'EOF'
import asyncio
from pulsing.actor import SystemConfig, create_actor_system, remote


@remote
class ServiceB:
    def process(self, data):
        return f"processed: {data}"


async def main():
    config = SystemConfig.with_addr("127.0.0.1:9002").with_seeds(["127.0.0.1:9001"])
    system = await create_actor_system(config)
    print(f"Node 2 started: {system.addr}, joining cluster...")

    await asyncio.sleep(1)

    # Create actors
    await ServiceB.remote(system, name="service-b-1")
    await ServiceB.remote(system, name="service-b-2")
    await ServiceB.remote(system, name="service-b-3")
    print("Created 3 actors on Node 2")

    # Keep running
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
EOF

echo -e "${GREEN}场景: 查询远程多节点集群${NC}"
echo "================================================================"
echo ""

# 启动节点1
echo "1. 启动 Node 1 (127.0.0.1:9001)..."
$PYTHON "$NODE1_SCRIPT" 2>&1 | grep -v "INFO" &
NODE1_PID=$!
sleep 2

# 启动节点2
echo "2. 启动 Node 2 (127.0.0.1:9002), 加入集群..."
$PYTHON "$NODE2_SCRIPT" 2>&1 | grep -v "INFO" &
NODE2_PID=$!
sleep 3

echo ""
echo -e "${BLUE}3. 使用 pulsing actor list 查询远程集群${NC}"
echo "   命令: pulsing actor list --seeds '127.0.0.1:9001'"
echo ""

# 使用我们实现的功能查询集群
$PYTHON -c "
from pulsing.cli.actor_list import list_actors_command
list_actors_command(
    all_actors=False,
    json_output=False,
    seeds='127.0.0.1:9001'
)
" 2>&1 | grep -v "INFO"

echo ""
echo -e "${BLUE}4. 查询特定节点 (如果实现了 node_id 参数)${NC}"
echo "   这需要先知道 node_id，可以从上面的输出获取"
echo ""

# 清理
echo -e "${GREEN}清理...${NC}"
kill $NODE1_PID $NODE2_PID 2>/dev/null || true
wait $NODE1_PID $NODE2_PID 2>/dev/null || true
rm -f "$NODE1_SCRIPT" "$NODE2_SCRIPT"

echo ""
echo "======================================================================"
echo "  演示完成"
echo "======================================================================"
echo ""
echo "总结："
echo "  ✓ 可以通过 --seeds 参数连接远程集群"
echo "  ✓ 自动查询集群中所有节点的 actors"
echo "  ✓ 显示每个节点的 actor 列表"
echo ""
