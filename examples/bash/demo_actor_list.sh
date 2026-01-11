#!/usr/bin/env bash
# Pulsing Actor List 命令演示
# 演示在实际应用场景中如何使用 actor list 功能

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
echo "  Pulsing Actor List 演示"
echo "======================================================================"
echo ""

# 检查 pyenv
if ! command -v pyenv &> /dev/null; then
    echo "错误: 需要 pyenv"
    exit 1
fi

PYTHON="pyenv exec python"
echo -e "${BLUE}Python:${NC} $($PYTHON --version)"
echo ""

# 创建演示应用
DEMO_APP=$(mktemp /tmp/pulsing_demo.XXXXXX.py)

cat > "$DEMO_APP" << 'EOF'
import asyncio
from pulsing.actor import init, remote, get_system
from pulsing.cli.actor_list import list_actors_impl


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
    print("=" * 80)
    print("演示：在应用中使用 pulsing actor list")
    print("=" * 80)
    print()
    
    # 初始化
    print("1. 初始化 actor system...")
    await init()
    system = get_system()
    print(f"   ✓ 系统启动: {system.addr}\n")
    
    # 创建 actors
    print("2. 创建业务 actors...")
    await Counter.remote(system, name="counter-1")
    await Counter.remote(system, name="counter-2")
    await Calculator.remote(system, name="calculator")
    print("   ✓ 创建了 3 个 actors\n")
    
    # 使用 Python API 列出 actors
    print("3. 使用 Python API 查看 actors:")
    print("   " + "-" * 76)
    names = system.local_actor_names()
    user_actors = [n for n in names if not n.startswith("_")]
    print(f"   本地 actors: {', '.join(sorted(user_actors))}\n")
    
    # 使用 CLI 功能（表格格式）
    print("4. 使用 CLI 格式化输出（只显示用户 actors）:")
    print("   " + "-" * 76)
    await list_actors_impl(all_actors=False, output_format="table")
    print()
    
    # 显示所有 actors
    print("5. 显示所有 actors（包括系统 actors）:")
    print("   " + "-" * 76)
    await list_actors_impl(all_actors=True, output_format="table")
    print()
    
    # JSON 格式
    print("6. JSON 格式输出:")
    print("   " + "-" * 76)
    await list_actors_impl(all_actors=False, output_format="json")
    print()
    
    print("=" * 80)
    print("✓ 演示完成")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
EOF

echo -e "${GREEN}运行演示...${NC}"
echo ""

# 运行演示（过滤日志）
$PYTHON "$DEMO_APP" 2>&1 | grep -v "INFO"

# 清理
rm -f "$DEMO_APP"

echo ""
echo "======================================================================"
echo "  使用说明"
echo "======================================================================"
echo ""
echo "在你的应用中集成 actor list:"
echo ""
echo -e "${BLUE}  from pulsing.actor import init, get_system${NC}"
echo -e "${BLUE}  from pulsing.cli.actor_list import list_actors_impl${NC}"
echo ""
echo -e "${BLUE}  await init()${NC}"
echo -e "${BLUE}  # ... 创建 actors ...${NC}"
echo ""
echo -e "${BLUE}  # 表格格式${NC}"
echo -e "${BLUE}  await list_actors_impl(all_actors=False, output_format='table')${NC}"
echo ""
echo -e "${BLUE}  # JSON 格式${NC}"
echo -e "${BLUE}  await list_actors_impl(all_actors=False, output_format='json')${NC}"
echo ""
echo -e "${BLUE}  # 或使用底层 API${NC}"
echo -e "${BLUE}  names = get_system().local_actor_names()${NC}"
echo ""
