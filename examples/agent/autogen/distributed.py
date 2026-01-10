"""
AutoGen 分布式示例 - 基于 PulsingRuntime

演示多进程 Agent 协作：Writer 写作 -> Editor 审核 -> Manager 协调

Usage:
    ./run_distributed.sh           # torchrun 一键启动
    ./run_distributed.sh --manual  # 手动模式
    python distributed.py          # 单机模式
"""

import asyncio
import os
import sys
from dataclasses import dataclass

from autogen_core import AgentId


# ============================================================================
# 消息类型
# ============================================================================


@dataclass
class RequestToSpeak:
    topic: str


@dataclass
class ChatMessage:
    content: str
    sender: str


# ============================================================================
# Agent 定义
# ============================================================================


class WriterAgent:
    """Writer - 负责内容创作"""

    def __init__(self):
        self.id = None
        self._runtime = None

    async def bind_id_and_runtime(self, id, runtime):
        self.id = id
        self._runtime = runtime

    async def on_message(self, message, ctx):
        if isinstance(message, RequestToSpeak):
            content = f"Here's my writing about {message.topic}: It was a dark and stormy night..."
            return ChatMessage(content=content, sender="writer")
        return None


class EditorAgent:
    """Editor - 负责内容审核"""

    def __init__(self):
        self.id = None
        self._runtime = None

    async def bind_id_and_runtime(self, id, runtime):
        self.id = id
        self._runtime = runtime

    async def on_message(self, message, ctx):
        if isinstance(message, RequestToSpeak):
            return ChatMessage(
                content="The writing is good, but consider adding more details.",
                sender="editor",
            )
        return None


# ============================================================================
# 分布式配置
# ============================================================================

ROLE_MAP = {
    0: ("writer", WriterAgent),
    1: ("editor", EditorAgent),
    2: ("manager", None),
}

PULSING_BASE_PORT = 19000


def get_distributed_config():
    """从环境变量获取分布式配置 (兼容 torchrun)"""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", -1)))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    pulsing_base_port = int(os.environ.get("PULSING_BASE_PORT", PULSING_BASE_PORT))
    return rank, world_size, master_addr, pulsing_base_port


def get_pulsing_config(rank: int, master_addr: str, pulsing_base_port: int):
    """根据 rank 生成 Pulsing 配置"""
    my_port = pulsing_base_port + rank
    my_addr = f"0.0.0.0:{my_port}"
    seeds = [] if rank == 0 else [f"{master_addr}:{pulsing_base_port}"]
    return my_addr, seeds


# ============================================================================
# 运行函数
# ============================================================================


async def run_with_rank(
    rank: int, world_size: int, master_addr: str, pulsing_base_port: int
):
    """根据 rank 运行对应角色"""
    from pulsing.autogen import PulsingRuntime

    my_addr, seeds = get_pulsing_config(rank, master_addr, pulsing_base_port)
    role_name, agent_class = ROLE_MAP.get(rank, (f"worker_{rank}", None))

    print(f"[{role_name}] Starting at {my_addr}")

    runtime = PulsingRuntime(addr=my_addr, seeds=seeds)
    await runtime.start()

    if agent_class:
        await runtime.register_factory(role_name, lambda cls=agent_class: cls())

    await asyncio.sleep(2)  # 等待集群稳定

    if rank == world_size - 1:
        await run_manager_logic(runtime)
        await runtime.stop()
    else:
        print(f"[{role_name}] Ready")
        try:
            await runtime.stop_when_signal()
        except asyncio.CancelledError:
            pass


async def run_manager_logic(runtime):
    """Manager 业务逻辑"""
    print("\n" + "=" * 50)
    print("Manager: Starting conversation")
    print("=" * 50)

    writer_id = AgentId("writer", "default")
    editor_id = AgentId("editor", "default")

    # Writer 写作
    print("\n[1] Asking Writer...")
    response = await runtime.send_message(
        RequestToSpeak(topic="AI"), recipient=writer_id
    )
    print(f"    Response: {response}")

    # Editor 审核
    print("\n[2] Asking Editor...")
    response = await runtime.send_message(
        RequestToSpeak(topic="review"), recipient=editor_id
    )
    print(f"    Response: {response}")

    print("\n" + "=" * 50)
    print("Manager: Done!")
    print("=" * 50)


async def run_standalone():
    """单机模式"""
    from pulsing.autogen import PulsingRuntime

    print("Running in standalone mode")
    runtime = PulsingRuntime()
    await runtime.start()

    await runtime.register_factory("writer", WriterAgent)
    await runtime.register_factory("editor", EditorAgent)

    await run_manager_logic(runtime)
    await runtime.stop()


# ============================================================================
# 手动模式 (多终端启动)
# ============================================================================


async def run_writer():
    from pulsing.autogen import PulsingRuntime

    runtime = PulsingRuntime(addr="0.0.0.0:8001", seeds=[])
    await runtime.start()
    await runtime.register_factory("writer", WriterAgent)
    print("[writer] Ready")
    await runtime.stop_when_signal()


async def run_editor():
    from pulsing.autogen import PulsingRuntime

    runtime = PulsingRuntime(addr="0.0.0.0:8002", seeds=["127.0.0.1:8001"])
    await runtime.start()
    await runtime.register_factory("editor", EditorAgent)
    print("[editor] Ready")
    await runtime.stop_when_signal()


async def run_manager():
    from pulsing.autogen import PulsingRuntime

    runtime = PulsingRuntime(addr="0.0.0.0:8003", seeds=["127.0.0.1:8001"])
    await runtime.start()
    await asyncio.sleep(2)
    await run_manager_logic(runtime)
    await runtime.stop()


# ============================================================================
# 入口
# ============================================================================


def main():
    rank, world_size, master_addr, pulsing_base_port = get_distributed_config()

    # torchrun 模式
    if rank >= 0 and world_size > 0:
        asyncio.run(run_with_rank(rank, world_size, master_addr, pulsing_base_port))
        return

    # 命令行模式
    role = sys.argv[1].lower() if len(sys.argv) > 1 else "standalone"

    handlers = {
        "writer": run_writer,
        "editor": run_editor,
        "manager": run_manager,
        "standalone": run_standalone,
    }

    if role in handlers:
        asyncio.run(handlers[role]())
    else:
        print("Usage: python distributed.py [writer|editor|manager|standalone]")
        sys.exit(1)


if __name__ == "__main__":
    main()
