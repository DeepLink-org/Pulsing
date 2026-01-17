"""
AutoGen Distributed Example - Based on PulsingRuntime

Demonstrates multi-process Agent collaboration: Writer writes -> Editor reviews -> Manager coordinates

Usage:
    ./run_distributed.sh           # torchrun one-click startup
    ./run_distributed.sh --manual  # Manual mode
    python distributed.py          # Standalone mode
"""

import asyncio
import os
import sys
from dataclasses import dataclass

from autogen_core import AgentId


# ============================================================================
# Message Types
# ============================================================================


@dataclass
class RequestToSpeak:
    topic: str


@dataclass
class ChatMessage:
    content: str
    sender: str


# ============================================================================
# Agent Definitions
# ============================================================================


class WriterAgent:
    """Writer - Responsible for content creation"""

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
    """Editor - Responsible for content review"""

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
# Distributed Configuration
# ============================================================================

ROLE_MAP = {
    0: ("writer", WriterAgent),
    1: ("editor", EditorAgent),
    2: ("manager", None),
}

PULSING_BASE_PORT = 19000


def get_distributed_config():
    """Get distributed configuration from environment variables (compatible with torchrun)"""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", -1)))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    pulsing_base_port = int(os.environ.get("PULSING_BASE_PORT", PULSING_BASE_PORT))
    return rank, world_size, master_addr, pulsing_base_port


def get_pulsing_config(rank: int, master_addr: str, pulsing_base_port: int):
    """Generate Pulsing configuration based on rank"""
    my_port = pulsing_base_port + rank
    my_addr = f"0.0.0.0:{my_port}"
    seeds = [] if rank == 0 else [f"{master_addr}:{pulsing_base_port}"]
    return my_addr, seeds


# ============================================================================
# Run Functions
# ============================================================================


async def run_with_rank(
    rank: int, world_size: int, master_addr: str, pulsing_base_port: int
):
    """Run corresponding role based on rank"""
    from pulsing.autogen import PulsingRuntime

    my_addr, seeds = get_pulsing_config(rank, master_addr, pulsing_base_port)
    role_name, agent_class = ROLE_MAP.get(rank, (f"worker_{rank}", None))

    print(f"[{role_name}] Starting at {my_addr}")

    runtime = PulsingRuntime(addr=my_addr, seeds=seeds)
    await runtime.start()

    if agent_class:
        await runtime.register_factory(role_name, lambda cls=agent_class: cls())

    await asyncio.sleep(2)  # Wait for cluster to stabilize

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
    """Manager business logic"""
    print("\n" + "=" * 50)
    print("Manager: Starting conversation")
    print("=" * 50)

    writer_id = AgentId("writer", "default")
    editor_id = AgentId("editor", "default")

    # Writer writes
    print("\n[1] Asking Writer...")
    response = await runtime.send_message(
        RequestToSpeak(topic="AI"), recipient=writer_id
    )
    print(f"    Response: {response}")

    # Editor reviews
    print("\n[2] Asking Editor...")
    response = await runtime.send_message(
        RequestToSpeak(topic="review"), recipient=editor_id
    )
    print(f"    Response: {response}")

    print("\n" + "=" * 50)
    print("Manager: Done!")
    print("=" * 50)


async def run_standalone():
    """Standalone mode"""
    from pulsing.autogen import PulsingRuntime

    print("Running in standalone mode")
    runtime = PulsingRuntime()
    await runtime.start()

    await runtime.register_factory("writer", WriterAgent)
    await runtime.register_factory("editor", EditorAgent)

    await run_manager_logic(runtime)
    await runtime.stop()


# ============================================================================
# Manual Mode (Multi-terminal Startup)
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
# Entry Point
# ============================================================================


def main():
    rank, world_size, master_addr, pulsing_base_port = get_distributed_config()

    # torchrun mode
    if rank >= 0 and world_size > 0:
        asyncio.run(run_with_rank(rank, world_size, master_addr, pulsing_base_port))
        return

    # Command line mode
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
