"""
AutoGen Compatibility Example - Standalone Mode

Demonstrates how PulsingRuntime can directly replace AutoGen SingleThreadedAgentRuntime

Usage:
    pip install autogen-core
    python simple.py
"""

import asyncio
from dataclasses import dataclass

from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
)
from pulsing.autogen import PulsingRuntime


# Define message types
@dataclass
class Ping:
    content: str


@dataclass
class Pong:
    content: str
    count: int


# Use AutoGen native Agent definition
class PingPongAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("PingPong Agent")
        self._count = 0

    @message_handler
    async def handle_ping(self, message: Ping, ctx: MessageContext) -> Pong:
        self._count += 1
        print(f"[PingPongAgent] Received: {message.content}, count: {self._count}")
        return Pong(content=f"Pong: {message.content}", count=self._count)


async def main_with_pulsing():
    """Use PulsingRuntime"""
    print("=" * 50)
    print("Running with PulsingRuntime (standalone mode)")
    print("=" * 50)

    # Create runtime - standalone mode
    runtime = PulsingRuntime()
    await runtime.start()

    # Register Agent
    await runtime.register_factory(
        "ping_pong",
        lambda: PingPongAgent(),
    )

    # Send messages
    for i in range(3):
        response = await runtime.send_message(
            Ping(content=f"Hello {i}"),
            recipient=AgentId("ping_pong", "default"),
        )
        print(f"[Main] Response: {response}")

    await runtime.stop()
    print("Done!\n")


async def main_with_autogen():
    """Use AutoGen SingleThreadedAgentRuntime (control group)"""
    print("=" * 50)
    print("Running with AutoGen SingleThreadedAgentRuntime")
    print("=" * 50)

    runtime = SingleThreadedAgentRuntime()

    await PingPongAgent.register(
        runtime,
        "ping_pong",
        lambda: PingPongAgent(),
    )

    runtime.start()

    for i in range(3):
        response = await runtime.send_message(
            Ping(content=f"Hello {i}"),
            recipient=AgentId("ping_pong", "default"),
        )
        print(f"[Main] Response: {response}")

    await runtime.stop()
    print("Done!\n")


async def main():
    # First use Pulsing
    await main_with_pulsing()

    # Then use AutoGen for comparison
    await main_with_autogen()


if __name__ == "__main__":
    asyncio.run(main())
