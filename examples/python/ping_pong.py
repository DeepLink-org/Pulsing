#!/usr/bin/env python3
"""
Ping-Pong Example - Simplest Actor Communication

Usage: python examples/python/ping_pong.py
"""

import asyncio
import pulsing as pul


class PingPong:
    async def receive(self, msg):
        if msg == "ping":
            return "pong"
        return f"echo: {msg}"


async def main():
    system = await pul.actor_system()
    actor = await system.spawn(PingPong())

    # Simple string message
    print(await actor.ask("ping"))  # -> pong
    print(await actor.ask("hello"))  # -> echo: hello

    await asyncio.sleep(1)  # Allow background tasks to complete
    await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
