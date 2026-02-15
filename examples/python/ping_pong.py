#!/usr/bin/env python3
"""
Ping-Pong Example - Simplest Actor Communication

Usage: python examples/python/ping_pong.py
"""

import asyncio
import pulsing as pul


@pul.remote
class PingPong:
    def ping(self) -> str:
        return "pong"

    def echo(self, msg: str) -> str:
        return f"echo: {msg}"


async def main():
    await pul.init()
    try:
        actor = await PingPong.spawn()

        print(await actor.ping())  # -> pong
        print(await actor.echo("hello"))  # -> echo: hello

        await asyncio.sleep(1)  # Allow background tasks to complete
    finally:
        await pul.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
