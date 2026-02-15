#!/usr/bin/env python3
"""
Named Actors Example

Named actors can be discovered by name instead of specific ActorId,
enabling service discovery.

Usage: python examples/python/named_actors.py
"""

import asyncio

import pulsing as pul


@pul.remote
class EchoActor:
    """Simple echo actor that can be discovered by name."""

    def echo(self, message: str) -> dict[str, str]:
        print(f"[Echo] {message}")
        return {"echo": message}


async def main():
    print("=== Pulsing Named Actors ===\n")

    await pul.init()
    try:
        print("✓ System started\n")

        await EchoActor.spawn(name="echo")
        print("✓ Created: echo (named, discoverable)\n")

        print("--- Resolve by name ---")
        actor = await EchoActor.resolve("echo")
        resp = await actor.echo("Hello!")
        print(f"Response: {resp['echo']}\n")

        print("✓ Done!")
    finally:
        await pul.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
