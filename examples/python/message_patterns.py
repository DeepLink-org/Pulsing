#!/usr/bin/env python3
"""
Message Patterns - Common actor communication patterns

Usage: python examples/python/message_patterns.py
"""

import asyncio

import pulsing as pul


class PatternDemo:
    """Base Actor with various message patterns."""

    def __init__(self):
        self.value = 0

    async def receive(self, msg):
        # Pattern 1: Simple object messaging (dict, list, string, etc.)
        if isinstance(msg, dict):
            if msg.get("action") == "add":
                self.value += msg.get("n", 1)
                return {"value": self.value}
            if msg.get("action") == "get":
                return {"value": self.value}

        # Pattern 2: Streaming response - just return a generator!
        if msg == "stream":

            async def generate():
                for token in ["Hello", " ", "World", "!"]:
                    yield {"token": token}
                    await asyncio.sleep(0.1)

            return generate()

        return f"unknown: {msg}"


@pul.remote
class RemotePatternDemo:
    """@pul.remote Actor with cleaner API (recommended)."""

    def __init__(self):
        self.value = 0

    # Sync method - simple request/response
    def add(self, n: int = 1) -> dict:
        self.value += n
        return {"value": self.value}

    def get(self) -> dict:
        return {"value": self.value}

    # Async generator - automatic streaming
    async def stream(self):
        for token in ["Hello", " ", "World", "!"]:
            yield {"token": token}
            await asyncio.sleep(0.1)


async def main():
    system = await pul.actor_system()

    print("=" * 50)
    print("Pattern 1: Base Actor with dict messages")
    print("=" * 50)

    actor = await system.spawn(PatternDemo(), name="demo")

    print(await actor.ask({"action": "add", "n": 10}))  # {'value': 10}
    print(await actor.ask({"action": "add", "n": 5}))  # {'value': 15}
    print(await actor.ask({"action": "get"}))  # {'value': 15}

    print("\n" + "=" * 50)
    print("Pattern 2: Base Actor streaming (return generator)")
    print("=" * 50)

    response = await actor.ask("stream")
    async for chunk in response.stream_reader():
        print(chunk["token"], end="")
    print()

    print("\n" + "=" * 50)
    print("Pattern 3: @pul.remote (recommended)")
    print("=" * 50)

    service = await RemotePatternDemo.local(system)

    # Direct method calls - no need for ask/tell!
    print(await service.add(10))  # {'value': 10}
    print(await service.add(5))  # {'value': 15}
    print(await service.get())  # {'value': 15}

    print("\n--- Async generator streaming ---")
    async for chunk in service.stream():
        print(chunk["token"], end="")
    print()

    await system.shutdown()
    print("\n✓ Done!")


if __name__ == "__main__":
    asyncio.run(main())
