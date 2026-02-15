#!/usr/bin/env python3
"""
Message Patterns - Common actor communication patterns

Usage: python examples/python/message_patterns.py
"""

import asyncio

import pulsing as pul


@pul.remote
class PatternDemo:
    """Actor with various message patterns."""

    def __init__(self):
        self.value = 0

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
    await pul.init()
    try:
        print("=" * 50)
        print("Pattern 1: Simple method calls")
        print("=" * 50)

        demo = await PatternDemo.spawn(name="demo")

        print(await demo.add(10))  # {'value': 10}
        print(await demo.add(5))  # {'value': 15}
        print(await demo.get())  # {'value': 15}

        print("\n" + "=" * 50)
        print("Pattern 2: Typed resolve")
        print("=" * 50)

        resolved = await PatternDemo.resolve("demo")
        print(await resolved.get())

        print("\n" + "=" * 50)
        print("Pattern 3: Async generator streaming")
        print("=" * 50)

        async for chunk in demo.stream():
            print(chunk["token"], end="")
        print()
    finally:
        await pul.shutdown()

    print("\n✓ Done!")


if __name__ == "__main__":
    asyncio.run(main())
