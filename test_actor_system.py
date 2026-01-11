#!/usr/bin/env python3
"""Test script to run a system with actors for testing 'pulsing actor list'"""

import asyncio
from pulsing.actor import init, remote


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
    await init(addr="0.0.0.0:8888")
    print("Actor system started on 0.0.0.0:8888")

    # Create some named actors
    _counter1 = await Counter.remote(name="counter-1")
    _counter2 = await Counter.remote(name="counter-2")
    _calc = await Calculator.remote(name="calculator")

    print("Created actors: counter-1, counter-2, calculator")
    print("Actor system is running. Press Ctrl+C to stop.")

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    asyncio.run(main())
