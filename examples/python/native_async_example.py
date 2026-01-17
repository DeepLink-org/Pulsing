#!/usr/bin/env python3
"""
Pulsing Native Async API Example (Recommended)

Demonstrates pulsing.actor's concise async API.

Usage: python examples/python/native_async_example.py
"""

import asyncio

# Pulsing native API
from pulsing.actor import init, shutdown, remote


@remote
class Counter:
    """Distributed counter"""

    def __init__(self, init_value: int = 0):
        self.value = init_value

    def get(self) -> int:
        return self.value

    def increment(self, n: int = 1) -> int:
        self.value += n
        return self.value


@remote
class Calculator:
    """Distributed calculator"""

    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b


@remote
class AsyncWorker:
    """Async Worker"""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.count = 0

    async def process(self, data: str) -> dict:
        await asyncio.sleep(0.01)  # Simulate processing
        self.count += 1
        return {
            "worker": self.worker_id,
            "input": data,
            "output": data.upper(),
            "processed": self.count,
        }


async def main():
    print("=" * 60)
    print("Pulsing Native Async API Example")
    print("=" * 60)

    # Initialize (simple!)
    await init()
    print("✓ Pulsing initialized")

    # --- Counter ---
    print("\n--- Counter ---")
    counter = await Counter.spawn(init_value=10)

    # Direct await, no need for .remote() + get()
    print(f"Initial value: {await counter.get()}")
    print(f"increment(5): {await counter.increment(5)}")
    print(f"Final value: {await counter.get()}")

    # --- Calculator ---
    print("\n--- Calculator ---")
    calc = await Calculator.spawn()

    print(f"add(10, 20): {await calc.add(10, 20)}")
    print(f"multiply(5, 6): {await calc.multiply(5, 6)}")

    # --- Parallel calls ---
    print("\n--- Parallel Calls ---")
    results = await asyncio.gather(
        calc.add(1, 2),
        calc.add(3, 4),
        calc.multiply(5, 6),
    )
    print(f"Parallel results: {results}")

    # --- AsyncWorker ---
    print("\n--- Async Worker ---")
    worker = await AsyncWorker.spawn(worker_id="worker-001")
    result = await worker.process("hello pulsing")
    print(f"Process result: {result}")

    # --- Shutdown ---
    await shutdown()
    print("\n✓ Done!")


if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# API Comparison
# =============================================================================
#
# | Operation      | Pulsing Native (async)      | Ray Compat Layer (sync)     |
# |----------------|-----------------------------|-----------------------------|
# | Initialize     | await init()                | ray.init()                  |
# | Decorator      | @remote                     | @ray.remote                 |
# | Create actor   | await Counter.spawn()       | Counter.remote()            |
# | Call method    | await counter.incr()        | counter.incr.remote()       |
# | Get result     | Direct return               | ray.get(ref)                |
# | Shutdown       | await shutdown()            | ray.shutdown()              |
#
# Recommended to use native API:
# - More Pythonic (standard async/await)
# - No .remote() + get() boilerplate
# - Better performance (no sync wrapper overhead)
#
