#!/usr/bin/env python3
"""
@remote Decorator Example (Native Async API)

Demonstrates pulsing.actor's concise API:
- await init() for initialization
- @remote decorator
- await Counter.spawn() to create actor
- await counter.method() to call methods

Usage: python examples/python/remote_actor_example.py
"""

import asyncio
import logging

from pulsing.actor import init, shutdown, remote

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    def decrement(self, n: int = 1) -> int:
        self.value -= n
        return self.value


@remote
class KeyValueStore:
    """Distributed key-value store"""

    def __init__(self):
        self.store: dict = {}

    def put(self, key: str, value) -> None:
        self.store[key] = value

    def get(self, key: str, default=None):
        return self.store.get(key, default)

    def keys(self) -> list:
        return list(self.store.keys())


@remote
class AsyncWorker:
    """Supports async methods"""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.count = 0

    async def process(self, data: str) -> dict:
        await asyncio.sleep(0.01)
        self.count += 1
        return {"worker": self.worker_id, "input": data, "output": data.upper()}

    def status(self) -> dict:
        return {"worker_id": self.worker_id, "processed": self.count}


async def main():
    print("=" * 60)
    print("@remote Decorator Example (Native Async API)")
    print("=" * 60)

    # Simple initialization
    await init()

    # --- Counter ---
    print("\n--- Counter ---")
    counter = await Counter.spawn(init_value=10)

    # Direct await, no need for .remote() + get()
    print(f"Initial value: {await counter.get()}")
    print(f"increment(5): {await counter.increment(5)}")
    print(f"decrement(3): {await counter.decrement(3)}")
    print(f"Final value: {await counter.get()}")

    # --- KeyValueStore ---
    print("\n--- KeyValueStore ---")
    kv = await KeyValueStore.spawn()

    await kv.put("name", "Pulsing")
    await kv.put("version", "0.7.0")

    print(f"name: {await kv.get('name')}")
    print(f"version: {await kv.get('version')}")
    print(f"keys: {await kv.keys()}")

    # --- AsyncWorker ---
    print("\n--- AsyncWorker ---")
    worker = await AsyncWorker.spawn(worker_id="worker-001")

    result = await worker.process("hello")
    print(f"Process result: {result}")

    status = await worker.status()
    print(f"Status: {status}")

    # --- Parallel calls ---
    print("\n--- Parallel Calls ---")
    workers = [await AsyncWorker.spawn(worker_id=f"worker-{i}") for i in range(3)]

    tasks = [w.process(f"task-{i}") for i, w in enumerate(workers)]
    results = await asyncio.gather(*tasks)

    for r in results:
        print(f"  {r['worker']}: {r['input']} -> {r['output']}")

    print("\n✓ Done!")

    # Shutdown
    await shutdown()


if __name__ == "__main__":
    asyncio.run(main())
