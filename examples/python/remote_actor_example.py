#!/usr/bin/env python3
"""
@as_actor 装饰器示例

用法: python examples/python/remote_actor_example.py
"""

import asyncio
import logging

from pulsing.actor import SystemConfig, as_actor, create_actor_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@as_actor
class Counter:
    """分布式计数器"""

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


@as_actor
class KeyValueStore:
    """分布式键值存储"""

    def __init__(self):
        self.store: dict = {}

    def put(self, key: str, value) -> None:
        self.store[key] = value

    def get(self, key: str, default=None):
        return self.store.get(key, default)

    def keys(self) -> list:
        return list(self.store.keys())


@as_actor
class AsyncWorker:
    """支持异步方法"""

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
    print("@remote 装饰器示例")
    print("=" * 60)

    system = await create_actor_system(SystemConfig.standalone())

    try:
        # --- Counter ---
        print("\n--- Counter (本地创建) ---")
        counter = await Counter.local(system, init_value=10)

        print(f"初始值: {await counter.get()}")
        print(f"increment(5): {await counter.increment(5)}")
        print(f"decrement(3): {await counter.decrement(3)}")
        print(f"最终值: {await counter.get()}")

        # --- KeyValueStore ---
        print("\n--- KeyValueStore ---")
        kv = await KeyValueStore.local(system)

        await kv.put("name", "Pulsing")
        await kv.put("version", "0.1.0")

        print(f"name: {await kv.get('name')}")
        print(f"version: {await kv.get('version')}")
        print(f"keys: {await kv.keys()}")

        # --- AsyncWorker ---
        print("\n--- AsyncWorker ---")
        worker = await AsyncWorker.local(system, worker_id="worker-001")

        result = await worker.process("hello")
        print(f"处理结果: {result}")

        status = await worker.status()
        print(f"状态: {status}")

        # --- 并行调用 ---
        print("\n--- 并行调用 ---")
        workers = [
            await AsyncWorker.local(system, worker_id=f"worker-{i}") for i in range(3)
        ]

        tasks = [w.process(f"task-{i}") for i, w in enumerate(workers)]
        results = await asyncio.gather(*tasks)

        for r in results:
            print(f"  {r['worker']}: {r['input']} -> {r['output']}")

        # --- 远程创建（单节点会 fallback 到本地）---
        print("\n--- 远程创建（单节点 fallback）---")
        remote_counter = await Counter.remote(system, init_value=100)
        print(f"远程 Counter: {await remote_counter.get()}")

        print("\n✓ 完成!")

    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
