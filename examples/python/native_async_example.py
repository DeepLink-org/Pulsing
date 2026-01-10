#!/usr/bin/env python3
"""
Pulsing 原生异步 API 示例（推荐）

展示 pulsing.actor 的简洁异步 API。

用法: python examples/python/native_async_example.py
"""

import asyncio

# Pulsing 原生 API
from pulsing.actor import init, shutdown, remote


@remote
class Counter:
    """分布式计数器"""

    def __init__(self, init_value: int = 0):
        self.value = init_value

    def get(self) -> int:
        return self.value

    def increment(self, n: int = 1) -> int:
        self.value += n
        return self.value


@remote
class Calculator:
    """分布式计算器"""

    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b


@remote
class AsyncWorker:
    """异步 Worker"""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.count = 0

    async def process(self, data: str) -> dict:
        await asyncio.sleep(0.01)  # 模拟处理
        self.count += 1
        return {
            "worker": self.worker_id,
            "input": data,
            "output": data.upper(),
            "processed": self.count,
        }


async def main():
    print("=" * 60)
    print("Pulsing 原生异步 API 示例")
    print("=" * 60)

    # 初始化（简洁！）
    await init()
    print("✓ Pulsing 已初始化")

    # --- Counter ---
    print("\n--- Counter ---")
    counter = await Counter.spawn(init_value=10)

    # 直接 await，无需 .remote() + get()
    print(f"初始值: {await counter.get()}")
    print(f"increment(5): {await counter.increment(5)}")
    print(f"最终值: {await counter.get()}")

    # --- Calculator ---
    print("\n--- Calculator ---")
    calc = await Calculator.spawn()

    print(f"add(10, 20): {await calc.add(10, 20)}")
    print(f"multiply(5, 6): {await calc.multiply(5, 6)}")

    # --- 并行调用 ---
    print("\n--- 并行调用 ---")
    results = await asyncio.gather(
        calc.add(1, 2),
        calc.add(3, 4),
        calc.multiply(5, 6),
    )
    print(f"并行结果: {results}")

    # --- AsyncWorker ---
    print("\n--- 异步 Worker ---")
    worker = await AsyncWorker.spawn(worker_id="worker-001")
    result = await worker.process("hello pulsing")
    print(f"处理结果: {result}")

    # --- 关闭 ---
    await shutdown()
    print("\n✓ 完成!")


if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# API 对比
# =============================================================================
#
# | 操作           | Pulsing 原生 (async)        | Ray 兼容层 (sync)           |
# |----------------|-----------------------------|-----------------------------|
# | 初始化         | await init()                | ray.init()                  |
# | 装饰器         | @remote                     | @ray.remote                 |
# | 创建 actor     | await Counter.spawn()       | Counter.remote()            |
# | 调用方法       | await counter.incr()        | counter.incr.remote()       |
# | 获取结果       | 直接返回                    | ray.get(ref)                |
# | 关闭           | await shutdown()            | ray.shutdown()              |
#
# 推荐使用原生 API：
# - 更 Pythonic（标准 async/await）
# - 无需 .remote() + get() 样板代码
# - 更好的性能（无同步包装开销）
#
