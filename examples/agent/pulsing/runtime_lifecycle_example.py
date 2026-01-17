"""
Runtime 生命周期管理 - 最佳实践示例

演示如何正确处理反复创建销毁 runtime 的场景。
"""

import asyncio

from pulsing.agent import agent, cleanup, runtime


@agent(role="计数器", goal="累加数字")
class Counter:
    def __init__(self, initial: int = 0):
        self.value = initial

    async def increment(self) -> int:
        self.value += 1
        return self.value

    async def get_value(self) -> int:
        return self.value


async def example_simple():
    """示例 1: 简单场景（无需清理）"""
    print("\n=== 示例 1: 简单场景 ===")
    async with runtime():
        counter = await Counter.spawn(name="counter", initial=0)
        for _ in range(5):
            value = await counter.increment()
            print(f"当前值: {value}")


async def example_repeated_with_cleanup():
    """示例 2: 反复创建销毁（推荐模式）"""
    print("\n=== 示例 2: 反复创建销毁（带清理）===")

    for i in range(3):
        try:
            async with runtime():
                counter = await Counter.spawn(name=f"counter_{i}", initial=i * 10)
                value = await counter.increment()
                print(f"任务 {i}: 结果 = {value}")
        finally:
            cleanup()  # ⭐ 确保每次清理
            print(f"任务 {i}: 已清理")


async def example_batch_processing():
    """示例 3: 批处理（共享 runtime）"""
    print("\n=== 示例 3: 批处理（共享 runtime）===")
    try:
        async with runtime():
            # 创建多个 counter
            counters = []
            for i in range(5):
                counter = await Counter.spawn(name=f"counter_{i}", initial=i)
                counters.append(counter)

            # 并发处理
            results = await asyncio.gather(*[c.increment() for c in counters])
            print(f"结果: {results}")
    finally:
        cleanup()


async def example_error_handling():
    """示例 4: 异常处理"""
    print("\n=== 示例 4: 异常处理 ===")

    for i in range(2):
        try:
            async with runtime():
                counter = await Counter.spawn(name=f"counter_{i}", initial=i)
                await counter.increment()

                if i == 0:
                    # 模拟异常
                    raise ValueError("模拟错误")

                print(f"任务 {i} 成功")
        except ValueError as e:
            print(f"任务 {i} 失败: {e}")
        finally:
            cleanup()  # ⭐ 即使有异常也要清理
            print(f"任务 {i} 已清理")


async def example_helper_pattern():
    """示例 5: 使用辅助函数封装"""
    print("\n=== 示例 5: 辅助函数模式 ===")

    async def run_counter_task(task_id: int, increments: int) -> int:
        """封装的任务函数（自动清理）"""
        try:
            async with runtime():
                counter = await Counter.spawn(name=f"task_{task_id}", initial=0)
                for _ in range(increments):
                    await counter.increment()
                return await counter.get_value()
        finally:
            cleanup()

    # 运行多个任务
    tasks = [run_counter_task(i, i + 1) for i in range(3)]
    results = []
    for task in tasks:
        result = await task
        results.append(result)
        print(f"任务完成，结果: {result}")

    print(f"所有结果: {results}")


async def main():
    """运行所有示例"""
    print("Runtime 生命周期管理 - 最佳实践\n")

    # 示例 1: 简单场景
    await example_simple()

    # 示例 2: 反复创建销毁（推荐）
    await example_repeated_with_cleanup()

    # 示例 3: 批处理
    await example_batch_processing()

    # 示例 4: 异常处理
    await example_error_handling()

    # 示例 5: 辅助函数模式
    await example_helper_pattern()

    print("\n✅ 所有示例完成！")


if __name__ == "__main__":
    asyncio.run(main())
