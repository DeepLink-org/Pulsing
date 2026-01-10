#!/usr/bin/env python3
"""
Ray 兼容层示例（迁移用）

展示如何使用 pulsing.compat.ray 从 Ray 迁移到 Pulsing。
迁移只需修改一行 import！

用法: python examples/python/ray_compat_example.py
"""

# ============================================
# 从 Ray 迁移：只需改这一行！
# ============================================
# Before: import ray
# After:
from pulsing.compat import ray


@ray.remote
class Counter:
    """分布式计数器 (Ray 风格)"""

    def __init__(self, init_value: int = 0):
        self.value = init_value

    def get(self) -> int:
        return self.value

    def increment(self, n: int = 1) -> int:
        self.value += n
        return self.value


@ray.remote
class Calculator:
    """分布式计算器 (Ray 风格)"""

    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b


def main():
    print("=" * 60)
    print("Ray 兼容层示例 (from pulsing.compat import ray)")
    print("=" * 60)

    # 初始化 (Ray 风格)
    ray.init()
    print("✓ Pulsing (Ray compat) 已初始化")

    # --- Counter ---
    print("\n--- Counter ---")
    counter = Counter.remote(init_value=10)

    # Ray 风格调用
    print(f"初始值: {ray.get(counter.get.remote())}")
    print(f"increment(5): {ray.get(counter.increment.remote(5))}")
    print(f"最终值: {ray.get(counter.get.remote())}")

    # --- Calculator ---
    print("\n--- Calculator ---")
    calc = Calculator.remote()

    print(f"add(10, 20): {ray.get(calc.add.remote(10, 20))}")
    print(f"multiply(5, 6): {ray.get(calc.multiply.remote(5, 6))}")

    # --- 批量获取 ---
    print("\n--- 批量获取 ---")
    refs = [
        calc.add.remote(1, 2),
        calc.add.remote(3, 4),
        calc.multiply.remote(5, 6),
    ]
    results = ray.get(refs)
    print(f"批量结果: {results}")

    # --- Object Store ---
    print("\n--- put/get ---")
    ref = ray.put({"message": "Hello from pulsing.compat.ray!"})
    print(f"结果: {ray.get(ref)}")

    # 关闭 (Ray 风格)
    ray.shutdown()
    print("\n✓ 完成!")


if __name__ == "__main__":
    main()


# =============================================================================
# 迁移指南
# =============================================================================
#
# Step 1: 修改 import
# -------------------
# Before:
#     import ray
#
# After:
#     from pulsing.compat import ray
#
# Step 2: 其余代码完全不变！
# -------------------------
# ray.init()
# @ray.remote
# Counter.remote()
# counter.incr.remote()
# ray.get(ref)
# ray.shutdown()
#
# =============================================================================
# 下一步：迁移到原生 API（可选，性能更好）
# =============================================================================
#
# from pulsing.actor import init, shutdown, remote
#
# await init()
#
# @remote
# class Counter:
#     ...
#
# counter = await Counter.spawn()
# result = await counter.incr()  # 无需 .remote() + get()！
#
# await shutdown()
#
