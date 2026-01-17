#!/usr/bin/env python3
"""
Ray Compatibility Layer Example (for Migration)

Demonstrates how to use pulsing.compat.ray to migrate from Ray to Pulsing.
Migration only requires changing one import line!

Usage: python examples/python/ray_compat_example.py
"""

# ============================================
# Migrate from Ray: Just change this line!
# ============================================
# Before: import ray
# After:
from pulsing.compat import ray


@ray.remote
class Counter:
    """Distributed counter (Ray style)"""

    def __init__(self, init_value: int = 0):
        self.value = init_value

    def get(self) -> int:
        return self.value

    def increment(self, n: int = 1) -> int:
        self.value += n
        return self.value


@ray.remote
class Calculator:
    """Distributed calculator (Ray style)"""

    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b


def main():
    print("=" * 60)
    print("Ray Compatibility Layer Example (from pulsing.compat import ray)")
    print("=" * 60)

    # Initialize (Ray style)
    ray.init()
    print("✓ Pulsing (Ray compat) initialized")

    # --- Counter ---
    print("\n--- Counter ---")
    counter = Counter.remote(init_value=10)

    # Ray style calls
    print(f"Initial value: {ray.get(counter.get.remote())}")
    print(f"increment(5): {ray.get(counter.increment.remote(5))}")
    print(f"Final value: {ray.get(counter.get.remote())}")

    # --- Calculator ---
    print("\n--- Calculator ---")
    calc = Calculator.remote()

    print(f"add(10, 20): {ray.get(calc.add.remote(10, 20))}")
    print(f"multiply(5, 6): {ray.get(calc.multiply.remote(5, 6))}")

    # --- Batch get ---
    print("\n--- Batch Get ---")
    refs = [
        calc.add.remote(1, 2),
        calc.add.remote(3, 4),
        calc.multiply.remote(5, 6),
    ]
    results = ray.get(refs)
    print(f"Batch results: {results}")

    # --- Object Store ---
    print("\n--- put/get ---")
    ref = ray.put({"message": "Hello from pulsing.compat.ray!"})
    print(f"Result: {ray.get(ref)}")

    # Shutdown (Ray style)
    ray.shutdown()
    print("\n✓ Done!")


if __name__ == "__main__":
    main()


# =============================================================================
# Migration Guide
# =============================================================================
#
# Step 1: Change import
# -------------------
# Before:
#     import ray
#
# After:
#     from pulsing.compat import ray
#
# Step 2: Rest of the code remains unchanged!
# -------------------------
# ray.init()
# @ray.remote
# Counter.remote()
# counter.incr.remote()
# ray.get(ref)
# ray.shutdown()
#
# =============================================================================
# Next Step: Migrate to Native API (Optional, Better Performance)
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
# result = await counter.incr()  # No need for .remote() + get()!
#
# await shutdown()
#
