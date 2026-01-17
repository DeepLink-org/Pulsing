"""
Runtime Lifecycle Management - Best Practices Example

Demonstrates how to properly handle scenarios of repeatedly creating and destroying runtime.
"""

import asyncio

from pulsing.agent import agent, cleanup, runtime


@agent(role="Counter", goal="Accumulate numbers")
class Counter:
    def __init__(self, initial: int = 0):
        self.value = initial

    async def increment(self) -> int:
        self.value += 1
        return self.value

    async def get_value(self) -> int:
        return self.value


async def example_simple():
    """Example 1: Simple scenario (no cleanup needed)"""
    print("\n=== Example 1: Simple Scenario ===")
    async with runtime():
        counter = await Counter.spawn(name="counter", initial=0)
        for _ in range(5):
            value = await counter.increment()
            print(f"Current value: {value}")


async def example_repeated_with_cleanup():
    """Example 2: Repeated create/destroy (recommended pattern)"""
    print("\n=== Example 2: Repeated Create/Destroy (with cleanup) ===")

    for i in range(3):
        try:
            async with runtime():
                counter = await Counter.spawn(name=f"counter_{i}", initial=i * 10)
                value = await counter.increment()
                print(f"Task {i}: result = {value}")
        finally:
            cleanup()  # ⭐ Ensure cleanup each time
            print(f"Task {i}: cleaned up")


async def example_batch_processing():
    """Example 3: Batch processing (shared runtime)"""
    print("\n=== Example 3: Batch Processing (shared runtime) ===")
    try:
        async with runtime():
            # Create multiple counters
            counters = []
            for i in range(5):
                counter = await Counter.spawn(name=f"counter_{i}", initial=i)
                counters.append(counter)

            # Concurrent processing
            results = await asyncio.gather(*[c.increment() for c in counters])
            print(f"Results: {results}")
    finally:
        cleanup()


async def example_error_handling():
    """Example 4: Error handling"""
    print("\n=== Example 4: Error Handling ===")

    for i in range(2):
        try:
            async with runtime():
                counter = await Counter.spawn(name=f"counter_{i}", initial=i)
                await counter.increment()

                if i == 0:
                    # Simulate error
                    raise ValueError("Simulated error")

                print(f"Task {i} succeeded")
        except ValueError as e:
            print(f"Task {i} failed: {e}")
        finally:
            cleanup()  # ⭐ Clean up even on error
            print(f"Task {i} cleaned up")


async def example_helper_pattern():
    """Example 5: Using helper function pattern"""
    print("\n=== Example 5: Helper Function Pattern ===")

    async def run_counter_task(task_id: int, increments: int) -> int:
        """Encapsulated task function (auto cleanup)"""
        try:
            async with runtime():
                counter = await Counter.spawn(name=f"task_{task_id}", initial=0)
                for _ in range(increments):
                    await counter.increment()
                return await counter.get_value()
        finally:
            cleanup()

    # Run multiple tasks
    tasks = [run_counter_task(i, i + 1) for i in range(3)]
    results = []
    for task in tasks:
        result = await task
        results.append(result)
        print(f"Task completed, result: {result}")

    print(f"All results: {results}")


async def main():
    """Run all examples"""
    print("Runtime Lifecycle Management - Best Practices\n")

    # Example 1: Simple scenario
    await example_simple()

    # Example 2: Repeated create/destroy (recommended)
    await example_repeated_with_cleanup()

    # Example 3: Batch processing
    await example_batch_processing()

    # Example 4: Error handling
    await example_error_handling()

    # Example 5: Helper function pattern
    await example_helper_pattern()

    print("\n✅ All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
