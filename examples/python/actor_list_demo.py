#!/usr/bin/env python3
"""Example demonstrating pulsing actor list command"""

import asyncio
from pulsing.actor import init, remote, get_system
from pulsing.cli.actor_list import list_actors_impl


@remote
class Counter:
    """A simple counter actor"""

    def __init__(self, initial_value=0):
        self.count = initial_value

    def increment(self):
        self.count += 1
        return self.count

    def get_count(self):
        return self.count


@remote
class Calculator:
    """A simple calculator actor"""

    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b


async def main():
    print("=== Pulsing Actor List Demo ===\n")

    # Initialize actor system
    print("1. Initializing actor system...")
    await init()
    system = get_system()
    print("   ✓ Actor system initialized\n")

    # Create some named actors
    print("2. Creating actors...")
    counter1 = await Counter.remote(system, name="counter-1")
    counter2 = await Counter.remote(system, name="counter-2")
    calc = await Calculator.remote(system, name="calculator")
    print("   ✓ Created 3 actors: counter-1, counter-2, calculator\n")

    # Use the actors
    print("3. Testing actors...")
    result1 = await counter1.increment()
    result2 = await calc.add(10, 20)
    print(f"   counter-1.increment() = {result1}")
    print(f"   calculator.add(10, 20) = {result2}\n")

    # List user actors only (default)
    print("4. Listing user actors (pulsing actor list):")
    print("-" * 80)
    await list_actors_impl(all_actors=False, output_format="table")
    print()

    # List all actors including system ones
    print("5. Listing all actors including system actors (pulsing actor list --all_actors True):")
    print("-" * 80)
    await list_actors_impl(all_actors=True, output_format="table")
    print()

    # JSON output
    print("6. JSON format output (pulsing actor list --json True):")
    print("-" * 80)
    await list_actors_impl(all_actors=False, output_format="json")
    print()

    # Using lower-level API
    print("7. Using lower-level API (system.local_actor_names()):")
    print("-" * 80)
    all_names = system.local_actor_names()
    user_names = [n for n in all_names if not n.startswith("_")]
    print(f"   All actor names: {all_names}")
    print(f"   User actor names: {user_names}\n")

    print("=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
