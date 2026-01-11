#!/usr/bin/env python3
"""Test pulsing actor list in same process"""

import asyncio
import sys
import os

# Ensure we're using the local pulsing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

from pulsing.actor import init, remote, get_system
from pulsing.admin import list_actors


@remote
class Counter:
    def __init__(self):
        self.count = 0


@remote
class Calculator:
    def add(self, a, b):
        return a + b


async def main():
    # Start system
    await init(addr="0.0.0.0:8888")
    system = get_system()
    print("✓ Actor system started on 0.0.0.0:8888\n")

    # Create actors
    await Counter.remote(system, name="counter-1")
    await Counter.remote(system, name="counter-2")
    await Calculator.remote(system, name="calculator")
    print("✓ Created 3 actors\n")

    # List actors directly (Python API)
    print("=== Using Python API (pulsing.admin.list_actors) ===\n")

    # Get all local actor names
    names = system.local_actor_names()
    print(f"All local actor names: {names}\n")

    # Filter to user actors
    user_actors = [n for n in names if not n.startswith("_")]
    print(f"User actors: {user_actors}\n")

    print("=== Formatted list ===\n")
    print(f"{'Name':<30} {'Type':<20}")
    print("-" * 50)
    for name in user_actors:
        actor_type = "user"
        print(f"{name:<30} {actor_type:<20}")

    print(f"\nTotal: {len(user_actors)} actor(s)")


if __name__ == "__main__":
    asyncio.run(main())
