#!/usr/bin/env python3
"""Integration test for pulsing actor list command"""

import asyncio
import subprocess
import time
from pulsing.actor import init, remote


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
    from pulsing.actor import get_system

    system = get_system()
    print("✓ Actor system started on 0.0.0.0:8888")

    # Create actors
    await Counter.remote(system, name="counter-1")
    await Counter.remote(system, name="counter-2")
    await Calculator.remote(system, name="calculator")
    print("✓ Created 3 actors")

    # Wait a bit for actors to fully initialize
    await asyncio.sleep(0.5)

    # Run list command (subprocess in same PYTHONPATH)
    result = subprocess.run(
        ["python", "-m", "pulsing.cli", "actor", "list"], capture_output=True, text=True
    )

    print("\n--- Output from 'pulsing actor list' ---")
    print(result.stdout)

    if result.returncode != 0:
        print("STDERR:")
        print(result.stderr)

    # Test --all flag
    result_all = subprocess.run(
        ["python", "-m", "pulsing.cli", "actor", "list", "--all_actors", "True"],
        capture_output=True,
        text=True,
    )

    print("\n--- Output from 'pulsing actor list --all_actors True' ---")
    print(result_all.stdout)

    # Test JSON output
    result_json = subprocess.run(
        ["python", "-m", "pulsing.cli", "actor", "list", "--json", "True"],
        capture_output=True,
        text=True,
    )

    print("\n--- Output from 'pulsing actor list --json True' ---")
    print(result_json.stdout)


if __name__ == "__main__":
    asyncio.run(main())
