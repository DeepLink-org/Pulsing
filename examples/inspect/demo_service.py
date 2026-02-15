#!/usr/bin/env python3
"""
Demo Service for Inspect CLI

This example starts a multi-node cluster with various actors to demonstrate
the inspect CLI commands.

Usage:
    Terminal 1: python examples/inspect/demo_service.py --port 8000
    Terminal 2: python examples/inspect/demo_service.py --port 8001 --seed 127.0.0.1:8000
    Terminal 3: python examples/inspect/demo_service.py --port 8002 --seed 127.0.0.1:8000

Then in another terminal, try:
    pulsing inspect cluster --seeds 127.0.0.1:8000
    pulsing inspect actors --seeds 127.0.0.1:8000
    pulsing inspect metrics --seeds 127.0.0.1:8000
"""

import argparse
import asyncio
import random
import time

import pulsing as pul


@pul.remote
class WorkerActor:
    """A simple worker actor that processes tasks"""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.tasks_processed = 0

    async def process(self, task: str) -> dict[str, str | int]:
        self.tasks_processed += 1
        result = f"Processed: {task} (total: {self.tasks_processed})"
        print(f"[Worker {self.worker_id}] {result}")
        return {"result": result, "worker": self.worker_id}

    def stats(self) -> dict[str, str | int]:
        return {"worker_id": self.worker_id, "tasks": self.tasks_processed}


@pul.remote
class DispatcherActor:
    """A dispatcher actor that distributes tasks to workers (for demo purposes)"""

    def __init__(self):
        self.workers = []
        self.tasks_dispatched = 0

    def route(self, task: str) -> dict[str, str | int | bool]:
        self.tasks_dispatched += 1
        worker_id = f"worker-{random.randint(1, 3)}"
        return {
            "task": task,
            "worker": worker_id,
            "dispatched": self.tasks_dispatched,
        }

    def stats(self) -> dict[str, int | bool]:
        return {"dispatcher": True, "tasks_dispatched": self.tasks_dispatched}


@pul.remote
class CacheActor:
    """A cache actor that stores key-value pairs"""

    def __init__(self):
        self.cache = {}

    def get(self, key: str) -> dict[str, object]:
        value = self.cache.get(key, None)
        return {"key": key, "value": value, "found": value is not None}

    def set(self, key: str, value: object) -> dict[str, str | bool]:
        self.cache[key] = value
        return {"key": key, "success": True}

    def stats(self) -> dict[str, int]:
        return {"cache_size": len(self.cache)}


async def run_node(port: int, seed: str | None):
    """Run a node in the cluster"""
    print(f"\n{'=' * 60}")
    print(f"Pulsing Demo Service - Node on port {port}")
    print(f"{'=' * 60}\n")

    addr = f"127.0.0.1:{port}"
    seeds = [seed] if seed else None

    await pul.init(addr=addr, seeds=seeds)
    system = pul.ActorSystem(pul.get_system())
    print(f"✓ System started: {addr}")
    if seed:
        print(f"  Joined via: {seed}")
    print()

    # Create different actors based on node role
    if seed is None:
        # Node 1: Create dispatcher and some workers
        print("Creating actors on node 1...")
        await DispatcherActor.spawn(name="dispatcher")
        print("  ✓ actors/dispatcher")

        for i in range(1, 3):
            worker_name = f"worker-{i}"
            await WorkerActor.spawn(worker_name, name=worker_name)
            print(f"  ✓ actors/{worker_name}")

        print("\n✓ Node 1 ready!")
        print("\nTo start more nodes:")
        print(
            "  Terminal 2: python examples/inspect/demo_service.py --port 8001 --seed 127.0.0.1:8000"
        )
        print(
            "  Terminal 3: python examples/inspect/demo_service.py --port 8002 --seed 127.0.0.1:8000"
        )
        print("\nThen try inspect commands:")
        print("  pulsing inspect cluster --seeds 127.0.0.1:8000")
        print("  pulsing inspect actors --seeds 127.0.0.1:8000")
        print("  pulsing inspect metrics --seeds 127.0.0.1:8000")
        print("  pulsing inspect watch --seeds 127.0.0.1:8000 --kind cluster\n")

    elif port == 8001:
        # Node 2: Add more workers
        await asyncio.sleep(1)  # Wait for cluster discovery
        print("Creating actors on node 2...")
        for i in range(3, 5):
            worker_name = f"worker-{i}"
            await WorkerActor.spawn(worker_name, name=worker_name)
            print(f"  ✓ actors/{worker_name}")
        print("\n✓ Node 2 ready!")

    elif port == 8002:
        # Node 3: Add cache
        await asyncio.sleep(1)
        print("Creating actors on node 3...")
        await CacheActor.spawn(name="cache")
        print("  ✓ actors/cache")
        print("\n✓ Node 3 ready!")

    # Keep running and periodically show cluster status
    try:
        while True:
            await asyncio.sleep(10)
            members = await system.members()
            alive = [m for m in members if m.get("status") == "Alive"]
            print(
                f"[{time.strftime('%H:%M:%S')}] Cluster: {len(alive)}/{len(members)} nodes alive"
            )
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        await pul.shutdown()
        print("✓ Shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="Pulsing Demo Service for Inspect CLI")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind (default: 8000)"
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Seed node to join (e.g., 127.0.0.1:8000)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_node(args.port, args.seed))
    except KeyboardInterrupt:
        print("\nInterrupted")


if __name__ == "__main__":
    main()
