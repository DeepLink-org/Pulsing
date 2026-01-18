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

from pulsing.actor import Actor, ActorId, Message, SystemConfig, create_actor_system


class WorkerActor(Actor):
    """A simple worker actor that processes tasks"""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.tasks_processed = 0

    def on_start(self, actor_id: ActorId):
        print(f"[Worker {self.worker_id}] Started")

    def receive(self, msg: Message) -> Message:
        if msg.msg_type == "ProcessTask":
            task = msg.to_json().get("task", "")
            self.tasks_processed += 1
            result = f"Processed: {task} (total: {self.tasks_processed})"
            print(f"[Worker {self.worker_id}] {result}")
            return Message.from_json(
                "TaskResult", {"result": result, "worker": self.worker_id}
            )
        elif msg.msg_type == "GetStats":
            return Message.from_json(
                "Stats", {"worker_id": self.worker_id, "tasks": self.tasks_processed}
            )
        return Message.empty()


class RouterActor(Actor):
    """A router actor that distributes tasks to workers"""

    def __init__(self):
        self.workers = []
        self.tasks_routed = 0

    def on_start(self, actor_id: ActorId):
        print("[Router] Started")

    def receive(self, msg: Message) -> Message:
        if msg.msg_type == "RouteTask":
            self.tasks_routed += 1
            task = msg.to_json().get("task", "")
            # Simulate routing logic
            worker_id = f"worker-{random.randint(1, 3)}"
            return Message.from_json(
                "Routed",
                {"task": task, "worker": worker_id, "routed": self.tasks_routed},
            )
        elif msg.msg_type == "GetStats":
            return Message.from_json(
                "Stats", {"router": True, "tasks_routed": self.tasks_routed}
            )
        return Message.empty()


class CacheActor(Actor):
    """A cache actor that stores key-value pairs"""

    def __init__(self):
        self.cache = {}

    def on_start(self, actor_id: ActorId):
        print("[Cache] Started")

    def receive(self, msg: Message) -> Message:
        if msg.msg_type == "Get":
            key = msg.to_json().get("key", "")
            value = self.cache.get(key, None)
            return Message.from_json(
                "Value", {"key": key, "value": value, "found": value is not None}
            )
        elif msg.msg_type == "Set":
            data = msg.to_json()
            key = data.get("key", "")
            value = data.get("value", "")
            self.cache[key] = value
            return Message.from_json("SetResult", {"key": key, "success": True})
        elif msg.msg_type == "GetStats":
            return Message.from_json("Stats", {"cache_size": len(self.cache)})
        return Message.empty()


async def run_node(port: int, seed: str | None):
    """Run a node in the cluster"""
    print(f"\n{'='*60}")
    print(f"Pulsing Demo Service - Node on port {port}")
    print(f"{'='*60}\n")

    config = SystemConfig.with_addr(f"127.0.0.1:{port}")
    if seed:
        config = config.with_seeds([seed])
        print(f"Joining cluster via: {seed}")

    system = await create_actor_system(config)
    print(f"✓ System started: {system.node_id} @ {system.addr}\n")

    # Create different actors based on node role
    if seed is None:
        # Node 1: Create router and some workers
        print("Creating actors on node 1...")
        await system.spawn("router", RouterActor(), public=True)
        print("  ✓ actors/router")

        for i in range(1, 3):
            worker_name = f"worker-{i}"
            await system.spawn(worker_name, WorkerActor(worker_name), public=True)
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
            await system.spawn(worker_name, WorkerActor(worker_name), public=True)
            print(f"  ✓ actors/{worker_name}")
        print("\n✓ Node 2 ready!")

    elif port == 8002:
        # Node 3: Add cache
        await asyncio.sleep(1)
        print("Creating actors on node 3...")
        await system.spawn("cache", CacheActor(), public=True)
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
        await system.shutdown()
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
