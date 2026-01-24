#!/usr/bin/env python3
"""
Cluster Example - Multi-node Communication

Demonstrates distributed actors with cross-node messaging.

Usage:
    Terminal 1: python examples/python/cluster.py --port 8000
    Terminal 2: python examples/python/cluster.py --port 8001 --seed 127.0.0.1:8000
"""

import argparse
import asyncio

import pulsing as pul


class SharedCounter:
    def __init__(self, node_id: str):
        self.count = 0
        self.node_id = node_id

    def on_start(self, actor_id):
        print(f"[{actor_id}] Started on {self.node_id}")

    async def receive(self, msg):
        if msg.get("action") == "get":
            return {"count": self.count, "from_node": self.node_id}
        elif msg.get("action") == "incr":
            n = msg.get("n", 1)
            self.count += n
            print(f"[{self.node_id}] +{n} -> {self.count}")
            return {"count": self.count, "from_node": self.node_id}
        return {"error": "unknown action"}


async def run_node(port: int, seed: str | None):
    print(f"=== Pulsing Cluster - Node {port} ===\n")

    addr = f"127.0.0.1:{port}"
    seeds = [seed] if seed else None

    system = await pul.actor_system(addr, seeds=seeds)
    print(f"✓ Started: {system.node_id} @ {system.addr}")
    if seed:
        print(f"  Joined via: {seed}")
    print()

    if seed is None:
        # Node 1: Create actor
        await system.spawn(
            SharedCounter(str(system.node_id)),
            name="counter",
        )
        print("✓ Created: counter")
        print("Start node 2: python cluster.py --port 8001 --seed 127.0.0.1:8000\n")

        try:
            while True:
                await asyncio.sleep(5)
                members = await system.members()
                print(f"Cluster: {len(members)} members")
        except asyncio.CancelledError:
            pass
        await system.shutdown()
    else:
        # Node 2: Join and interact
        await asyncio.sleep(2)

        # Resolve remote actor
        actor = None
        for _ in range(10):
            try:
                actor = await system.resolve("counter")
                break
            except Exception:
                print(".", end="", flush=True)
                await asyncio.sleep(0.5)

        if not actor:
            print("\n✗ Failed to resolve actor")
            return

        print("✓ Resolved\n")

        # Interact using simple Python dicts
        resp = await actor.ask({"action": "get"})
        print(f"Initial: {resp['count']} (from {resp['from_node']})")

        for i in range(1, 4):
            resp = await actor.ask({"action": "incr", "n": i * 10})
            print(f"After +{i * 10}: {resp['count']} (from {resp['from_node']})")

        print("\n✓ Done!")
        await system.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=str, default=None)
    args = parser.parse_args()

    try:
        asyncio.run(run_node(args.port, args.seed))
    except KeyboardInterrupt:
        print("\nInterrupted")


if __name__ == "__main__":
    main()
