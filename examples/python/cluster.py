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


@pul.remote
class SharedCounter:
    def __init__(self, node_id: str):
        self.count = 0
        self.node_id = node_id

    def get(self) -> dict[str, int | str]:
        return {"count": self.count, "from_node": self.node_id}

    def incr(self, n: int = 1) -> dict[str, int | str]:
        self.count += n
        print(f"[{self.node_id}] +{n} -> {self.count}")
        return {"count": self.count, "from_node": self.node_id}


async def run_node(port: int, seed: str | None):
    print(f"=== Pulsing Cluster - Node {port} ===\n")

    addr = f"127.0.0.1:{port}"
    seeds = [seed] if seed else None

    await pul.init(addr=addr, seeds=seeds)
    try:
        print(f"✓ Started: {addr}")
        if seed:
            print(f"  Joined via: {seed}")
        print()

        if seed is None:
            await SharedCounter.spawn(str(port), name="counter")
            print("✓ Created: counter")
            print("Start node 2: python cluster.py --port 8001 --seed 127.0.0.1:8000\n")

            try:
                while True:
                    await asyncio.sleep(5)
                    print("Cluster running... (press Ctrl+C to stop)")
            except asyncio.CancelledError:
                pass
        else:
            await asyncio.sleep(2)

            actor = None
            for _ in range(10):
                try:
                    actor = await SharedCounter.resolve("counter")
                    break
                except Exception:
                    print(".", end="", flush=True)
                    await asyncio.sleep(0.5)

            if not actor:
                print("\n✗ Failed to resolve actor")
                return

            print("✓ Resolved\n")

            resp = await actor.get()
            print(f"Initial: {resp['count']} (from {resp['from_node']})")

            for i in range(1, 4):
                resp = await actor.incr(i * 10)
                print(f"After +{i * 10}: {resp['count']} (from {resp['from_node']})")

            print("\n✓ Done!")
    finally:
        await pul.shutdown()


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
