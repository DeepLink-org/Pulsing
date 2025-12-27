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

from pulsing.actor import Actor, ActorId, Message, SystemConfig, create_actor_system


class SharedCounter(Actor):
    def __init__(self, node_id: str):
        self.count = 0
        self.node_id = node_id

    def on_start(self, actor_id: ActorId):
        print(f"[{actor_id}] Started on {self.node_id}")

    def receive(self, msg: Message) -> Message:
        if msg.msg_type == "GetCount":
            return Message.from_json(
                "CountResponse", {"count": self.count, "from_node": self.node_id}
            )
        elif msg.msg_type == "Increment":
            n = msg.to_json().get("n", 1)
            self.count += n
            print(f"[{self.node_id}] +{n} -> {self.count}")
            return Message.from_json(
                "CountResponse", {"count": self.count, "from_node": self.node_id}
            )
        return Message.empty()


async def run_node(port: int, seed: str | None):
    print(f"=== Pulsing Cluster - Node {port} ===\n")

    config = SystemConfig.with_addr(f"127.0.0.1:{port}")
    if seed:
        config = config.with_seeds([seed])
        print(f"Joining via: {seed}")

    system = await create_actor_system(config)
    print(f"✓ Started: {system.node_id} @ {system.addr}\n")

    path = "services/counter"

    if seed is None:
        # Node 1: Create actor
        await system.spawn_named(
            path, "counter", SharedCounter(str(system.node_id)), public=True
        )
        print(f"✓ Created: {path}")
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
                actor = await system.resolve_named(path)
                break
            except Exception:
                print(".", end="", flush=True)
                await asyncio.sleep(0.5)

        if not actor:
            print("\n✗ Failed to resolve actor")
            return

        print("✓ Resolved\n")

        # Interact
        resp = (await actor.ask(Message.from_json("GetCount", {}))).to_json()
        print(f"Initial: {resp['count']} (from {resp['from_node']})")

        for i in range(1, 4):
            resp = (
                await actor.ask(Message.from_json("Increment", {"n": i * 10}))
            ).to_json()
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
