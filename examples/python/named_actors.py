#!/usr/bin/env python3
"""
Named Actors Example

Named actors can be discovered by name instead of specific ActorId,
enabling service discovery.

Usage: python examples/python/named_actors.py
"""

import asyncio

import pulsing as pul


class EchoActor:
    """Simple echo actor that can be discovered by name."""

    def on_start(self, actor_id):
        print(f"[{actor_id}] Started")

    async def receive(self, msg):
        # Accept dict messages
        message = msg.get("message", "") if isinstance(msg, dict) else str(msg)
        print(f"[Echo] {message}")
        return {"echo": message}


async def main():
    print("=== Pulsing Named Actors ===\n")

    system = await pul.actor_system()
    print(f"✓ System started: {system.node_id}\n")

    # Create named actor (named actors are discoverable via resolve)
    await system.spawn(EchoActor(), name="echo")
    print("✓ Created: echo (named, discoverable)\n")

    # Resolve by name
    print("--- Resolve by name ---")
    actor = await system.resolve("echo")
    resp = await actor.ask({"message": "Hello!"})
    print(f"Response: {resp['echo']}\n")

    # List instances
    instances = await system.get_named_instances("actors/echo")
    print(f"Instances of 'actors/echo': {len(instances)}")
    for i in instances:
        print(f"  {i['node_id']} @ {i['addr']} ({i['status']})")

    print("\n✓ Done!")
    await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
