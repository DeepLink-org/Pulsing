#!/usr/bin/env python3
"""
Named Actors Example

Named actors can be discovered by name instead of specific ActorId,
enabling service discovery.

Usage: python examples/python/named_actors.py
"""

import asyncio

import pulsing as pul
from pulsing.actor import Actor, ActorId, Message


class EchoActor(Actor):
    def on_start(self, actor_id: ActorId):
        print(f"[{actor_id}] Started")

    async def receive(self, msg: Message) -> Message:
        message = msg.to_json().get("message", "")
        print(f"[Echo] {message}")
        return Message.from_json(
            "EchoResponse",
            {"echo": message, "actor": msg.to_json().get("_actor_id", "unknown")},
        )


async def main():
    print("=== Pulsing Named Actors ===\n")

    system = await pul.actor_system()
    print(f"✓ System started: {system.node_id}\n")

    # Create named public actor
    await system.spawn(EchoActor(), name="echo", public=True)
    print("✓ Created: echo (public=True)\n")

    # Resolve by name
    print("--- Resolve by name ---")
    actor = await system.resolve("echo")
    resp = (await actor.ask(Message.from_json("Echo", {"message": "Hello!"}))).to_json()
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
