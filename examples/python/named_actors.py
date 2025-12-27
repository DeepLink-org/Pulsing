#!/usr/bin/env python3
"""
Named Actors Example

Named actors can be discovered by service path (e.g., "services/echo")
instead of specific ActorId, enabling service discovery.

Usage: python examples/python/named_actors.py
"""

import asyncio

from pulsing.actor import Actor, ActorId, Message, SystemConfig, create_actor_system


class EchoActor(Actor):
    def on_start(self, actor_id: ActorId):
        print(f"[{actor_id}] Started")

    def receive(self, msg: Message) -> Message:
        message = msg.to_json().get("message", "")
        print(f"[Echo] {message}")
        return Message.from_json(
            "EchoResponse",
            {"echo": message, "actor": msg.to_json().get("_actor_id", "unknown")},
        )


async def main():
    print("=== Pulsing Named Actors ===\n")

    system = await create_actor_system(SystemConfig.standalone())
    print(f"✓ System started: {system.node_id}\n")

    # Create named actor
    path = "services/echo"
    await system.spawn_named(path, "echo", EchoActor(), public=True)
    print(f"✓ Created: {path} (local name: echo)\n")

    # Resolve by service path
    print("--- Resolve by path ---")
    actor = await system.resolve_named(path)
    resp = (await actor.ask(Message.from_json("Echo", {"message": "Hello!"}))).to_json()
    print(f"Response: {resp['echo']}\n")

    # List instances
    instances = await system.get_named_instances(path)
    print(f"Instances of '{path}': {len(instances)}")
    for i in instances:
        print(f"  {i['node_id']} @ {i['addr']} ({i['status']})")

    print("\n✓ Done!")
    await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
