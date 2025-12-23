#!/usr/bin/env python3
"""
Ping-Pong Actor Example

This example demonstrates basic actor communication using the Pulsing Actor System.

Usage:
    python examples/actor/ping_pong.py
"""

import asyncio
from pulsing.actor import (
    create_actor_system,
    SystemConfig,
    Message,
    Actor,
    ActorId,
)


class PingPongActor(Actor):
    """A simple actor that responds to Ping messages with Pong."""
    
    def __init__(self):
        self.count = 0
    
    def on_start(self, actor_id: ActorId):
        print(f"✓ Actor '{actor_id.name}' started on node {actor_id.node}")
    
    def on_stop(self):
        print(f"✓ Actor stopped with final count: {self.count}")
    
    def receive(self, msg: Message) -> Message:
        """Handle incoming messages."""
        if msg.msg_type == "Ping":
            data = msg.to_json()
            value = data.get("value", 1)
            self.count += value
            print(f"  Received Ping({value}), count is now {self.count}")
            return Message.from_json("Pong", {"result": self.count})
        elif msg.msg_type == "GetCount":
            return Message.from_json("Count", {"count": self.count})
        else:
            print(f"  Unknown message type: {msg.msg_type}")
            return Message.empty()


class AsyncPingPongActor(Actor):
    """An async version of the ping-pong actor."""
    
    def __init__(self):
        self.count = 0
    
    def on_start(self, actor_id: ActorId):
        print(f"✓ Async Actor '{actor_id.name}' started")
    
    async def receive(self, msg: Message) -> Message:
        """Handle incoming messages asynchronously."""
        # Simulate some async work
        await asyncio.sleep(0.01)
        
        if msg.msg_type == "Ping":
            data = msg.to_json()
            value = data.get("value", 1)
            self.count += value
            print(f"  [Async] Received Ping({value}), count is now {self.count}")
            return Message.from_json("Pong", {"result": self.count})
        else:
            return Message.empty()


async def main():
    print("=" * 50)
    print("Pulsing Actor System - Ping Pong Example")
    print("=" * 50)
    print()
    
    # Create actor system in standalone mode
    config = SystemConfig.standalone()
    system = await create_actor_system(config)
    print(f"✓ Actor system started at {system.addr}")
    print(f"  Node ID: {system.node_id}")
    print()
    
    # Spawn synchronous actor
    print("Spawning sync actor...")
    sync_actor = await system.spawn("counter", PingPongActor())
    print(f"  {sync_actor}")
    print()
    
    # Send some ping messages using ask_json
    print("Sending Ping messages to sync actor:")
    for i in range(1, 4):
        response = await sync_actor.ask_json("Ping", {"value": i * 10})
        print(f"  → Response: Pong(result={response['result']})")
    print()
    
    # Spawn async actor
    print("Spawning async actor...")
    async_actor = await system.spawn("async_counter", AsyncPingPongActor())
    print(f"  {async_actor}")
    print()
    
    # Send messages to async actor
    print("Sending Ping messages to async actor:")
    for i in range(1, 4):
        response = await async_actor.ask_json("Ping", {"value": i * 100})
        print(f"  → Response: Pong(result={response['result']})")
    print()
    
    # Fire-and-forget message using tell
    print("Sending tell (fire-and-forget) to sync actor...")
    await sync_actor.tell_json("Ping", {"value": 1000})
    await asyncio.sleep(0.1)  # Give time for processing
    
    # Get final count
    response = await sync_actor.ask_json("GetCount", {})
    print(f"  Final count: {response['count']}")
    print()
    
    # List local actors
    print(f"Local actors: {system.local_actor_names()}")
    print()
    
    # Shutdown
    print("Shutting down...")
    await system.shutdown()
    print("✓ Done!")


if __name__ == "__main__":
    asyncio.run(main())

