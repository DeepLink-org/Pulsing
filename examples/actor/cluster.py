#!/usr/bin/env python3
"""
Cluster Actor Example

This example demonstrates how to create a distributed actor cluster
where actors on different nodes can communicate with each other.

Usage:
    # Terminal 1 - Start first node
    python examples/actor/cluster.py --port 8000
    
    # Terminal 2 - Start second node and join cluster
    python examples/actor/cluster.py --port 8001 --seed 127.0.0.1:8000
"""

import asyncio
import argparse
from pulsing.actor import (
    create_actor_system,
    SystemConfig,
    Message,
    Actor,
    ActorId,
    ActorRef,
)


class GreeterActor(Actor):
    """An actor that greets callers."""
    
    def __init__(self, node_name: str):
        self.node_name = node_name
        self.greet_count = 0
    
    def on_start(self, actor_id: ActorId):
        print(f"[{self.node_name}] GreeterActor started as '{actor_id.name}'")
    
    def receive(self, msg: Message) -> Message:
        if msg.msg_type == "Greet":
            data = msg.to_json()
            name = data.get("name", "stranger")
            self.greet_count += 1
            greeting = f"Hello {name} from {self.node_name}! (greet #{self.greet_count})"
            print(f"[{self.node_name}] Greeting: {greeting}")
            return Message.from_json("Greeting", {"message": greeting})
        elif msg.msg_type == "Stats":
            return Message.from_json("Stats", {
                "node": self.node_name,
                "greet_count": self.greet_count
            })
        else:
            return Message.empty()


async def run_node(port: int, seed_addr: str | None = None):
    """Run an actor system node."""
    node_name = f"Node-{port}"
    print(f"\n{'=' * 50}")
    print(f"Starting {node_name}")
    print(f"{'=' * 50}\n")
    
    # Create configuration
    config = SystemConfig.with_addr(f"0.0.0.0:{port}")
    if seed_addr:
        print(f"Joining cluster via seed: {seed_addr}")
        config = config.with_seeds([seed_addr])
    
    # Create actor system
    system = await create_actor_system(config)
    print(f"✓ System started at {system.addr}")
    print(f"  Node ID: {system.node_id}")
    
    # Spawn local greeter actor
    greeter = await system.spawn("greeter", GreeterActor(node_name))
    print(f"✓ Spawned local greeter: {greeter}")
    
    # If we joined a cluster, try to call the remote greeter
    if seed_addr:
        print("\nWaiting for cluster discovery...")
        await asyncio.sleep(2)  # Wait for gossip to propagate
        
        members = await system.members()
        print(f"\nCluster members ({len(members)}):")
        for m in members:
            print(f"  - {m['node_id']} @ {m['addr']} ({m['status']})")
        
        # Try to greet through local greeter
        print("\nSending greetings through local actor...")
        response = await greeter.ask_json("Greet", {"name": "Cluster User"})
        print(f"Local response: {response['message']}")
    
    print(f"\n[{node_name}] Running... Press Ctrl+C to stop.\n")
    
    # Keep running and periodically show stats
    try:
        while True:
            await asyncio.sleep(5)
            members = await system.members()
            print(f"[{node_name}] Active members: {len(members)}, Local actors: {system.local_actor_names()}")
    except asyncio.CancelledError:
        pass
    finally:
        print(f"\n[{node_name}] Shutting down...")
        await system.shutdown()
        print(f"[{node_name}] Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="Cluster Actor Example")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--seed", type=str, default=None, help="Seed node address to join")
    args = parser.parse_args()
    
    try:
        asyncio.run(run_node(args.port, args.seed))
    except KeyboardInterrupt:
        print("\nInterrupted by user")


if __name__ == "__main__":
    main()

