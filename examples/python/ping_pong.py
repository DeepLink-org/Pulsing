#!/usr/bin/env python3
"""
Ping-Pong Example - Basic Actor Communication

Demonstrates:
- Actor lifecycle (on_start, on_stop)
- Request-response (ask) and fire-and-forget (tell) patterns

Usage: python examples/python/ping_pong.py
"""

import asyncio

from pulsing.actor import Actor, ActorId, Message, SystemConfig, create_actor_system


class Counter(Actor):
    def __init__(self):
        self.count = 0

    def on_start(self, actor_id: ActorId):
        print(f"[{actor_id}] Started with count: {self.count}")

    def on_stop(self):
        print(f"Stopped with count: {self.count}")

    def receive(self, msg: Message) -> Message:
        if msg.msg_type == "Ping":
            value = msg.to_json().get("value", 1)
            self.count += value
            print(f"  Ping({value}) -> count = {self.count}")
            return Message.from_json("Pong", {"result": self.count})
        elif msg.msg_type == "GetCount":
            return Message.from_json("Count", {"count": self.count})
        return Message.empty()


async def main():
    print("=== Pulsing Ping-Pong Example ===\n")

    system = await create_actor_system(SystemConfig.standalone())
    actor = await system.spawn("counter", Counter())
    print("✓ System started, actor spawned\n")

    # Request-Response (ask)
    print("--- Request-Response (ask) ---")
    for i in range(1, 4):
        msg = Message.from_json("Ping", {"value": i * 10})
        resp = (await actor.ask(msg)).to_json()
        print(f"Ping({i * 10}) -> Pong({resp['result']})")

    # Fire-and-Forget (tell)
    print("\n--- Fire-and-Forget (tell) ---")
    await actor.tell(Message.from_json("Ping", {"value": 100}))
    print("Sent Ping(100) without waiting")
    await asyncio.sleep(0.05)

    resp = (await actor.ask(Message.from_json("GetCount", {}))).to_json()
    print(f"Final count: {resp['count']}\n")

    await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
