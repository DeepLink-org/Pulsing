#!/usr/bin/env python3
"""
Message Patterns - Common actor communication patterns

Usage: python examples/python/message_patterns.py
"""

import asyncio

from pulsing.actor import (
    Actor,
    Message,
    StreamMessage,
    SystemConfig,
    create_actor_system,
)


class PatternDemo(Actor):
    def __init__(self):
        self.value = 0

    async def receive(self, msg):
        # Pattern 1: Simple object messaging (dict, list, string, etc.)
        if isinstance(msg, dict):
            if msg.get("action") == "add":
                self.value += msg.get("n", 1)
                return {"value": self.value}
            if msg.get("action") == "get":
                return {"value": self.value}

        # Pattern 2: Streaming response (e.g., LLM token generation)
        if msg == "stream":
            stream_msg, writer = StreamMessage.create("tokens")

            async def produce():
                try:
                    for token in ["Hello", " ", "World", "!"]:
                        await writer.write({"token": token})
                        await asyncio.sleep(0.1)
                    await writer.close()
                except Exception:
                    pass  # Stream closed, ignore

            asyncio.create_task(produce())
            return stream_msg

        # Pattern 3: JSON Message (for Rust actor compatibility)
        if isinstance(msg, Message):
            return Message.from_json("Echo", {"received": msg.msg_type})

        return f"unknown: {msg}"


async def main():
    system = await create_actor_system(SystemConfig.standalone())
    actor = await system.spawn("demo", PatternDemo())

    # Pattern 1: Dict messages
    print("--- Dict Messages ---")
    print(await actor.ask({"action": "add", "n": 10}))  # {'value': 10}
    print(await actor.ask({"action": "add", "n": 5}))  # {'value': 15}
    print(await actor.ask({"action": "get"}))  # {'value': 15}

    # Pattern 2: Streaming (transparent Python objects)
    print("\n--- Streaming ---")
    response = await actor.ask("stream")
    async for chunk in response.stream_reader():
        print(chunk["token"], end="", flush=True)  # Directly a Python dict
    print()

    # Pattern 3: JSON Message (backward compatible)
    print("\n--- JSON Message ---")
    resp = await actor.ask(Message.from_json("Test", {"data": 123}))
    print(resp.to_json())  # {'received': 'Test'}

    await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
