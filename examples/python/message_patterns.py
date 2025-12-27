#!/usr/bin/env python3
"""
Message Patterns Example

Three core patterns:
1. Single -> Single (RPC)
2. Single -> Stream (Server Streaming) - e.g., LLM token generation
3. Stream -> Single (Client Streaming) - limited in Python, uses batch

Usage: python examples/python/message_patterns.py
"""

import asyncio
import json

from pulsing.actor import (
    Actor,
    Message,
    StreamMessage,
    SystemConfig,
    create_actor_system,
)


class DemoActor(Actor):
    async def receive(self, msg: Message):
        if msg.msg_type == "Greet":
            # Pattern 1: RPC
            name = msg.to_json().get("name", "stranger")
            print(f"[Actor] Greet: {name}")
            return Message.from_json("Greeting", {"message": f"Hello, {name}!"})

        elif msg.msg_type == "CountTo":
            # Pattern 2: Server Streaming
            n = msg.to_json().get("n", 3)
            print(f"[Actor] CountTo: {n}")

            stream_msg, writer = StreamMessage.create("CountItem")

            async def produce():
                for i in range(1, n + 1):
                    await writer.write_json({"value": i})
                    await asyncio.sleep(0.05)
                writer.close()

            asyncio.create_task(produce())
            return stream_msg

        elif msg.msg_type == "Sum":
            # Pattern 3: Client Streaming (batch mode in Python)
            items = msg.to_json().get("items", [])
            print(f"[Actor] Sum: {items}")
            return Message.from_json("SumResult", {"total": sum(items)})

        return Message.empty()


async def main():
    print("=== Pulsing Message Patterns ===\n")

    system = await create_actor_system(SystemConfig.standalone())
    actor = await system.spawn("demo", DemoActor())

    # Pattern 1: RPC
    print("--- Pattern 1: RPC ---")
    resp = (await actor.ask(Message.from_json("Greet", {"name": "Pulsing"}))).to_json()
    print(f"Response: {resp['message']}\n")

    # Pattern 2: Server Streaming
    print("--- Pattern 2: Server Streaming ---")
    req = Message.from_json("CountTo", {"n": 3})
    response = await actor.ask(req)
    async for chunk in response.stream_reader():
        item = json.loads(chunk)
        print(f"Received: {item['value']}")
    print()

    # Pattern 3: Client Streaming (batch mode)
    print("--- Pattern 3: Client Streaming (batch) ---")
    resp = (
        await actor.ask(Message.from_json("Sum", {"items": [10, 20, 30]}))
    ).to_json()
    print(f"Sum: {resp['total']}\n")

    await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
