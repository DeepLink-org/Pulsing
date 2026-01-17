"""
🚀 Pulsing Quick Start - First Multi-Agent Application

Run: python examples/quickstart/hello_agent.py

You will see two agents greeting each other!
"""

import asyncio
from pulsing.actor import remote, resolve
from pulsing.agent import runtime


@remote
class Greeter:
    """A simple greeting agent"""

    def __init__(self, display_name: str):
        self.display_name = display_name
        print(f"✨ [{self.display_name}] is online")

    def greet(self, message: str) -> str:
        """Receive a greeting"""
        print(f"📨 [{self.display_name}] received message: {message}")
        return f"Hello! I'm {self.display_name}"

    async def say_hello_to(self, peer_name: str) -> str:
        """Greet another agent"""
        peer = await resolve(peer_name)
        print(f"👋 [{self.display_name}] is greeting [{peer_name}]...")
        reply = await peer.greet(f"Hi, I'm {self.display_name}!")
        print(f"💬 [{self.display_name}] received reply: {reply}")
        return reply


async def main():
    print("=" * 50)
    print("🎉 Pulsing Multi-Agent Quick Start")
    print("=" * 50)

    async with runtime():
        # Create two agents
        alice = await Greeter.spawn(display_name="Alice", name="alice")
        bob = await Greeter.spawn(display_name="Bob", name="bob")

        print("\n--- Agents greeting each other ---\n")

        # Alice greets Bob
        await alice.say_hello_to("bob")

        print()

        # Bob greets Alice
        await bob.say_hello_to("alice")

    print("\n" + "=" * 50)
    print("✅ Done! You've created your first Multi-Agent application")
    print("=" * 50)
    print("\nNext steps:")
    print(
        "  - Try MBTI discussion: python examples/agent/pulsing/mbti_discussion.py --mock"
    )
    print(
        "  - Try parallel ideas:   python examples/agent/pulsing/parallel_ideas_async.py --mock"
    )


if __name__ == "__main__":
    asyncio.run(main())
