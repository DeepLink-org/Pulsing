"""
Pulsing Ping-Pong Example using @remote decorator
Same functionality as AutoGen version
"""

from pulsing.actor import remote, runtime


# Define Agent
@remote
class PingPongAgent:
    async def ping(self, message: str) -> str:
        return f"pong: {message}"


# Run
async def main():
    async with runtime():
        agent = await PingPongAgent.spawn(name="pingpong")
        response = await agent.ping("hello")
        print(f"Received: {response}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
