"""Path B smoke: spawn a Python actor and round-trip a pickled message via ask()."""

import asyncio

import pulsing.core as core
from pulsing._core import Message


class Echo:
    def receive(self, msg):
        return msg


async def main():
    assert core.is_initialized()
    system = core.get_system()
    ref = await system.spawn(Echo())
    sent = Message("ping", b"hello")
    got = await ref.ask(sent)
    assert got.msg_type == "ping"
    assert got.payload == b"hello"
    print("cli_actor_smoke ok")


asyncio.run(main())
