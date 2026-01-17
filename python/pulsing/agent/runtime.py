"""Actor system lifecycle management"""

from __future__ import annotations

from contextlib import asynccontextmanager

from pulsing.actor import get_system, init, shutdown


@asynccontextmanager
async def runtime(
    *,
    addr: str | None = None,
    seeds: list[str] | None = None,
    passphrase: str | None = None,
):
    """
    Actor system runtime context manager.

    Example:
        from pulsing.agent import runtime

        async with runtime():
            agent = await MyAgent.spawn(name="agent")
            result = await agent.run()

        # Distributed mode
        async with runtime(addr="0.0.0.0:8001", seeds=["node1:8001"]):
            ...
    """
    await init(addr=addr, seeds=seeds, passphrase=passphrase)
    try:
        yield get_system()
    finally:
        await shutdown()
