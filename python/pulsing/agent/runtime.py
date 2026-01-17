"""Actor 系统生命周期管理"""

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
    Actor 系统运行时上下文管理器。

    Example:
        from pulsing.agent import runtime

        async with runtime():
            agent = await MyAgent.spawn(name="agent")
            result = await agent.run()

        # 分布式模式
        async with runtime(addr="0.0.0.0:8001", seeds=["node1:8001"]):
            ...
    """
    await init(addr=addr, seeds=seeds, passphrase=passphrase)
    try:
        yield get_system()
    finally:
        await shutdown()
