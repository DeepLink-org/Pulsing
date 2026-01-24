"""Actor system lifecycle management."""

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
    """Actor system runtime context manager."""
    await init(addr=addr, seeds=seeds, passphrase=passphrase)
    try:
        yield get_system()
    finally:
        await shutdown()
