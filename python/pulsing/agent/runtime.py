"""Actor system lifecycle management."""

from __future__ import annotations

from contextlib import asynccontextmanager

from pulsing.core import get_system, init, shutdown


@asynccontextmanager
async def runtime(
    *,
    addr: str | None = None,
    seeds: list[str] | None = None,
    passphrase: str | None = None,
    head_addr: str | None = None,
    is_head_node: bool = False,
):
    """Actor system runtime context manager."""
    await init(
        addr=addr,
        seeds=seeds,
        passphrase=passphrase,
        head_addr=head_addr,
        is_head_node=is_head_node,
    )
    try:
        yield get_system()
    finally:
        await shutdown()
