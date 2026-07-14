"""Minimal isolated spawn: child process runs logic; cluster sees one bridge actor.

The worker connects out-of-cluster via ``Connect``; the parent registers
``IsolatedBridgeActor`` as the only cluster-visible actor.

Run::

    uv run python -m pulsing.examples.isolated_spawn_minimal

Or::

    uv run python examples/python/isolated_actor_spawn.py

The actor class lives in :mod:`pulsing.examples.isolated_spawn_payload` so it is
never bound to ``__main__`` (which would break unpickling in ``isolated_worker``).
"""

from __future__ import annotations

import asyncio
import os

import pulsing as pul
from pulsing.core.isolated_bridge import IsolatedSpawnHandle
from pulsing.core.proxy import ActorProxy
from pulsing.core.remote import _extract_methods

from pulsing.examples.isolated_spawn_payload import DemoWorker


async def main() -> None:
    await pul.init(addr="127.0.0.1:0")
    handle = await pul.spawn(
        DemoWorker(),
        new_process=True,
        name="demo_isolated",
        public=True,
        restart_policy="never",
    )
    if not isinstance(handle, IsolatedSpawnHandle):
        raise TypeError("expected IsolatedSpawnHandle from isolated spawn")

    methods, async_methods = _extract_methods(DemoWorker)
    proxy = ActorProxy(handle.ref, methods, async_methods)

    print("parent pid:", os.getpid())
    print("child pid (actor logic):", await proxy.pid())
    print("double(11) =", await proxy.double(11))

    if handle.process.returncode is None:
        handle.process.terminate()
        try:
            await asyncio.wait_for(handle.process.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            handle.process.kill()
            await handle.process.wait()

    await pul.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
