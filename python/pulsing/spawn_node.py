"""Child-process entrypoint: join cluster using env-injected seeds.

Run via::

    PULSING_NODE_ADDR=127.0.0.1:0 PULSING_SEEDS=127.0.0.1:8000 \\
        python -m pulsing.spawn_node

Optional: ``PULSING_PASSPHRASE`` (same as parent for mTLS).

Used by :func:`pulsing.spawn` with ``new_process=True`` and ``actor=None`` (full child cluster node).
Isolated user actors use ``python -m pulsing.isolated_worker`` instead.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys


def _parse_seeds(raw: str) -> list[str] | None:
    seeds = [s.strip() for s in raw.split(",") if s.strip()]
    return seeds or None


async def _run() -> None:
    from pulsing import actor_system

    addr = os.environ.get("PULSING_NODE_ADDR")
    if not addr:
        print(
            "pulsing.spawn_node: set PULSING_NODE_ADDR (bind addr for this node)",
            file=sys.stderr,
        )
        raise SystemExit(2)

    seeds = _parse_seeds(os.environ.get("PULSING_SEEDS", ""))
    passphrase = os.environ.get("PULSING_PASSPHRASE") or None

    system = await actor_system(
        addr=addr,
        seeds=seeds,
        passphrase=passphrase,
    )
    print(
        f"pulsing.spawn_node: ready node_id={system.node_id} addr={system.addr}",
        flush=True,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, AttributeError):
            # Windows: no add_signal_handler for SIGTERM in some configs
            pass

    try:
        await stop.wait()
    finally:
        await system.shutdown()


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_run())


if __name__ == "__main__":
    main()
