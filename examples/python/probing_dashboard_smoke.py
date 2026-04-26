#!/usr/bin/env python3
"""
Pulsing + probing dashboard demo with cascading actor messages.

Actors randomly forward messages to other actors, creating variable-length
call chains visible in the Trace Timeline.

Quick SQL check (no HTTP server)::

    uv run python examples/python/probing_dashboard_smoke.py --once

Live demo::

    uv run python examples/python/probing_dashboard_smoke.py --port 8765

Then open http://127.0.0.1:8765/pulsing to see the Trace Timeline.
Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys


def _row_estimate(payload: dict) -> int:
    size = payload.get("size")
    if isinstance(size, int) and size > 0:
        return size
    cols = payload.get("cols") or []
    if not cols:
        return 0
    c0 = cols[0]
    if isinstance(c0, dict):
        for _tag, seq in c0.items():
            if isinstance(seq, list):
                return len(seq)
    return 0


def _summarize_df(payload: dict) -> str:
    names = payload.get("names") or []
    return f"columns={names!r} rows≈{_row_estimate(payload)}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--once",
        action="store_true",
        help="Run a quick SQL smoke test and exit (no HTTP server)",
    )
    p.add_argument(
        "--port", type=int, default=8765, help="Probing HTTP port. Default: 8765"
    )
    p.add_argument("--actors", type=int, default=6, help="Number of actors. Default: 6")
    p.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between message bursts. Default: 2",
    )
    p.add_argument(
        "--max-depth", type=int, default=4, help="Max forwarding depth. Default: 4"
    )
    return p.parse_args()


async def _run_once() -> int:
    try:
        import probing  # noqa: F401
    except ImportError:
        print("error: probing is not installed.", file=sys.stderr)
        return 1

    import pulsing as pul
    from pulsing.integrations.probing import (
        refresh_probing_tables_async,
        stop_probing_integration,
    )

    @pul.remote
    class _SmokeActor:
        def __init__(self) -> None:
            pass

        def ping(self) -> str:
            return "ok"

    stop_probing_integration()
    await pul.init()
    try:
        await _SmokeActor.spawn(name="actors/smoke_demo")
        ok = await refresh_probing_tables_async()
        if not ok:
            print(
                "error: refresh_probing_tables_async() returned False", file=sys.stderr
            )
            return 2

        from probing import _core

        queries = [
            ("pulsing.members", "SELECT * FROM pulsing.members LIMIT 20"),
            ("pulsing.metrics", "SELECT * FROM pulsing.metrics LIMIT 30"),
            ("pulsing.actors", "SELECT * FROM pulsing.actors LIMIT 30"),
            ("pulsing.spans", "SELECT * FROM pulsing.spans LIMIT 30"),
        ]

        print("Pulsing ↔ probing smoke: OK\n")
        for title, sql in queries:
            raw = _core.query_json(sql)
            payload = json.loads(raw)
            print(f"--- {title} ---")
            print(_summarize_df(payload))
            cols = payload.get("cols") or []
            if cols and isinstance(cols[0], dict):
                for tag, seq in cols[0].items():
                    if isinstance(seq, list) and seq:
                        print(f"sample (first column {tag!r}): {seq[:3]!r}")
                        break
            print()
        return 0
    finally:
        await pul.shutdown()


async def _run_live(args: argparse.Namespace) -> int:
    try:
        import probing  # noqa: F401
    except ImportError:
        print("error: probing is not installed.", file=sys.stderr)
        return 1

    import pulsing as pul
    from pulsing.integrations.probing import stop_probing_integration

    # Registry: all actor proxies indexed by name, shared with the actor class.
    actor_names: list[str] = []
    actor_proxies: dict[str, object] = {}

    @pul.remote
    class CascadeActor:
        """Actor that randomly forwards messages to peers, creating call chains."""

        def __init__(self, label: str = "", max_depth: int = 4) -> None:
            self.label = label
            self.max_depth = max_depth

        async def cascade(self, depth: int = 0, origin: str = "") -> str:
            """Receive a message and optionally forward to random peers."""
            # Simulate own computation (counted as self time)
            await asyncio.sleep(random.uniform(0, 0.5))

            if depth >= self.max_depth:
                return f"{self.label}: leaf (depth={depth})"

            peers = [n for n in actor_names if n != self.label]
            fan_out = random.randint(0, min(2, len(peers)))
            targets = random.sample(peers, fan_out) if fan_out > 0 else []

            results = []
            for t in targets:
                proxy = actor_proxies.get(t)
                if proxy is not None:
                    try:
                        r = await proxy.cascade(depth=depth + 1, origin=self.label)
                        results.append(r)
                    except Exception:
                        pass

            return f"{self.label}: forwarded to {len(results)} peers at depth={depth}"

        def ping(self) -> str:
            return f"{self.label}: ok"

    stop_probing_integration()
    await pul.init()

    try:
        # Spawn actors
        for i in range(args.actors):
            name = f"actors/node_{i}"
            proxy = await CascadeActor.spawn(
                name=name,
                label=name,
                max_depth=args.max_depth,
            )
            actor_names.append(name)
            actor_proxies[name] = proxy

        async def _message_burst() -> None:
            """Periodically start cascading messages from random actors."""
            while True:
                # Pick 1–3 random actors to initiate a cascade
                initiators = random.sample(
                    actor_names, min(random.randint(1, 3), len(actor_names))
                )
                for name in initiators:
                    proxy = actor_proxies.get(name)
                    if proxy is not None:
                        try:
                            await proxy.cascade(depth=0, origin="external")
                        except Exception:
                            pass
                await asyncio.sleep(args.interval)

        burst_task = asyncio.create_task(_message_burst(), name="cascade-burst")

        host = "127.0.0.1"
        print()
        print("Pulsing + probing cascade demo running.")
        print(f"  Actors: {args.actors}, max chain depth: {args.max_depth}")
        print(f"  Web UI:  http://{host}:{args.port}/pulsing")
        print(f"  SQL:     http://{host}:{args.port}/analytics")
        print()
        print("Press Ctrl+C to stop.")
        print()

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        return 0
    finally:
        burst_task.cancel()
        try:
            await burst_task
        except asyncio.CancelledError:
            pass
        await pul.shutdown()


def main() -> None:
    args = _parse_args()
    if args.once:
        raise SystemExit(asyncio.run(_run_once()))

    os.environ.setdefault("PROBING_PORT", str(args.port))
    try:
        raise SystemExit(asyncio.run(_run_live(args)))
    except KeyboardInterrupt:
        print("\nStopped.")
        raise SystemExit(0) from None


if __name__ == "__main__":
    main()
