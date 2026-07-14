#!/usr/bin/env python3
"""Entry point for the minimal isolated-spawn demo.

Implementation (picklable actor class + ``async def main``) lives in
``pulsing.examples.isolated_spawn_minimal`` so the child worker can import it.

Run from repo root::

    uv run python examples/python/isolated_actor_spawn.py

Equivalent::

    uv run python -m pulsing.examples.isolated_spawn_minimal
"""

from __future__ import annotations

import asyncio

from pulsing.examples.isolated_spawn_minimal import main


if __name__ == "__main__":
    asyncio.run(main())
