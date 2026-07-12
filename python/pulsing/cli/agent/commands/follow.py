# SPDX-License-Identifier: Apache-2.0
"""Append-only follow loops for dashboard panes."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from datetime import datetime

PrintFn = Callable[[], Awaitable[str]]


def should_emit(prev: str, text: str, *, delta: bool) -> bool:
    return (not delta) or (text != prev)


async def follow_output(
    produce: PrintFn,
    *,
    interval: float,
    scroll: bool = True,
    delta: bool = True,
) -> None:
    """Poll ``produce``; append to terminal (scroll) instead of full-screen refresh."""
    prev = ""
    sec = max(0.1, interval)
    try:
        while True:
            text = (await produce()).rstrip()
            if should_emit(prev, text, delta=delta):
                stamp = datetime.now().strftime("%H:%M:%S")
                print(f"── {stamp} ──")
                print(text)
                print(flush=True)
                prev = text
            elif not scroll:
                # fixed viewport: overwrite last block (unused today)
                pass
            await asyncio.sleep(sec)
    except asyncio.CancelledError:
        raise
    except KeyboardInterrupt:
        if scroll:
            print("\n(stopped)")
