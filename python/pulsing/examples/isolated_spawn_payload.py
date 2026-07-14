"""Picklable worker body for :mod:`pulsing.examples.isolated_spawn_minimal`.

Kept in its own module so it is never executed as ``__main__`` (which would
break pickling for ``python -m ...`` entrypoints).
"""

from __future__ import annotations

import os


class DemoWorker:
    """User logic executed in the child OS process."""

    def double(self, n: int) -> int:
        return n * 2

    def pid(self) -> int:
        return os.getpid()
