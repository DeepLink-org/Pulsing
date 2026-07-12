"""Actors used by isolated-spawn tests; module path must stay stable for ``pickle``.

Plain user class (no ``@remote``) so the class object in ``sys.modules`` stays picklable
across pytest's import cycles; execution still goes through ``_WrappedActor`` in the worker.
"""

from __future__ import annotations


class IsoMathActor:
    """Minimal actor body for subprocess IPC round-trip tests."""

    def mul(self, a: int, b: int) -> int:
        return a * b
