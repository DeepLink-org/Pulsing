# SPDX-License-Identifier: Apache-2.0
"""Control-flow exceptions for Python code cells."""


class CellYield(Exception):
    """Raised by ``yield_control()`` to pause the cell (Codex yield semantics)."""


class CellExit(Exception):
    """Raised by ``exit()`` to finish the cell successfully."""
