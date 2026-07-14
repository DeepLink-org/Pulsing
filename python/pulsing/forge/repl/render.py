# SPDX-License-Identifier: Apache-2.0
"""Nushell-inspired fixed-width table rendering (no extra deps)."""

from __future__ import annotations


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "(empty)"
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    sep = " │ "
    header_line = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = "─┼─".join("─" * w for w in widths)
    body = [
        sep.join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in rows
    ]
    return "\n".join([header_line, rule, *body])
