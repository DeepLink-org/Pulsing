# SPDX-License-Identifier: Apache-2.0
"""Cluster activity watch."""

from __future__ import annotations

import time

from pulsing.agent.cluster.activity import format_activity_table


def test_format_activity_table_empty() -> None:
    assert "no agents" in format_activity_table([]).lower()


def test_format_activity_table_busy_first() -> None:
    now = time.time()
    text = format_activity_table(
        [
            {
                "name": "guide",
                "npc_class": "artisan",
                "state": "tool",
                "from": "player",
                "tool": "Read",
                "detail": "src/a.py",
                "since": now - 3,
            },
            {
                "name": "scout",
                "npc_class": "scholar",
                "state": "idle",
                "from": "",
                "detail": "",
                "since": now - 120,
            },
        ],
    )
    assert "guide" in text
    assert "Read" in text
    assert "scout" in text
    assert "idle" in text
