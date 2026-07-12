# SPDX-License-Identifier: Apache-2.0
"""Follow scroll output."""

from __future__ import annotations

from pulsing.cli.agent.commands.follow import should_emit


def test_should_emit_delta() -> None:
    assert should_emit("", "a", delta=True) is True
    assert should_emit("a", "a", delta=True) is False
    assert should_emit("a", "b", delta=True) is True
    assert should_emit("a", "a", delta=False) is True
