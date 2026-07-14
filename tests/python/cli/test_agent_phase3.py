# SPDX-License-Identifier: Apache-2.0
"""Phase 3: CLI commands live in pulsing.cli.agent."""

from __future__ import annotations

from pulsing.cli.agent.commands.follow import should_emit
from pulsing.cli.agent.helpers import DEFAULT_PROG as AGENT_DEFAULT_PROG


def test_agent_default_prog() -> None:
    assert AGENT_DEFAULT_PROG == "pulsing agent"


def test_should_emit_delta() -> None:
    assert should_emit("a", "a", delta=True) is False
    assert should_emit("a", "b", delta=True) is True
    assert should_emit("a", "a", delta=False) is True
