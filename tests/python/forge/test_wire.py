# SPDX-License-Identifier: Apache-2.0
"""L2 Forge wire checks — structured output and side effects."""

from __future__ import annotations

import json

import pytest

from pulsing.forge.context import ToolCallContext
from pulsing.forge.handlers import dispatch_tool
from pulsing.testing.forge_harness import run_wire_check, wire_check_tools

pytestmark = [pytest.mark.forge, pytest.mark.forge_l2]


@pytest.mark.parametrize("tool", sorted(wire_check_tools()))
def test_l2_wire_local(local_forge, forge_workspace, tool: str) -> None:
    run_wire_check(local_forge, forge_workspace, tool)


@pytest.mark.parametrize("tool", sorted(wire_check_tools()))
def test_l2_wire_hybrid(hybrid_forge, forge_workspace, tool: str) -> None:
    run_wire_check(hybrid_forge, forge_workspace, tool)


def test_get_context_remaining_unknown_without_session_budget(forge_workspace) -> None:
    """No ToolSession (defaults to NullToolSession) → degrade to status="unknown"."""
    ctx = ToolCallContext(cwd=str(forge_workspace))
    out = dispatch_tool("get_context_remaining", {}, ctx=ctx)
    assert not out.is_error
    payload = json.loads(out.content)
    assert payload == {"tokens_remaining": None, "status": "unknown"}
