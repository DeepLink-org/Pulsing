# SPDX-License-Identifier: Apache-2.0
"""Permission checks."""

from __future__ import annotations

from pulsing.agent.loop.permissions import PermissionChecker
from pulsing.agent.loop.tools_pkg import ReadTool, WriteTool


def test_auto_approve_allows_write() -> None:
    c = PermissionChecker(auto_approve=True)
    w = WriteTool()
    assert c.check(w, {"file_path": "/tmp/x", "content": "y"}) == "allow"


def test_read_only_allowed_without_auto_approve() -> None:
    c = PermissionChecker(auto_approve=False)
    r = ReadTool()
    assert c.check(r, {"file_path": "/etc/hosts"}) == "allow"


def test_write_denied_without_auto_approve() -> None:
    c = PermissionChecker(auto_approve=False)
    w = WriteTool()
    assert c.check(w, {"file_path": "/tmp/x", "content": "y"}) == "deny"


def test_prompt_callback_once() -> None:
    calls = {"n": 0}

    def cb(_tool: str, _summary: str) -> str:
        calls["n"] += 1
        return "once" if calls["n"] == 1 else "deny"

    c = PermissionChecker(prompt_callback=cb)
    w = WriteTool()
    assert c.check(w, {"file_path": "/tmp/x", "content": "y"}) == "allow"
    assert c.check(w, {"file_path": "/tmp/x", "content": "z"}) == "deny"
