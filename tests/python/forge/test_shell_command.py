# SPDX-License-Identifier: Apache-2.0
"""shell_command / exec_command sandbox + path boundary tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.testing.forge_harness import local_runtime

pytestmark = pytest.mark.forge

_RESTRICTED_PATH = "/usr/bin:/bin:/usr/local/bin"


def test_shell_command_missing_cmd(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    out = rt.call_tool("shell_command", {})
    assert out.is_error
    assert "missing cmd/command" in out.content


def test_shell_command_workdir_escape_rejected(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    out = rt.call_tool(
        "shell_command",
        {"command": "echo hi", "workdir": "../escape"},
    )
    assert out.is_error
    assert "outside working directory" in out.content


def test_shell_command_restricted_sandbox_limits_path(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace, sandbox_policy="restricted")
    out = rt.call_tool(
        "shell_command",
        {"command": "echo $PATH", "timeout_ms": 5000},
    )
    assert not out.is_error
    assert _RESTRICTED_PATH in out.content
    assert "restricted env" in out.content


def test_shell_command_login_restricted_still_sandboxed(forge_workspace: Path) -> None:
    """Login shells may source profile and widen PATH; assert wrapper label, not PATH."""
    rt = local_runtime(forge_workspace, sandbox_policy="restricted")
    out = rt.call_tool(
        "shell_command",
        {"command": "echo ok", "login": True, "timeout_ms": 5000},
    )
    assert not out.is_error
    assert "restricted env (env -i + bash -l)" in out.content
    assert "sandbox=off" not in out.content
