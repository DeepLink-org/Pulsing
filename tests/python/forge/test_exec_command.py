# SPDX-License-Identifier: Apache-2.0
"""exec_command sandbox, workdir boundary, and PTY cleanup tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.testing.forge_harness import local_runtime

pytestmark = pytest.mark.forge


def test_exec_command_missing_cmd(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    out = rt.call_tool("exec_command", {"yield_time_ms": 300})
    assert out.is_error
    assert "missing cmd/command" in out.content


def test_exec_command_rejects_workdir_escape(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    out = rt.call_tool(
        "exec_command",
        {"cmd": "echo hi", "workdir": "..", "yield_time_ms": 300, "tty": False},
    )
    assert out.is_error
    assert "outside working directory" in out.content


def test_exec_command_rejects_tty_with_restricted_sandbox(
    forge_workspace: Path,
) -> None:
    rt = local_runtime(forge_workspace, sandbox_policy="restricted")
    out = rt.call_tool(
        "exec_command",
        {"cmd": "echo hi", "yield_time_ms": 300, "tty": True},
    )
    assert out.is_error
    assert "cannot use sandbox policy" in out.content


def test_exec_command_pipe_mode_runs_under_restricted_sandbox(
    forge_workspace: Path,
) -> None:
    rt = local_runtime(forge_workspace, sandbox_policy="restricted")
    out = rt.call_tool(
        "exec_command",
        {"cmd": "echo $PATH", "yield_time_ms": 500, "tty": False},
    )
    assert not out.is_error
    output = (out.structured or {}).get("output") or ""
    assert "/usr/bin:/bin:/usr/local/bin" in output


def test_exec_command_close_kills_background_sessions(forge_workspace: Path) -> None:
    for tty in (False, True):
        rt = local_runtime(forge_workspace)
        out = rt.call_tool(
            "exec_command",
            {"cmd": "sleep 30", "yield_time_ms": 100, "tty": tty},
        )
        session_id = (out.structured or {}).get("session_id")
        assert session_id is not None
        session = rt.context.exec._sessions[session_id]
        proc = session.pty.proc if session.pty is not None else session.proc
        assert proc is not None
        assert proc.poll() is None

        rt.close()
        assert proc.poll() is not None
        assert not rt.context.exec._sessions
