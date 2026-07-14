# SPDX-License-Identifier: Apache-2.0
"""write_stdin boundary tests — unknown session, empty/oversized input, concurrency."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from pulsing.forge.exec_output import MAX_STDIN_BYTES
from pulsing.testing.forge_harness import local_runtime

pytestmark = pytest.mark.forge


def test_write_stdin_unknown_session_returns_clear_error(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    out = rt.call_tool("write_stdin", {"session_id": 999, "chars": "hi"})
    assert out.is_error
    assert "999" in out.content


def test_write_stdin_missing_session_id(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    out = rt.call_tool("write_stdin", {"chars": "hi"})
    assert out.is_error


def test_write_stdin_invalid_session_id_type(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    out = rt.call_tool("write_stdin", {"session_id": "not-a-number", "chars": "hi"})
    assert out.is_error


def test_write_stdin_empty_input_on_live_session(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    started = rt.call_tool(
        "exec_command", {"cmd": "cat", "yield_time_ms": 300, "tty": True}
    )
    session_id = (started.structured or {}).get("session_id")
    assert session_id is not None
    try:
        out = rt.call_tool(
            "write_stdin", {"session_id": session_id, "chars": "", "yield_time_ms": 300}
        )
        assert not out.is_error
    finally:
        rt.call_tool("write_stdin", {"session_id": session_id, "chars": "\x03"})


def test_write_stdin_oversized_input_rejected(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    started = rt.call_tool(
        "exec_command", {"cmd": "cat", "yield_time_ms": 300, "tty": True}
    )
    session_id = (started.structured or {}).get("session_id")
    assert session_id is not None
    try:
        huge = "a" * (MAX_STDIN_BYTES + 1)
        out = rt.call_tool("write_stdin", {"session_id": session_id, "chars": huge})
        assert out.is_error
        assert "too large" in out.content
    finally:
        rt.call_tool("write_stdin", {"session_id": session_id, "chars": "\x03"})


def test_write_stdin_concurrent_writes_do_not_crash(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    started = rt.call_tool(
        "exec_command", {"cmd": "cat", "yield_time_ms": 300, "tty": True}
    )
    session_id = (started.structured or {}).get("session_id")
    assert session_id is not None

    errors: list[Exception] = []

    def _write(i: int) -> None:
        try:
            rt.call_tool(
                "write_stdin",
                {"session_id": session_id, "chars": f"{i}\n", "yield_time_ms": 250},
            )
        except Exception as exc:  # noqa: BLE001 — assert no exception escapes
            errors.append(exc)

    threads = [threading.Thread(target=_write, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    rt.call_tool("write_stdin", {"session_id": session_id, "chars": "\x03"})
    assert not errors


def test_write_stdin_rejects_non_tty_session(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    started = rt.call_tool(
        "exec_command", {"cmd": "sleep 5", "yield_time_ms": 300, "tty": False}
    )
    session_id = (started.structured or {}).get("session_id")
    assert session_id is not None
    out = rt.call_tool("write_stdin", {"session_id": session_id, "chars": "hi"})
    assert out.is_error
    assert "tty=true" in out.content
    rt.call_tool("write_stdin", {"session_id": session_id, "chars": "\x03"})


def test_write_stdin_rejects_utf8_oversized_by_bytes_not_chars(
    forge_workspace: Path,
) -> None:
    rt = local_runtime(forge_workspace)
    started = rt.call_tool(
        "exec_command", {"cmd": "cat", "yield_time_ms": 300, "tty": True}
    )
    session_id = (started.structured or {}).get("session_id")
    assert session_id is not None
    try:
        huge = "é" * (MAX_STDIN_BYTES // 2 + 1)
        assert len(huge) < MAX_STDIN_BYTES
        assert len(huge.encode("utf-8")) > MAX_STDIN_BYTES
        out = rt.call_tool("write_stdin", {"session_id": session_id, "chars": huge})
        assert out.is_error
        assert "too large" in out.content
    finally:
        rt.call_tool("write_stdin", {"session_id": session_id, "chars": "\x03"})


def test_write_stdin_rejects_unknown_session_after_exit(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    started = rt.call_tool(
        "exec_command", {"cmd": "cat", "yield_time_ms": 300, "tty": True}
    )
    session_id = (started.structured or {}).get("session_id")
    assert session_id is not None
    ended = rt.call_tool(
        "write_stdin", {"session_id": session_id, "chars": "\x03", "yield_time_ms": 500}
    )
    assert (ended.structured or {}).get("exit_code") is not None
    out = rt.call_tool("write_stdin", {"session_id": session_id, "chars": "hi"})
    assert out.is_error
    assert "unknown session_id" in out.content or "already exited" in out.content


def test_exec_command_rejects_workdir_escape(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    out = rt.call_tool(
        "exec_command",
        {"cmd": "echo hi", "workdir": "..", "yield_time_ms": 300, "tty": True},
    )
    assert out.is_error
    assert "outside working directory" in out.content
