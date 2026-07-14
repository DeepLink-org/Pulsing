# SPDX-License-Identifier: Apache-2.0
"""Tests for pulsing.forge (independent of Craft)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.forge import (
    LocalToolRuntime,
    LocalToolSession,
    PlanItem,
    StepStatus,
    ToolResult,
    UpdatePlanArgs,
)


def test_local_read(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("hi", encoding="utf-8")
    rt = LocalToolRuntime(cwd=str(tmp_path))
    out = rt.call_tool("Read", {"file_path": str(p)})
    assert out == ToolResult(content="hi")


def test_local_glob(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("1", encoding="utf-8")
    rt = LocalToolRuntime(cwd=str(tmp_path))
    out = rt.call_tool("Glob", {"pattern": "*.py", "path": str(tmp_path)})
    assert "a.py" in out.content


def test_shell_command_codex_args(tmp_path: Path) -> None:
    rt = LocalToolRuntime(cwd=str(tmp_path))
    out = rt.call_tool(
        "shell_command",
        {"command": "echo hi", "workdir": str(tmp_path), "timeout_ms": 5000},
    )
    assert not out.is_error
    assert "hi" in out.content


def test_exec_command_session(tmp_path: Path) -> None:
    rt = LocalToolRuntime(cwd=str(tmp_path))
    out = rt.call_tool(
        "exec_command",
        {"cmd": "sleep 1 && echo done", "yield_time_ms": 300, "tty": False},
    )
    assert not out.is_error
    assert out.structured is not None


def test_exec_command_pty_and_streaming(tmp_path: Path) -> None:
    session = LocalToolSession()
    rt = LocalToolRuntime(cwd=str(tmp_path), session=session)
    out = rt.call_tool(
        "exec_command",
        {"cmd": "echo pty_ok", "yield_time_ms": 300, "tty": True},
    )
    assert not out.is_error
    assert out.structured is not None
    assert "pty_ok" in (out.structured.get("output") or "")
    assert session.exec_deltas


def test_runtime_close_kills_background_exec_sessions(tmp_path: Path) -> None:
    """Long-running exec_command sessions must not survive runtime teardown."""
    for tty in (False, True):
        rt = LocalToolRuntime(cwd=str(tmp_path))
        out = rt.call_tool(
            "exec_command",
            {"cmd": "sleep 30", "yield_time_ms": 100, "tty": tty},
        )
        assert out.structured is not None
        session_id = out.structured.get("session_id")
        assert session_id is not None
        session = rt.context.exec._sessions[session_id]
        if tty:
            assert session.pty is not None
            proc = session.pty.proc
            assert session.pty.poll() is None
        else:
            proc = session.proc
            assert proc is not None
            assert proc.poll() is None

        rt.close()

        assert proc.poll() is not None
        assert not rt.context.exec._sessions


def test_view_image_structured(tmp_path: Path) -> None:
    # minimal 1x1 PNG
    p = tmp_path / "x.png"
    p.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000a49444154789c630001000005000108d4a7e00000000049454e44ae426082"
        )
    )
    rt = LocalToolRuntime(cwd=str(tmp_path))
    out = rt.call_tool("view_image", {"path": str(p), "detail": "high"})
    assert not out.is_error
    assert out.structured is not None
    items = out.structured.get("content_items") or []
    assert items and items[0]["type"] == "input_image"
    assert str(items[0]["image_url"]).startswith("data:image/")


def test_apply_patch_add_file(tmp_path: Path) -> None:
    rt = LocalToolRuntime(cwd=str(tmp_path))
    patch = "*** Begin Patch\n*** Add File: hello.txt\n+hello world\n*** End Patch\n"
    out = rt.call_tool("apply_patch", {"patch": patch})
    assert not out.is_error
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hello world\n"


def test_update_plan_session(tmp_path: Path) -> None:
    session = LocalToolSession()
    rt = LocalToolRuntime(cwd=str(tmp_path), session=session)
    out = rt.call_tool(
        "update_plan",
        {"plan": [{"step": "one", "status": "pending"}]},
    )
    assert not out.is_error
    assert session.plan == [PlanItem(step="one", status=StepStatus.PENDING)]


def test_new_context_session(tmp_path: Path) -> None:
    session = LocalToolSession()
    rt = LocalToolRuntime(cwd=str(tmp_path), session=session)
    out = rt.call_tool("new_context", {})
    assert not out.is_error
    assert session.new_context_requested is True


def test_forge_environment_entrypoint(tmp_path: Path) -> None:
    from pulsing.forge import ForgeEnvironment

    env = ForgeEnvironment.ephemeral(cwd=str(tmp_path))
    out = env.runtime().call_tool(
        "shell_command", {"cmd": "echo ok", "workdir": str(tmp_path)}
    )
    assert not out.is_error
    assert "ok" in out.content


@pytest.mark.asyncio
async def test_tool_worker_actor_spawn() -> None:
    import pulsing as pul
    from pulsing.forge import ToolWorkerActor, ToolWorkerConfig

    await pul.init()
    try:
        worker = await ToolWorkerActor.spawn(
            config=ToolWorkerConfig(cwd="."),
            public=False,
        )
        pong = await worker.ping()
        assert pong.get("ok") is True
        out = await worker.Read(file_path="README.md")
        assert isinstance(out, dict)
        assert "content" in out
    finally:
        await pul.shutdown()
