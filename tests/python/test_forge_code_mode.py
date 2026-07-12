# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for Forge Code Mode (Python cell exec/wait)."""

from __future__ import annotations

from pathlib import Path

from pulsing.forge.handlers import dispatch_tool
from pulsing.forge.context import ToolCallContext


def _ctx(tmp_path: Path) -> ToolCallContext:
    return ToolCallContext(cwd=tmp_path)


def test_exec_completes_with_text(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    out = dispatch_tool("exec", {"source": 'text("hello from cell")\n'}, ctx=ctx)
    assert not out.is_error
    assert "hello from cell" in out.content
    assert out.structured is not None
    assert out.structured["kind"] == "result"
    assert out.structured["cell_id"].startswith("cell-")


def test_exec_yield_control(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    source = 'text("part1")\nyield_control()\ntext("part2")\n'
    out = dispatch_tool("exec", {"source": source}, ctx=ctx)
    assert not out.is_error
    assert out.structured is not None
    assert out.structured["kind"] == "yielded"
    assert "part1" in out.content
    assert "part2" not in out.content

    cell_id = out.structured["cell_id"]
    wait_out = dispatch_tool(
        "wait",
        {"cell_id": cell_id, "terminate": False},
        ctx=ctx,
    )
    assert not wait_out.is_error
    assert wait_out.structured is not None
    assert wait_out.structured["kind"] == "result"
    assert "part2" in wait_out.content


def test_exec_nested_read(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("forge code mode", encoding="utf-8")
    ctx = _ctx(tmp_path)
    source = f'text(tools.call("Read", {{"file_path": "{sample}"}}))\n'
    out = dispatch_tool("exec", {"source": source}, ctx=ctx)
    assert not out.is_error
    assert "forge code mode" in out.content


def test_exec_pragma_parsed(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    source = '# @exec: {"yield_time_ms": 5000}\ntext("ok")\n'
    out = dispatch_tool("exec", {"source": source}, ctx=ctx)
    assert not out.is_error
    assert "ok" in out.content


def test_wait_unknown_cell(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    out = dispatch_tool("wait", {"cell_id": "cell-missing"}, ctx=ctx)
    assert out.is_error
    assert "unknown cell_id" in out.content
