# SPDX-License-Identifier: Apache-2.0
"""Tests for pulsing.forge.host (ForgeAgent + CliEventSink)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from pulsing.forge.host import ForgeAgent
from pulsing.forge.host.cli_events import CliEventSink
from pulsing.forge.result import ToolResult

pytestmark = pytest.mark.forge


def test_cli_event_sink_tool_lines() -> None:
    out = io.StringIO()
    sink = CliEventSink(out=out, err=out)
    sink.on_tool_begin("Glob", {"pattern": "*.md"})
    sink.on_tool_end("Glob", ToolResult(content="a.md"))
    text = out.getvalue()
    assert "→ Glob" in text
    assert "← Glob [ok]" in text


@pytest.mark.asyncio
async def test_forge_agent_demo_glob(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# hi\n", encoding="utf-8")
    out = io.StringIO()
    agent = ForgeAgent(
        cwd=tmp_path,
        provider="demo",
        events=CliEventSink(out=out, stream_assistant=False),
    )
    try:
        answer = await agent.run("please glob project files")
        assert "Glob" in out.getvalue() or answer
        assert len(agent.messages) >= 3
    finally:
        agent.close()


@pytest.mark.asyncio
async def test_forge_agent_demo_read_then_answer(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# demo project\n", encoding="utf-8")
    agent = ForgeAgent(
        cwd=tmp_path,
        provider="demo",
        events=CliEventSink(stream_assistant=False),
    )
    try:
        await agent.run("read readme and summarize")
        roles = [m["role"] for m in agent.messages]
        assert "assistant" in roles
        assert "user" in roles
    finally:
        agent.close()
