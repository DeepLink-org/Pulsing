# SPDX-License-Identifier: Apache-2.0
"""Tests for pulsing.forge.host (ForgeAgent + CliEventSink)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from pulsing.forge import ForgeClient, LegacyPythonForgeAgent
from pulsing.forge.host import ForgeAgent
from pulsing.forge.host.cli_events import CliEventSink
from pulsing.forge.result import ToolResult

pytestmark = pytest.mark.forge


def test_default_and_legacy_agents_have_explicit_ownership() -> None:
    assert ForgeAgent is not LegacyPythonForgeAgent
    assert "_runtime" not in ForgeAgent.__dataclass_fields__
    assert "_client" in ForgeAgent.__dataclass_fields__
    assert "_runtime" in LegacyPythonForgeAgent.__dataclass_fields__


def test_native_forge_client_exposes_versioned_events(tmp_path: Path) -> None:
    client = ForgeClient()
    session_id = client.create_session(
        cwd=str(tmp_path),
        provider="demo",
        model="demo",
    )
    receipt = client.start_turn(session_id, "hello")
    outcome = client.wait_turn(
        session_id,
        str(receipt["turn_id"]),
        int(receipt["accepted_seq"]),
    )

    events = outcome["events"]
    assert outcome["terminal"]["status"] == "completed"
    assert events[-1]["protocol"] == "forge.event"
    assert events[-1]["version"] == {"major": 1, "minor": 0}
    assert events[-1]["kind"] == "turn_completed"
    assert events[-1]["payload"]["text"] == "(demo) Noted: hello"
    assert client.snapshot(session_id)["spec"]["approval_policy"] == "always"


def test_canonical_agent_rejects_python_owned_provider_credentials() -> None:
    with pytest.raises(ValueError, match="Rust process configuration"):
        ForgeAgent(provider="openai", api_key="python-owned-secret")


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


@pytest.mark.asyncio
async def test_forge_agent_reuses_one_rust_session_across_turns(
    tmp_path: Path,
) -> None:
    agent = ForgeAgent(
        cwd=tmp_path,
        provider="demo",
        events=CliEventSink(stream_assistant=False),
    )
    try:
        first = await agent.run("remember alpha")
        first_snapshot = agent.session
        second = await agent.run("remember beta")
        second_snapshot = agent.session

        assert first == "(demo) Noted: remember alpha"
        assert second == "(demo) Noted: remember beta"
        assert first_snapshot["id"] == second_snapshot["id"]
        assert len(first_snapshot["turns"]) == 1
        assert len(second_snapshot["turns"]) == 2
        assert [message["role"] for message in agent.messages] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
    finally:
        agent.close()
