# SPDX-License-Identifier: Apache-2.0
"""Workspace Agent: spawn, RPC."""

from __future__ import annotations

import inspect
import uuid
from pathlib import Path

import pytest

from pulsing.agent.cluster.constants import full_agent_name
from pulsing.agent.npc.config import NpcConfig


def _spawn_config(
    *,
    cwd: str,
    ws: str,
    name: str,
    public: bool = True,
) -> tuple[NpcConfig, str]:
    config = NpcConfig(
        model="claude-sonnet-4-20250514",
        cwd=cwd,
        agent_name=name,
        workspace_id=ws,
        auto_approve=True,
    )
    return config, full_agent_name(
        name, workspace_id=ws
    ) if public else f"craft_agent_{uuid.uuid4().hex[:12]}"


def test_imports_without_anthropic_at_module_level() -> None:
    from pulsing.agent.actors import Agent

    assert Agent is not None


def test_chat_stream_is_async_generator() -> None:
    from pulsing.agent.actors import Agent

    assert inspect.isasyncgenfunction(Agent._cls.chat_stream)


@pytest.mark.asyncio
async def test_spawn_and_get_session_id() -> None:
    import pulsing as pul
    from pulsing.agent.actors import Agent

    await pul.init()
    try:
        config, name = _spawn_config(cwd=".", ws="local", name="a", public=False)
        agent = await Agent.spawn(config=config, name=name, public=False)
        sid = await agent.get_session_id()
        assert len(sid) >= 8
    finally:
        await pul.shutdown()


@pytest.mark.asyncio
async def test_deliver_message_rejects_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import pulsing as pul
    from pulsing.agent.actors import Agent
    from pulsing.agent.workspace.config import default_config, save_config

    monkeypatch.chdir(tmp_path)
    cfg = default_config(tmp_path)
    save_config(cfg)
    ws = cfg.cluster_id

    await pul.init()
    try:
        name = f"rx-{uuid.uuid4().hex[:8]}"
        config, full = _spawn_config(cwd=str(tmp_path), ws=ws, name=name)
        agent = await Agent.spawn(config=config, name=full, public=True)
        await asyncio.sleep(0.3)
        out = await agent.deliver_message(from_sender="peer", message="")
        assert out.get("ok") is False
    finally:
        await pul.shutdown()


@pytest.mark.asyncio
async def test_deliver_message_wait_false_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import pulsing as pul
    from pulsing.agent.actors import Agent
    from pulsing.agent.workspace.config import default_config, save_config

    monkeypatch.chdir(tmp_path)
    cfg = default_config(tmp_path)
    save_config(cfg)
    ws = cfg.cluster_id

    await pul.init()
    try:
        name = f"wf-{uuid.uuid4().hex[:8]}"
        config, full = _spawn_config(cwd=str(tmp_path), ws=ws, name=name)
        agent = await Agent.spawn(config=config, name=full, public=True)
        await asyncio.sleep(0.3)
        out = await agent.deliver_message(
            from_sender="peer",
            message="hi",
            channel="whisper",
            wait=False,
        )
        assert out.get("ok") is True
        assert out.get("accepted") is True
        assert out.get("channel") == "whisper"
    finally:
        await pul.shutdown()
