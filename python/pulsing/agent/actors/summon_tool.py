# SPDX-License-Identifier: Apache-2.0
"""Summon tool: spawn NPC and optional whisper round-trip."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pulsing as pul

from pulsing.agent.cluster.constants import full_agent_name
from pulsing.agent.loop.tool_base import ToolResult


def _parse_wait(kwargs: dict[str, Any]) -> bool:
    wait = kwargs.get("wait", True)
    if isinstance(wait, str):
        return wait.strip().lower() not in ("0", "false", "no", "off")
    return bool(wait)


async def tool_summon(agent: Any, kwargs: dict[str, Any]) -> ToolResult:
    if agent._summon_depth >= agent._max_summon_depth:
        return ToolResult(content="Summon: max depth reached.", is_error=True)
    goal = str(kwargs.get("goal") or kwargs.get("task") or "").strip()
    if not goal:
        return ToolResult(content="Summon: goal required.", is_error=True)
    ws = agent._workspace_id
    if not ws:
        return ToolResult(content="Summon: no workspace.", is_error=True)

    child = str(kwargs.get("name") or "").strip() or f"sub-{uuid.uuid4().hex[:6]}"
    task_id = str(kwargs.get("task_id") or "").strip() or f"s-{uuid.uuid4().hex[:10]}"
    npc_class = str(kwargs.get("npc_class") or "artisan").strip()
    wait = _parse_wait(kwargs)
    timeout = float(kwargs.get("timeout", 600.0))
    try:
        from pulsing.agent.actors import Agent
        from pulsing.agent.npc import spawn_npc
        from pulsing.agent.npc.config import NpcConfig

        cfg = NpcConfig(
            model=str(kwargs.get("model") or agent._model),
            cwd=agent._cwd,
            provider=agent._provider,
            api_key=agent._api_key,
            base_url=agent._base_url,
            auto_approve=agent._auto_approve,
            sandbox_policy=agent._sandbox_policy,
            dangerously_disable_sandbox=agent._dangerously_disable_sandbox,
            agent_name=child,
            workspace_id=ws,
            npc_class=npc_class,
            personality=str(kwargs.get("personality") or ""),
            summon_depth=agent._summon_depth + 1,
            max_summon_depth=agent._max_summon_depth,
            shared_tool_worker=agent._shared_tool_worker,
        )
        await spawn_npc(cfg, public=True)
        from_name = agent._cluster_short_name or "parent"
        peer = await pul.resolve(
            full_agent_name(child, workspace_id=ws),
            cls=Agent,
            timeout=120.0,
        )
        result = await peer.deliver_message(
            from_name,
            goal,
            channel="whisper",
            wait=wait,
            timeout=timeout,
        )
    except Exception as e:
        return ToolResult(content=f"Summon failed: {e!r}", is_error=True)

    ok = result.get("ok", True)
    status = "completed" if ok and wait else ("accepted" if ok else "failed")
    text = str(result.get("assistant_text") or result.get("error") or "")[:8000]
    return ToolResult(
        content=json.dumps(
            {
                "task_id": task_id,
                "npc": child,
                "class": npc_class,
                "status": status,
                "wait": wait,
                "goal": goal,
                "assistant_text": text,
            },
            ensure_ascii=False,
        ),
        is_error=not ok,
    )
