# SPDX-License-Identifier: Apache-2.0
"""Attach runtime state to a craft agent actor."""

from __future__ import annotations

import asyncio
from typing import Any

import pulsing as pul

from pulsing.agent.actors.activity import init_activity
from pulsing.agent.actors.log import init_log
from pulsing.agent.actors.session import AgentSession
from pulsing.agent.cluster.constants import short_agent_name, full_agent_name
from pulsing.agent.npc.config import NpcConfig
from pulsing.agent.actors.forge_runtime import init_forge_host
from pulsing.agent.loop.constants import (
    CLUSTER_TOOL_NAMES,
    FORGE_HOST_TOOL_NAMES,
    FORGE_ISOLATED_TOOL_NAMES,
    NPC_TOOL_NAMES,
    QUEST_TOOL_NAMES,
)
from pulsing.agent.loop.llm_chat import LlmChat
from pulsing.agent.loop.permissions import PermissionChecker
from pulsing.agent.loop.sandbox import normalize_policy
from pulsing.agent.loop.sandbox_manager import SandboxManager
from pulsing.agent.loop.split_tools import build_tools_for_agent

DEFAULT_SYSTEM_PROMPT = (
    "You are a coding agent with filesystem and shell tools. "
    "Prefer small, safe steps. Current working directory context is implied by the user."
)


def setup_agent(
    agent: Any, config: NpcConfig, *, stream_assistant: bool | None = None
) -> None:
    system_prompt, allow, forbid, npc_class_name, personality = (
        config.resolved_profile()
    )
    if stream_assistant is None:
        stream_assistant = False

    agent._provider = (config.provider or "anthropic").strip().lower()
    agent._api_key = config.api_key
    agent._base_url = config.base_url
    agent._model = config.model
    agent._cwd = config.cwd
    agent._auto_approve = config.auto_approve
    agent._system_prompt = (system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
    agent._summon_depth = max(0, int(config.summon_depth))
    agent._max_summon_depth = max(1, int(config.max_summon_depth))
    agent._prompt_callback = config.prompt_callback
    agent._stream_assistant = stream_assistant
    agent._sandbox_policy = normalize_policy(config.sandbox_policy)
    agent._dangerously_disable_sandbox = config.dangerously_disable_sandbox
    agent._sandbox_mgr = SandboxManager(
        agent._sandbox_policy,
        dangerously_disable_sandbox=config.dangerously_disable_sandbox,
    )
    agent._cluster_short_name = (
        short_agent_name(config.agent_name, workspace_id=config.workspace_id)
        if config.agent_name
        else ""
    )
    agent._workspace_id = (config.workspace_id or "").strip() or None
    agent._agent_role = (config.agent_role or "").strip()
    agent._agent_description = (config.agent_description or "").strip()
    agent._npc_class = npc_class_name
    agent._personality = personality
    agent._cluster_enabled = bool(agent._cluster_short_name) and bool(
        agent._workspace_id
    )
    agent._shared_tool_worker = bool(config.shared_tool_worker)
    agent._summon_enabled = bool(
        agent._workspace_id
        and agent._cluster_short_name
        and agent._summon_depth < agent._max_summon_depth
    )
    agent._lock = asyncio.Lock()
    agent._worker_spawn: pul.IsolatedSpawnHandle | None = None
    agent._worker_proxy: pul.ActorProxy | None = None
    agent._checker = PermissionChecker(
        auto_approve=config.auto_approve,
        prompt_callback=config.prompt_callback,
    )
    tool_list = build_tools_for_agent(
        agent._checker,
        cwd=config.cwd,
        cluster_enabled=agent._cluster_enabled,
        summon_enabled=agent._summon_enabled,
        tool_allowlist=set(allow) if allow else None,
        tool_forbid=set(forbid) if forbid else None,
    )
    agent._tools_by_name = {t.name: t for t in tool_list}
    agent._local_tools = {
        n: t
        for n, t in agent._tools_by_name.items()
        if n not in FORGE_ISOLATED_TOOL_NAMES
        and n not in FORGE_HOST_TOOL_NAMES
        and n not in CLUSTER_TOOL_NAMES
        and n not in NPC_TOOL_NAMES
        and n not in QUEST_TOOL_NAMES
    }
    agent._session = AgentSession(model=config.model)
    agent._forge_events: list[dict[str, Any]] = []
    agent._forge_stream_sink = None
    ws = (config.workspace_id or "").strip()
    short = agent._cluster_short_name
    agent._forge_host_name = (
        full_agent_name(short, workspace_id=ws) if ws and short else None
    )
    agent._event_sink_name = None
    agent._forge_actors_ready = False
    agent._forge_inbox_proxy = None
    agent._mcp_hub_name = None
    agent._code_cell_registry_name = None
    agent._initial_messages: list[dict] = []
    agent._llm: LlmChat | None = None
    agent._on_start_done = False
    agent._on_start_lock = asyncio.Lock()
    init_activity(agent)
    init_log(agent)
    init_forge_host(agent)
