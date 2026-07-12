# SPDX-License-Identifier: Apache-2.0
"""Attach unified Forge host runtime to workspace Agent."""

from __future__ import annotations

from typing import Any

from pulsing.agent.actors.forge_events import make_host_emit
from pulsing.agent.actors.forge_session import build_agent_forge_session
from pulsing.forge.backend import ForgeHostConfig, create_host_runtime
from pulsing.forge.events import ForgeEvent
from pulsing.forge.integrated import ForgeHostLink
from pulsing.forge.rust_runtime import RUST_FORGE_AVAILABLE


def _tokens_remaining(agent: Any) -> int | None:
    session = getattr(agent, "_forge_session", None)
    if session is not None:
        return session.tokens_remaining()
    return None


def init_forge_host(agent: Any) -> None:
    emit = make_host_emit(agent)
    session = build_agent_forge_session(agent, emit)
    agent._forge_session = session

    checker = getattr(agent, "_checker", None)
    user_input_cb = getattr(checker, "prompt_user_input", None) if checker else None
    exec_cb = getattr(checker, "prompt_exec_approval", None) if checker else None
    perms_cb = getattr(checker, "prompt_request_permissions", None) if checker else None
    plugin_cb = getattr(checker, "prompt_plugin_install", None) if checker else None
    auto_approve = bool(getattr(agent, "_auto_approve", False))

    host_cfg = ForgeHostConfig(
        cwd=agent._cwd,
        sandbox_policy=agent._sandbox_policy,
        dangerously_disable_sandbox=agent._dangerously_disable_sandbox,
        auto_approve=auto_approve,
        session=session,
    )

    if RUST_FORGE_AVAILABLE:
        agent._forge_host = create_host_runtime(
            host_cfg,
            event_callback=lambda raw: emit(ForgeEvent.from_dict(raw)),
            user_input_callback=user_input_cb,
            exec_approval_callback=exec_cb,
            request_permissions_callback=perms_cb,
            tokens_remaining_callback=lambda: _tokens_remaining(agent),
            plugin_install_callback=plugin_cb,
        )
    else:
        agent._forge_host = ForgeHostLink(
            cwd=agent._cwd,
            sandbox_policy=agent._sandbox_policy,
            dangerously_disable_sandbox=agent._dangerously_disable_sandbox,
            session=session,
            emit=emit,
        )

    agent._forge_worker = None
