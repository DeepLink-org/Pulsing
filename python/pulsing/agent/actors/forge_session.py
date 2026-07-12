# SPDX-License-Identifier: Apache-2.0
"""Agent host ToolSession — plan state + permission prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pulsing.forge.discovery.catalog import ToolCatalog
from pulsing.forge.p2p_session import EmitFn, P2PToolSession
from pulsing.forge.session import PlanItem, UpdatePlanArgs


@dataclass
class AgentForgeSession(P2PToolSession):
    """Host-side session: local state + P2P forge events."""

    user_input: Any = None
    plugin_install: Any = None
    plan: list[PlanItem] = field(default_factory=list)
    new_context_requested: bool = False
    token_budget: int | None = None
    context_window: int = 200_000
    tool_catalog: ToolCatalog = field(default_factory=ToolCatalog)

    def update_plan(self, args: UpdatePlanArgs) -> None:
        self.plan = list(args.plan)
        super().update_plan(args)

    def request_new_context(self) -> None:
        self.new_context_requested = True
        super().request_new_context()

    def tokens_remaining(self) -> int | None:
        if self.token_budget is not None:
            return self.token_budget
        agent = getattr(self, "_agent", None)
        llm = getattr(agent, "_llm", None) if agent is not None else None
        if llm is not None:
            return llm.estimate_tokens_remaining(self.context_window)
        return None

    def request_plugin_install(self, args: dict[str, Any]) -> bool:
        cb = self.plugin_install
        if cb is not None:
            return bool(cb(args))
        return super().request_plugin_install(args)


def build_agent_forge_session(agent: Any, emit: EmitFn | None) -> AgentForgeSession:
    checker = getattr(agent, "_checker", None)
    user_input_cb = getattr(checker, "prompt_user_input", None) if checker else None
    plugin_cb = getattr(checker, "prompt_plugin_install", None) if checker else None
    session = AgentForgeSession(
        emit=emit,
        user_input=user_input_cb,
        plugin_install=plugin_cb,
    )
    session.tool_catalog.load_codex_plugins()
    session._agent = agent  # noqa: SLF001 — host back-ref for token estimate
    return session
