# SPDX-License-Identifier: Apache-2.0
"""Workspace agent — single @remote actor."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from pulsing.core.remote import remote

from pulsing.agent.actors.actor import AgentActor
from pulsing.agent.actors.activity import get_activity, set_activity
from pulsing.agent.actors.log import append_log, get_logs
from pulsing.agent.actors.bootstrap import setup_agent
from pulsing.agent.cluster.constants import full_agent_name
from pulsing.agent.npc.config import NpcConfig

MessageChannel = Literal["say", "whisper"]


def format_incoming(sender: str, body: str, *, channel: MessageChannel) -> str:
    s = (sender or "peer").strip() or "peer"
    tag = f"{s} whispers" if channel == "whisper" else s
    return f"[{tag}]\n{body.strip()}"


@remote
class Agent(AgentActor):
    """Workspace agent. Spawn via :func:`pulsing.agent.npc.spawn_npc`."""

    def __init__(self, config: NpcConfig) -> None:
        setup_agent(self, config)

    def metadata(self) -> dict[str, str]:
        md: dict[str, str] = {
            "agent.kind": "workspace",
            "agent.name": self._cluster_short_name,
        }
        if self._npc_class:
            md["agent.class"] = self._npc_class
        if self._agent_role:
            md["agent.role"] = self._agent_role
        if self._workspace_id:
            md["agent.workspace_id"] = self._workspace_id
        return md

    def ping(self) -> dict[str, Any]:
        return {"ok": True, "kind": "npc", "name": self._cluster_short_name}

    def get_cluster_info(self) -> dict[str, Any]:
        short = self._cluster_short_name
        ws = self._workspace_id or ""
        return {
            "full_name": (
                full_agent_name(short, workspace_id=ws) if short and ws else None
            ),
            "workspace_id": ws or None,
            "role": self._agent_role,
            "model": self._model,
            "cwd": self._cwd,
            "kind": "npc",
            "name": short,
            "npc_class": self._npc_class,
            "summon_depth": self._summon_depth,
            "description": self._agent_description,
            "provider": self._provider,
            "cluster_enabled": self._cluster_enabled,
            "shared_tool_worker": self._shared_tool_worker,
        }

    def get_activity(self) -> dict[str, Any]:
        return get_activity(self)

    def get_logs(self, since: int = 0) -> dict[str, Any]:
        return get_logs(self, since=since)

    async def on_forge_event(self, event: dict[str, Any]) -> None:
        from pulsing.agent.actors.forge_events import handle_forge_event

        await handle_forge_event(self, event)

    async def on_forge_side_effect(self, event: dict[str, Any]) -> None:
        from pulsing.agent.actors.forge_events import apply_forge_side_effects

        await apply_forge_side_effects(self, event)

    async def on_forge_stream_event(self, event: dict[str, Any]) -> None:
        from pulsing.agent.actors.forge_events import apply_forge_side_effects

        await apply_forge_side_effects(self, event)

    async def resolve_exec_approval(self, request: dict[str, Any]) -> dict[str, Any]:
        checker = getattr(self, "_checker", None)
        if checker is None:
            return {"decision": "denied"}
        decision = checker.prompt_exec_approval(dict(request))
        return {"decision": decision}

    async def resolve_request_permissions(self, args: dict[str, Any]) -> dict[str, Any]:
        checker = getattr(self, "_checker", None)
        if checker is None:
            return {"permissions": {}, "scope": "turn", "strict_auto_review": False}
        return checker.prompt_request_permissions(dict(args))

    def get_forge_events(self, since: int = 0) -> list[dict[str, Any]]:
        inbox = getattr(self, "_forge_inbox_proxy", None)
        if inbox is not None:
            try:
                return inbox.get_forge_events(since)
            except Exception:
                pass
        events = getattr(self, "_forge_events", [])
        if since <= 0:
            return list(events)
        return events[since:]

    async def deliver_message(
        self,
        from_sender: str,
        message: str,
        *,
        channel: MessageChannel = "say",
        wait: bool = True,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        body = (message or "").strip()
        if not body:
            return {"ok": False, "error": "empty message"}
        who = (from_sender or "peer").strip() or "peer"
        ch: MessageChannel = "whisper" if channel == "whisper" else "say"
        line = format_incoming(who, body, channel=ch)
        append_log(self, f"← {ch} {who}: {body[:160]}")
        if not wait:
            set_activity(
                self, state="thinking", detail="queued message", from_sender=who
            )
            self.delayed(0).chat(line)
            return {"ok": True, "accepted": True, "from": who, "channel": ch}
        set_activity(self, state="thinking", detail="incoming message", from_sender=who)
        try:
            out = await asyncio.wait_for(self.chat(line), timeout=timeout)
        except asyncio.TimeoutError:
            set_activity(self, state="idle")
            append_log(self, f"✗ timed out after {timeout}s")
            return {
                "ok": False,
                "error": f"timed out after {timeout}s",
                "from": who,
                "channel": ch,
            }
        text = str(out.get("assistant_text") or "").strip()
        if text:
            append_log(self, f"→ {text[:240]}")
        return {**out, "from": who, "channel": ch}


NpcAgent = Agent
