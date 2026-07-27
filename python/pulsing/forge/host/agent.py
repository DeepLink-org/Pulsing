# SPDX-License-Identifier: Apache-2.0
"""Python client projection for the canonical Rust Forge Agent."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pulsing.forge.client import ForgeClient
from pulsing.forge.host.cli_events import CliEventSink
from pulsing.forge.result import ToolResult

DEFAULT_TOOL_NAMES: tuple[str, ...] = (
    "update_plan",
    "Glob",
    "Read",
    "Grep",
    "shell_command",
)

_DEFAULT_SYSTEM = """\
You are a capable coding agent with filesystem and shell tools.
Use tools to inspect the workspace before answering.
When multi-step work is needed, call update_plan first.
Be concise in final replies.\
"""


@dataclass
class ForgeAgent:
    """Client facade; Rust owns the Session, Agent loop, tools, and cancellation."""

    cwd: Path | str = "."
    provider: str = "demo"
    model: str = "demo"
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = 8192
    max_turns: int = 20
    tool_names: tuple[str, ...] = DEFAULT_TOOL_NAMES
    system_prompt: str = _DEFAULT_SYSTEM
    sandbox_policy: str = "off"
    auto_approve: bool = True
    events: CliEventSink = field(default_factory=CliEventSink)
    _client: ForgeClient | None = field(default=None, init=False, repr=False)
    _session_id: str | None = field(default=None, init=False, repr=False)
    _active_turn_id: str | None = field(default=None, init=False, repr=False)
    _messages: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.cwd = Path(self.cwd).resolve()
        if self.api_key is not None or self.base_url is not None:
            raise ValueError(
                "canonical ForgeAgent provider credentials are Rust process "
                "configuration; use OPENAI_API_KEY/OPENAI_BASE_URL or "
                "ANTHROPIC_API_KEY/ANTHROPIC_BASE_URL"
            )

    def _ensure_client(self) -> ForgeClient:
        if self._client is None:
            self._client = ForgeClient()
        return self._client

    def _ensure_session(self) -> str:
        if self._session_id is None:
            self._session_id = self._ensure_client().create_session(
                cwd=str(self.cwd),
                provider=self.provider,
                model=self.model,
                max_tokens=self.max_tokens,
                max_turns=self.max_turns,
                sandbox=self.sandbox_policy,
                auto_approve=self.auto_approve,
                tool_names=self.tool_names,
                system_prompt=self.system_prompt,
            )
        return self._session_id

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Client-side conversation projection; never used to drive execution."""
        return list(self._messages)

    @property
    def session(self) -> dict[str, Any]:
        """Read-only Rust Session snapshot."""
        return self._ensure_client().snapshot(self._ensure_session())

    async def run(self, prompt: str) -> str:
        session_id = self._ensure_session()
        self._messages.append({"role": "user", "content": prompt})
        receipt = await asyncio.to_thread(
            self._ensure_client().start_turn,
            session_id,
            prompt,
        )
        turn_id = str(receipt["turn_id"])
        self._active_turn_id = turn_id
        try:
            outcome = await asyncio.to_thread(
                self._ensure_client().wait_turn,
                session_id,
                turn_id,
                int(receipt["accepted_seq"]),
            )
        finally:
            self._active_turn_id = None

        self._project_events(list(outcome.get("events") or []))
        terminal = dict(outcome.get("terminal") or {})
        status = terminal.get("status")
        if status == "completed":
            return str(terminal.get("text") or "")
        if status == "cancelled":
            raise asyncio.CancelledError("Forge turn cancelled")
        message = str(terminal.get("message") or "Forge turn failed")
        raise RuntimeError(message)

    async def cancel(self) -> bool:
        if self._session_id is None or self._active_turn_id is None:
            return False
        await asyncio.to_thread(
            self._ensure_client().cancel_turn,
            self._session_id,
            self._active_turn_id,
        )
        return True

    def close(self) -> None:
        if (
            self._client is not None
            and self._session_id is not None
            and self._active_turn_id is not None
        ):
            try:
                self._client.cancel_turn(self._session_id, self._active_turn_id)
            except RuntimeError:
                pass
        self._active_turn_id = None
        self._session_id = None
        self._client = None

    def _project_events(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            kind = str(event.get("kind") or "")
            payload = dict(event.get("payload") or {})
            if kind == "turn_output_delta":
                self.events.on_assistant_delta(str(payload.get("delta") or ""))
            elif kind == "tool_started":
                name = str(payload.get("name") or "tool")
                self.events.on_tool_begin(name, {})
                self._messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "name": name}],
                    }
                )
            elif kind == "tool_completed":
                name = str(payload.get("name") or "tool")
                result = ToolResult(
                    content=str(payload.get("summary") or ""),
                    is_error=not bool(payload.get("ok")),
                )
                self.events.on_tool_end(name, result)
                self._messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "name": name,
                                "content": result.content,
                                "is_error": result.is_error,
                            }
                        ],
                    }
                )
            elif kind == "tool_cancelled":
                name = str(payload.get("name") or "tool")
                self.events.on_tool_end(
                    name,
                    ToolResult(content="cancelled", is_error=True),
                )
            elif kind == "turn_completed":
                text = str(payload.get("text") or "")
                self._messages.append({"role": "assistant", "content": text})
                self.events.on_assistant_end()
            elif kind == "turn_failed":
                self.events.on_error(str(payload.get("message") or "Forge turn failed"))


__all__ = ["DEFAULT_TOOL_NAMES", "ForgeAgent"]
