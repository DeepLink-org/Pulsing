# SPDX-License-Identifier: Apache-2.0
"""ForgeAgent — minimal coding-agent loop with Forge tools and CLI feedback."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pulsing.forge.host.llm import LLMClient, LLMMessage
from pulsing.forge.environment import ForgeEnvironment
from pulsing.forge.host.cli_events import CliEventSink
from pulsing.forge.hybrid_runtime import HybridForgeRuntime
from pulsing.forge.result import ToolResult
from pulsing.forge.rust_runtime import rust_forge_available
from pulsing.forge.runtime import LocalToolRuntime
from pulsing.forge.session import LocalToolSession
from pulsing.forge.tool_calls import (
    anthropic_tool_result_block,
    anthropic_tool_results_message,
    extract_tool_calls_anthropic,
    forge_tool_definitions,
)

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


def _text_from_content(content: list[Any]) -> str:
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    return "".join(parts)


@dataclass
class ForgeAgent:
    """Thin Host: LLM loop + Forge tools + default CLI event output.

    Example::

        agent = ForgeAgent(cwd=".", provider="demo")
        text = await agent.run("List README files in this project")
    """

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
    _messages: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    _runtime: HybridForgeRuntime | LocalToolRuntime | None = field(
        default=None, init=False, repr=False
    )
    _client: LLMClient | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cwd = Path(self.cwd).resolve()
        if self.provider == "openai" and not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            self.base_url = self.base_url or os.environ.get("OPENAI_BASE_URL")
        elif self.provider == "anthropic" and not self.api_key:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            self.base_url = self.base_url or os.environ.get("ANTHROPIC_BASE_URL")

    def _ensure_runtime(self) -> HybridForgeRuntime | LocalToolRuntime:
        if self._runtime is not None:
            return self._runtime
        session = LocalToolSession(token_budget=128_000)
        if rust_forge_available():
            self._runtime = HybridForgeRuntime.create(
                cwd=str(self.cwd),
                sandbox_policy=self.sandbox_policy,
                session=session,
                auto_approve=self.auto_approve,
                event_callback=self.events.on_forge_event,
                start_mcp=False,
            )
        else:
            self._runtime = ForgeEnvironment(
                cwd=str(self.cwd),
                sandbox_policy=self.sandbox_policy,
                session=session,
                auto_approve=self.auto_approve,
            ).runtime()
        return self._runtime

    def _ensure_client(self) -> LLMClient:
        if self._client is None:
            self._client = LLMClient(
                provider=self.provider,
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    @property
    def messages(self) -> list[dict[str, Any]]:
        return list(self._messages)

    @property
    def session(self) -> LocalToolSession:
        rt = self._ensure_runtime()
        return rt.python_runtime.session  # type: ignore[return-value]

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None

    async def run(self, prompt: str) -> str:
        """Run a multi-turn agent session until the model stops calling tools."""
        self._messages = []
        self._messages.append({"role": "user", "content": prompt})
        tools = forge_tool_definitions(list(self.tool_names))
        rt = self._ensure_runtime()
        final: LLMMessage | None = None

        for _ in range(self.max_turns):
            final = await self._stream_one_llm(self._messages, tools)
            self._messages.append({"role": "assistant", "content": list(final.content)})
            self.events.on_assistant_end()

            calls = extract_tool_calls_anthropic(final.content)
            if not calls:
                return _text_from_content(final.content)

            result_blocks = []
            for call in calls:
                result = await self._call_tool(rt, call.name, call.arguments)
                result_blocks.append(anthropic_tool_result_block(call.id, result))

            self._messages.append(anthropic_tool_results_message(result_blocks))

            if self.session.plan:
                self.events.on_plan_updated(
                    [item.to_dict() for item in self.session.plan]
                )

        text = _text_from_content(final.content) if final else ""
        return text or "(max turns reached)"

    async def _call_tool(
        self,
        rt: HybridForgeRuntime | LocalToolRuntime,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        self.events.on_tool_begin(name, arguments)
        result = await rt.acall_tool(name, arguments)
        self.events.on_tool_end(name, result)
        return result

    async def _stream_one_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMMessage:
        client = self._ensure_client()
        q: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _producer() -> None:
            try:
                stream = client.stream_messages(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self.system_prompt,
                    messages=messages,
                    tools=tools,
                )
                with stream as s:
                    for text in s.text_stream:
                        loop.call_soon_threadsafe(q.put_nowait, ("chunk", text))
                    loop.call_soon_threadsafe(
                        q.put_nowait, ("done", s.get_final_message())
                    )
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, ("error", exc))

        producer = asyncio.create_task(asyncio.to_thread(_producer))
        final: LLMMessage | None = None
        try:
            while True:
                kind, payload = await q.get()
                if kind == "chunk":
                    self.events.on_assistant_delta(str(payload))
                elif kind == "done":
                    final = payload
                    break
                elif kind == "error":
                    self.events.on_error(str(payload))
                    raise payload
        finally:
            await producer

        assert final is not None
        return final
