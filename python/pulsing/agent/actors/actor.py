# SPDX-License-Identifier: Apache-2.0
"""Workspace agent actor base: LLM session + tools. Mailbox serializes."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pulsing.agent.actors.activity import set_activity
from pulsing.agent.actors.tool_host import call_tool, stop_isolated_worker
from pulsing.agent.loop.llm_chat import LlmChat

logger = logging.getLogger(__name__)


class AgentActor:
    """Shared actor surface: ``chat`` / ``chat_stream`` + ``call_tool``."""

    async def _emit_text_chunk(self, chunk: str) -> None:
        if chunk:
            print(chunk, end="", flush=True)

    def _reply(self, out: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": out.get("ok", True),
            "events": out.get("events", []),
            "session_id": self._session.session_id,
            "assistant_text": out.get("assistant_text", ""),
            "streamed_assistant": bool(out.get("streamed_assistant")),
        }

    async def on_start(self, actor_id) -> None:
        async with self._on_start_lock:
            if self._on_start_done:
                return
            from pulsing.forge.setup_actors import ensure_forge_actors

            await ensure_forge_actors(self)
            cb = self._emit_text_chunk if self._stream_assistant else None
            self._llm = LlmChat.from_agent(self, text_stream_callback=cb)
            if self._initial_messages:
                self._llm.set_messages(self._initial_messages)
            self._on_start_done = True
            set_activity(self, state="idle")

    async def on_stop(self) -> None:
        await stop_isolated_worker(self)

    def get_session_id(self) -> str:
        return self._session.session_id

    async def call_tool(self, name: str, kwargs: dict[str, Any]) -> Any:
        from pulsing.agent.loop.tool_base import ToolResult

        return await call_tool(self, name, kwargs)

    async def chat(
        self,
        message: str,
        *,
        stream_assistant: bool | None = None,
    ) -> dict[str, Any]:
        if stream_assistant is not None:
            self._stream_assistant = bool(stream_assistant)
            if self._llm is not None:
                cb = self._emit_text_chunk if self._stream_assistant else None
                self._llm.set_stream_assistant(self._stream_assistant, cb)
        text = (message or "").strip()
        if not text:
            return {
                "ok": True,
                "events": [],
                "session_id": self._session.session_id,
                "assistant_text": "",
                "streamed_assistant": False,
            }
        if self._llm is None:
            return {"ok": False, "error": "agent not ready (on_start pending)"}
        set_activity(self, state="thinking", detail="LLM turn")
        try:
            out = await self._llm.respond(text)
        finally:
            set_activity(self, state="idle")
        return self._reply(out)

    async def chat_stream(
        self,
        message: str,
        *,
        stream_assistant: bool | None = None,
    ):
        if stream_assistant is not None:
            self._stream_assistant = bool(stream_assistant)
            if self._llm is not None:
                cb = self._emit_text_chunk if self._stream_assistant else None
                self._llm.set_stream_assistant(self._stream_assistant, cb)
        text = (message or "").strip()
        if not text:
            yield {"kind": "_final", **self._reply({"ok": True, "events": []})}
            return
        if self._llm is None:
            yield {"kind": "error", "message": "agent not ready (on_start pending)"}
            return

        q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def sink(ev: dict[str, Any]) -> None:
            await q.put(ev)

        self._forge_stream_sink = sink

        async def worker() -> None:
            try:
                out = await self._llm.respond(text, event_sink=sink)
                await q.put({"kind": "_final", **out})
            except BaseException as e:
                await q.put(
                    {
                        "kind": "_final",
                        "ok": False,
                        "error": repr(e),
                        "events": [],
                        "assistant_text": "",
                        "streamed_assistant": False,
                    }
                )
            finally:
                await q.put(None)

        task = asyncio.create_task(worker())
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                if item.get("kind") == "_final":
                    item.update(self._reply(item))
                yield item
        finally:
            self._forge_stream_sink = None
            await task
