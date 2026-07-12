# SPDX-License-Identifier: Apache-2.0
"""LLM multi-turn chat loop: streaming in a worker thread; tools via :class:`AgentToolBackend`."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from pulsing.agent.loop.compact import maybe_compact
from pulsing.agent.loop.llm_blocks import (
    AbortedError,
    block_id as _block_id,
    block_input as _block_input,
    block_name as _block_name,
    block_type as _block_type,
    tool_result_dict as _tool_result_dict,
)
from pulsing.agent.loop.llm_client import LLMClient
from pulsing.agent.loop.permissions import PermissionChecker
from pulsing.agent.loop.tool_base import Tool, ToolResult

_MAX_RETRIES = 5
_BASE_DELAY = 0.5
_MAX_DELAY = 16.0
_JITTER_FACTOR = 0.25


def _compute_retry_delay(attempt: int) -> float:
    delay = min(_BASE_DELAY * (2**attempt), _MAX_DELAY)
    jitter = delay * random.uniform(0, _JITTER_FACTOR)
    return delay + jitter


@runtime_checkable
class AgentToolBackend(Protocol):
    async def call_tool(self, name: str, kwargs: dict[str, Any]) -> ToolResult: ...


TextStreamCallback = Callable[[str], Awaitable[None] | None]

# Optional per-event hook for :meth:`respond` (e.g. ``chat_stream`` over RPC).
StreamEventSink = Callable[[dict[str, Any]], Any]


class LlmChat:
    """LLM chat loop: optional incremental assistant streaming via thread → ``asyncio.Queue``.

    Anthropic ``messages.stream`` stays synchronous; a worker thread pushes text chunks into
    an asyncio queue while the event loop consumes chunks and invokes ``text_stream_callback``.
    """

    def __init__(
        self,
        backend: AgentToolBackend,
        tools: dict[str, Tool],
        permission_checker: PermissionChecker,
        *,
        provider: str = "anthropic",
        model: str,
        max_tokens: int = 8192,
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str = "",
        session_store: Any | None = None,
        stream_assistant: bool = True,
        text_stream_callback: TextStreamCallback | None = None,
    ) -> None:
        self._backend = backend
        self._tools = tools
        self._permissions = permission_checker
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        self._client = LLMClient(provider=provider, api_key=api_key, base_url=base_url)
        self._system_prompt = system_prompt
        self._messages: list[dict[str, Any]] = []
        self._aborted = False
        self._user_msg_start: int | None = None
        self._session_store = session_store
        self._stream_assistant = stream_assistant
        self._text_stream_callback = text_stream_callback

    @classmethod
    def from_agent(
        cls, agent: Any, *, text_stream_callback: TextStreamCallback | None
    ) -> LlmChat:
        if agent._provider == "demo":
            ak = None
            bu = None
        elif agent._provider == "openai":
            ak = agent._api_key or os.environ.get("OPENAI_API_KEY")
            bu = agent._base_url or os.environ.get("OPENAI_BASE_URL")
        else:
            ak = agent._api_key or os.environ.get("ANTHROPIC_API_KEY")
            bu = agent._base_url or os.environ.get("ANTHROPIC_BASE_URL")

        return cls(
            agent,
            agent._tools_by_name,
            agent._checker,
            provider=agent._provider,
            model=agent._model,
            api_key=ak,
            base_url=bu,
            system_prompt=agent._system_prompt,
            session_store=agent._session,
            stream_assistant=agent._stream_assistant,
            text_stream_callback=text_stream_callback,
        )

    def get_messages(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def set_messages(self, messages: list[dict[str, Any]]) -> None:
        self._messages = [
            {"role": m["role"], "content": m.get("content", "")} for m in messages
        ]

    def _persist(self, message: dict[str, Any]) -> None:
        if self._session_store is not None:
            try:
                self._session_store.append_message(message)
            except Exception:
                pass

    def rollback_pending(self) -> None:
        if self._user_msg_start is not None:
            del self._messages[self._user_msg_start :]
            self._user_msg_start = None

    def set_stream_assistant(
        self,
        enabled: bool,
        text_stream_callback: TextStreamCallback | None,
    ) -> None:
        self._stream_assistant = enabled
        self._text_stream_callback = text_stream_callback

    def set_system_prompt(self, text: str) -> None:
        """Update system / role instructions for subsequent LLM calls."""
        self._system_prompt = (text or "").strip()

    def register_tool(self, tool: Tool) -> None:
        """Add a deferred/discovered tool for subsequent LLM turns."""
        self._tools[tool.name] = tool

    def estimate_tokens_remaining(self, context_window: int = 200_000) -> int | None:
        """Rough token budget estimate from transcript size."""
        used = 0
        for msg in self._messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                used += max(1, len(content) // 4)
            elif isinstance(content, list):
                used += max(1, len(json.dumps(content, default=str)) // 4)
        remaining = context_window - used - self._max_tokens
        return max(0, remaining)

    def append_synthetic_user_message(self, content: str) -> None:
        """Append a user message (e.g. inter-agent note) before the next LLM round."""
        msg = {"role": "user", "content": content}
        self._messages.append(msg)
        self._persist(msg)

    def _sync_llm_round(self) -> tuple[Any | None, list[Any], list[str]]:
        """Run one assistant generation (with retries) in a sync context.

        Returns ``(final, tool_uses, sidecar_errors)`` where ``sidecar_errors`` are
        non-fatal notices (e.g. max_tokens truncation).
        """
        sidecar: list[str] = []
        final = None
        tool_uses: list[Any] = []

        for attempt in range(_MAX_RETRIES):
            tool_uses = []
            try:
                self._messages = maybe_compact(self._messages)
                tool_schemas = [t.to_api_schema() for t in self._tools.values()]
                stream_obj = self._client.stream_messages(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=self._system_prompt,
                    tools=tool_schemas,
                    messages=self._messages,
                )
                with stream_obj as stream:
                    acc: list[str] = []
                    for text in stream.text_stream:
                        if self._aborted:
                            raise AbortedError
                        acc.append(text)
                    _ = "".join(acc)
                    if self._aborted:
                        raise AbortedError
                    final_msg = stream.get_final_message()
                    final = final_msg
                    if final.stop_reason == "max_tokens":
                        sidecar.append("Response truncated: max_tokens.")
                    for block in final.content:
                        if _block_type(block) == "tool_use":
                            tool_uses.append(block)
                break
            except AbortedError:
                raise
            except Exception as e:
                if self._client.is_authentication_error(e):
                    self._messages.pop()
                    raise _FatalLLM(str(e), pop_user=True, kind="auth") from e
                if self._client.is_retryable_error(e):
                    if attempt < _MAX_RETRIES - 1:
                        wait = _compute_retry_delay(attempt)
                        sidecar.append(
                            f"API error, retrying in {wait:.1f}s… ({self._client.error_message(e)})",
                        )
                        time.sleep(wait)
                        continue
                    self._messages.pop()
                    raise _FatalLLM(
                        f"API error after {_MAX_RETRIES} retries: {self._client.error_message(e)}",
                        pop_user=True,
                        kind="api",
                    ) from e
                if self._client.is_api_error(e):
                    self._messages.pop()
                    raise _FatalLLM(
                        f"API error: {self._client.error_message(e)}",
                        pop_user=True,
                        kind="api",
                    ) from e
                if self._aborted:
                    raise AbortedError from e
                raise
        return final, tool_uses, sidecar

    def _sync_produce_llm_to_queue(
        self,
        loop: asyncio.AbstractEventLoop,
        q: asyncio.Queue[Any],
    ) -> None:
        """Run retrying LLM stream in a worker thread; forward chunks via ``call_soon_threadsafe``."""

        def _put(item: Any) -> None:
            try:
                loop.call_soon_threadsafe(q.put_nowait, item)
            except RuntimeError:
                pass

        for attempt in range(_MAX_RETRIES):
            tool_uses: list[Any] = []
            sidecar: list[str] = []
            try:
                self._messages = maybe_compact(self._messages)
                tool_schemas = [t.to_api_schema() for t in self._tools.values()]
                stream_obj = self._client.stream_messages(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=self._system_prompt,
                    tools=tool_schemas,
                    messages=self._messages,
                )
                with stream_obj as stream:
                    for text in stream.text_stream:
                        if self._aborted:
                            _put(("abort",))
                            return
                        _put(("chunk", text))
                    if self._aborted:
                        _put(("abort",))
                        return
                    final_msg = stream.get_final_message()
                    final = final_msg
                    if final.stop_reason == "max_tokens":
                        sidecar.append("Response truncated: max_tokens.")
                    for block in final.content:
                        if _block_type(block) == "tool_use":
                            tool_uses.append(block)
                    _put(("done", final, tool_uses, sidecar))
                return
            except AbortedError:
                _put(("abort",))
                return
            except Exception as e:
                if self._client.is_authentication_error(e):
                    _put(("fatal", _FatalLLM(str(e), pop_user=True, kind="auth")))
                    return
                if self._client.is_retryable_error(e):
                    if attempt < _MAX_RETRIES - 1:
                        wait = _compute_retry_delay(attempt)
                        sidecar.append(
                            f"API error, retrying in {wait:.1f}s… ({self._client.error_message(e)})",
                        )
                        for msg in sidecar:
                            _put(("sidecar", msg))
                        sidecar.clear()
                        time.sleep(wait)
                        continue
                    _put(
                        (
                            "fatal",
                            _FatalLLM(
                                f"API error after {_MAX_RETRIES} retries: {self._client.error_message(e)}",
                                pop_user=True,
                                kind="api",
                            ),
                        ),
                    )
                    return
                if self._client.is_api_error(e):
                    _put(
                        (
                            "fatal",
                            _FatalLLM(
                                f"API error: {self._client.error_message(e)}",
                                pop_user=True,
                                kind="api",
                            ),
                        ),
                    )
                    return
                if self._aborted:
                    _put(("abort",))
                    return
                _put(("fatal", e))
                return

    async def _consume_llm_stream(
        self,
        q: asyncio.Queue[Any],
        assistant_chunks: list[str],
        event_sink: StreamEventSink | None = None,
    ) -> tuple[Any | None, list[Any], list[str]]:
        sidecar: list[str] = []
        while True:
            item = await q.get()
            kind = item[0]
            if kind == "chunk":
                _, text = item
                assistant_chunks.append(text)
                if event_sink is not None:
                    try:
                        r = event_sink({"kind": "assistant_delta", "text": text})
                        if inspect.isawaitable(r):
                            await r
                    except Exception:
                        pass
                cb = self._text_stream_callback
                if cb is not None:
                    try:
                        r = cb(text)
                        if inspect.isawaitable(r):
                            await r
                    except Exception:
                        pass
            elif kind == "sidecar":
                sidecar.append(str(item[1]))
            elif kind == "done":
                _, final, tool_uses, extra = item
                sidecar.extend(list(extra))
                return final, tool_uses, sidecar
            elif kind == "abort":
                raise AbortedError
            elif kind == "fatal":
                raise item[1]
            else:
                raise RuntimeError(f"unknown stream item: {item!r}")

    async def _run_llm_in_thread(
        self,
        assistant_chunks: list[str],
        event_sink: StreamEventSink | None = None,
    ) -> tuple[Any | None, list[Any], list[str]]:
        if self._stream_assistant:
            q: asyncio.Queue[Any] = asyncio.Queue()
            loop = asyncio.get_running_loop()
            producer = asyncio.create_task(
                asyncio.to_thread(self._sync_produce_llm_to_queue, loop, q),
            )
            try:
                return await self._consume_llm_stream(
                    q, assistant_chunks, event_sink=event_sink
                )
            finally:
                await producer
        return await asyncio.to_thread(self._sync_llm_round)

    async def _append_stream_event(
        self,
        events: list[dict[str, Any]],
        ev: dict[str, Any],
        sink: StreamEventSink | None,
    ) -> None:
        events.append(ev)
        if sink is None:
            return
        try:
            r = sink(ev)
            if inspect.isawaitable(r):
                await r
        except Exception:
            pass

    async def _execute_one_tool(self, tu: Any, *, skip_permission: bool) -> ToolResult:
        tn = _block_name(tu)
        ti = _block_input(tu)
        tool = self._tools.get(tn)
        if tool is None:
            return ToolResult(content=f"Unknown tool: {tn}", is_error=True)
        if not skip_permission and self._permissions.check(tool, ti) == "deny":
            return ToolResult(content="Permission denied.", is_error=True)
        try:
            return await self._backend.call_tool(tn, ti)
        except Exception as e:
            return ToolResult(content=f"Tool execution error: {e}", is_error=True)

    async def respond(
        self,
        user_input: str,
        *,
        event_sink: StreamEventSink | None = None,
    ) -> dict[str, Any]:
        """Execute one user message round; returns JSON-serializable summary (events list, text).

        If ``event_sink`` is set, each event dict (and ``assistant_delta`` chunks when
        streaming) is passed to the sink as it is produced (for RPC streaming).
        """
        self._aborted = False
        self._user_msg_start = len(self._messages)
        events: list[dict[str, Any]] = []
        assistant_chunks: list[str] = []

        self._messages.append({"role": "user", "content": user_input})
        self._persist(self._messages[-1])

        try:
            while True:
                if self._aborted:
                    raise AbortedError

                try:
                    final, tool_uses, sidecar = await self._run_llm_in_thread(
                        assistant_chunks,
                        event_sink=event_sink,
                    )
                except _FatalLLM as fe:
                    if fe.pop_user:
                        self.rollback_pending()
                    await self._append_stream_event(
                        events,
                        {"kind": "error", "message": str(fe)},
                        event_sink,
                    )
                    return {
                        "events": events,
                        "assistant_text": "",
                        "ok": False,
                        "streamed_assistant": bool(
                            self._stream_assistant and self._text_stream_callback,
                        ),
                    }
                except AbortedError:
                    raise
                except Exception as e:
                    await self._append_stream_event(
                        events,
                        {"kind": "error", "message": str(e)},
                        event_sink,
                    )
                    return {
                        "events": events,
                        "assistant_text": "".join(assistant_chunks),
                        "ok": False,
                        "streamed_assistant": bool(
                            self._stream_assistant and self._text_stream_callback,
                        ),
                    }

                for msg in sidecar:
                    await self._append_stream_event(
                        events,
                        {"kind": "error", "message": msg},
                        event_sink,
                    )

                if final is None:
                    self._messages.pop()
                    await self._append_stream_event(
                        events,
                        {"kind": "error", "message": "empty model response"},
                        event_sink,
                    )
                    return {
                        "events": events,
                        "assistant_text": "",
                        "ok": False,
                        "streamed_assistant": bool(
                            self._stream_assistant and self._text_stream_callback,
                        ),
                    }

                self._messages.append({"role": "assistant", "content": final.content})
                self._persist(self._messages[-1])
                if not self._stream_assistant:
                    for block in final.content:
                        if _block_type(block) == "text":
                            if isinstance(block, dict):
                                assistant_chunks.append(str(block.get("text", "")))
                            else:
                                assistant_chunks.append(
                                    str(getattr(block, "text", "")),
                                )

                if not tool_uses:
                    break

                tool_results: list[dict[str, Any]] = []
                batches: list[tuple[bool, list[Any]]] = []
                for tu in tool_uses:
                    t = self._tools.get(_block_name(tu))
                    is_concurrent = t is not None and t.is_read_only()
                    if batches and batches[-1][0] == is_concurrent and is_concurrent:
                        batches[-1][1].append(tu)
                    else:
                        batches.append((is_concurrent, [tu]))

                for is_concurrent, batch in batches:
                    if self._aborted:
                        raise AbortedError

                    if is_concurrent and len(batch) > 1:
                        approved: list[Any] = []
                        denied: dict[str, ToolResult] = {}
                        for tu in batch:
                            tn = _block_name(tu)
                            ti = _block_input(tu)
                            tool = self._tools.get(tn)
                            act = tool.get_activity_description(**ti) if tool else None
                            await self._append_stream_event(
                                events,
                                {
                                    "kind": "tool_call",
                                    "name": tn,
                                    "input": ti,
                                    "activity": act,
                                },
                                event_sink,
                            )
                            if tool and self._permissions.check(tool, ti) == "deny":
                                denied[_block_id(tu)] = ToolResult(
                                    content="Permission denied.",
                                    is_error=True,
                                )
                            else:
                                approved.append(tu)

                        executed: dict[str, ToolResult] = {}
                        if approved:
                            for tu in approved:
                                tn = _block_name(tu)
                                ti = _block_input(tu)
                                tool = self._tools.get(tn)
                                act = (
                                    tool.get_activity_description(**ti)
                                    if tool
                                    else None
                                )
                                await self._append_stream_event(
                                    events,
                                    {
                                        "kind": "tool_executing",
                                        "name": tn,
                                        "input": ti,
                                        "activity": act,
                                    },
                                    event_sink,
                                )
                            async_results = await asyncio.gather(
                                *(
                                    self._execute_one_tool(tu, skip_permission=True)
                                    for tu in approved
                                ),
                                return_exceptions=True,
                            )
                            for tu, res in zip(approved, async_results, strict=True):
                                tid = _block_id(tu)
                                if isinstance(res, BaseException):
                                    executed[tid] = ToolResult(
                                        content=f"Tool execution error: {res}",
                                        is_error=True,
                                    )
                                else:
                                    executed[tid] = res

                        for tu in batch:
                            tid = _block_id(tu)
                            tn = _block_name(tu)
                            ti = _block_input(tu)
                            result = denied.get(tid) or executed.get(tid)
                            if result is None:
                                result = ToolResult(content="No result", is_error=True)
                            await self._append_stream_event(
                                events,
                                {
                                    "kind": "tool_result",
                                    "name": tn,
                                    "input": ti,
                                    "content": result.content,
                                    "is_error": result.is_error,
                                },
                                event_sink,
                            )
                            tool_results.append(_tool_result_dict(tid, result))
                    else:
                        for tu in batch:
                            if self._aborted:
                                raise AbortedError
                            tn = _block_name(tu)
                            ti = _block_input(tu)
                            tool = self._tools.get(tn)
                            act = tool.get_activity_description(**ti) if tool else None
                            await self._append_stream_event(
                                events,
                                {
                                    "kind": "tool_call",
                                    "name": tn,
                                    "input": ti,
                                    "activity": act,
                                },
                                event_sink,
                            )
                            if tool and self._permissions.check(tool, ti) == "deny":
                                result = ToolResult(
                                    content="Permission denied.",
                                    is_error=True,
                                )
                            else:
                                await self._append_stream_event(
                                    events,
                                    {
                                        "kind": "tool_executing",
                                        "name": tn,
                                        "input": ti,
                                        "activity": act,
                                    },
                                    event_sink,
                                )
                                result = await self._execute_one_tool(
                                    tu,
                                    skip_permission=True,
                                )
                            await self._append_stream_event(
                                events,
                                {
                                    "kind": "tool_result",
                                    "name": tn,
                                    "input": ti,
                                    "content": result.content,
                                    "is_error": result.is_error,
                                },
                                event_sink,
                            )
                            tool_results.append(
                                _tool_result_dict(_block_id(tu), result)
                            )

                self._messages.append({"role": "user", "content": tool_results})
                self._persist(self._messages[-1])

        except AbortedError:
            self.rollback_pending()
            await self._append_stream_event(
                events,
                {"kind": "aborted"},
                event_sink,
            )
            return {
                "events": events,
                "assistant_text": "".join(assistant_chunks),
                "ok": True,
                "streamed_assistant": bool(
                    self._stream_assistant and self._text_stream_callback,
                ),
            }

        return {
            "events": events,
            "assistant_text": "".join(assistant_chunks),
            "ok": True,
            "streamed_assistant": bool(
                self._stream_assistant and self._text_stream_callback,
            ),
        }


class _FatalLLM(Exception):
    def __init__(self, message: str, *, pop_user: bool, kind: str) -> None:
        super().__init__(message)
        self.pop_user = pop_user
        self.kind = kind
