# SPDX-License-Identifier: Apache-2.0
"""LLM streaming for ``pulsing.agent.loop``: Anthropic ``messages.stream`` + OpenAI chat completions stream.

OpenAI message/tool conversion follows the same patterns from common OpenAI chat patterns
(adapted here under Apache-2.0).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator

import anthropic
import httpx

_OPENAI_IMPORT_ERROR: Exception | None = None
try:
    import openai
    from openai import OpenAI
except Exception as exc:  # pragma: no cover
    OpenAI = None  # type: ignore[misc, assignment]
    openai = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = exc

_VALID_PROVIDERS = frozenset({"anthropic", "openai", "demo"})


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMMessage:
    content: list[dict[str, Any]]
    usage: LLMUsage | None = None
    stop_reason: str | None = None


def _usage_from_anthropic(raw: Any) -> LLMUsage | None:
    if raw is None:
        return None
    return LLMUsage(
        input_tokens=int(getattr(raw, "input_tokens", 0) or 0),
        output_tokens=int(getattr(raw, "output_tokens", 0) or 0),
    )


def _normalize_anthropic_content(blocks: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(blocks, list):
        return out
    for block in blocks:
        if hasattr(block, "model_dump"):
            out.append(block.model_dump())
        elif isinstance(block, dict):
            out.append(block)
    return out


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _usage_from_openai(usage: Any) -> LLMUsage | None:
    if usage is None:
        return None
    return LLMUsage(
        input_tokens=int(_value(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(_value(usage, "completion_tokens", 0) or 0),
    )


def _normalize_openai_stop_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    mapping = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
    return mapping.get(reason, reason)


def _tool_result_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _user_content_blocks_to_openai(content: list[Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "image":
            source = block.get("source", {})
            media_type = source.get("media_type", "image/png")
            data = source.get("data", "")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{data}"},
                },
            )
    if not parts:
        return [{"type": "text", "text": ""}]
    return parts


def _tool_schema_to_openai(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        },
    }


def _to_openai_messages(
    system: str | None,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")

        if role == "user" and isinstance(content, list):
            tool_results = [
                b
                for b in content
                if isinstance(b, dict) and b.get("type") == "tool_result"
            ]
            if tool_results and len(tool_results) == len(content):
                for block in tool_results:
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": _tool_result_to_text(block.get("content", "")),
                        },
                    )
                continue

            out.append(
                {
                    "role": "user",
                    "content": _user_content_blocks_to_openai(content),
                },
            )
            continue

        if role == "assistant" and isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(
                                    block.get("input", {}),
                                    ensure_ascii=False,
                                ),
                            },
                        },
                    )
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(text_parts) or None,
            }
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            out.append(assistant_message)
            continue

        out.append({"role": role, "content": content})

    return out


def _build_openai_request(
    *,
    model: str,
    max_tokens: int,
    system: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    stream: bool,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": model,
        "messages": _to_openai_messages(system, messages),
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if tools:
        params["tools"] = [_tool_schema_to_openai(t) for t in tools]
    return params


class _AnthropicStream:
    def __init__(
        self,
        *,
        client: anthropic.Anthropic,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
    ) -> None:
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        self._raw = client.messages.stream(**kwargs)
        self._ctx: Any = None
        self.text_stream: Iterator[str] = iter(())

    def __enter__(self) -> _AnthropicStream:
        self._ctx = self._raw.__enter__()
        self.text_stream = self._ctx.text_stream
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return self._raw.__exit__(exc_type, exc, tb)

    def close(self) -> None:
        try:
            self._raw.close()
        except Exception:
            pass

    def get_final_message(self) -> LLMMessage:
        final = self._ctx.get_final_message()
        return LLMMessage(
            content=_normalize_anthropic_content(getattr(final, "content", [])),
            usage=_usage_from_anthropic(getattr(final, "usage", None)),
            stop_reason=getattr(final, "stop_reason", None),
        )


class _OpenAIStream:
    """OpenAI streaming completion; ``text_stream`` yields text deltas; final content uses Anthropic-shaped blocks."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
    ) -> None:
        self._client = client
        self._params = _build_openai_request(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            stream=True,
        )
        self._stream: Any = None
        self._text_parts: list[str] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._usage: LLMUsage | None = None
        self._finish_reason: str | None = None
        self.text_stream: Iterator[str] = iter(())

    def __enter__(self) -> _OpenAIStream:
        self._stream = self._client.chat.completions.create(**self._params)
        self.text_stream = self._iter_text()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
        return False

    def close(self) -> None:
        if self._stream is not None and hasattr(self._stream, "close"):
            try:
                self._stream.close()
            except Exception:
                pass

    def _iter_text(self) -> Iterator[str]:
        assert self._stream is not None
        for chunk in self._stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self._usage = _usage_from_openai(usage)
            for choice in _value(chunk, "choices", []) or []:
                finish_reason = _value(choice, "finish_reason")
                if finish_reason:
                    self._finish_reason = finish_reason
                delta = _value(choice, "delta", {}) or {}
                content = _value(delta, "content")
                if content:
                    self._text_parts.append(content)
                    yield content
                for tool_call in _value(delta, "tool_calls", []) or []:
                    index = int(_value(tool_call, "index", 0) or 0)
                    entry = self._tool_calls.setdefault(
                        index,
                        {"id": "", "name": "", "arguments": ""},
                    )
                    tool_id = _value(tool_call, "id")
                    if tool_id:
                        entry["id"] = tool_id
                    function = _value(tool_call, "function", {}) or {}
                    name = _value(function, "name")
                    if name:
                        entry["name"] = name
                    arguments = _value(function, "arguments")
                    if arguments:
                        entry["arguments"] += arguments

    def get_final_message(self) -> LLMMessage:
        content: list[dict[str, Any]] = []
        text = "".join(self._text_parts)
        if text:
            content.append({"type": "text", "text": text})
        for index in sorted(self._tool_calls):
            tool_call = self._tool_calls[index]
            raw_args = tool_call.get("arguments", "").strip()
            parsed_args: Any = {}
            if raw_args:
                try:
                    parsed_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    parsed_args = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id", ""),
                    "name": tool_call.get("name", ""),
                    "input": parsed_args if isinstance(parsed_args, dict) else {},
                },
            )
        return LLMMessage(
            content=content,
            usage=self._usage,
            stop_reason=_normalize_openai_stop_reason(self._finish_reason),
        )


class LLMClient:
    """Anthropic or OpenAI client; ``stream_messages`` returns a stream context manager."""

    def __init__(
        self,
        *,
        provider: str = "anthropic",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        p = (provider or "anthropic").strip().lower()
        if p not in _VALID_PROVIDERS:
            raise ValueError(
                f"Unsupported provider {provider!r}; use anthropic, openai, or demo"
            )
        self.provider = p
        if self.provider == "demo":
            self._client = None
        elif self.provider == "openai":
            if OpenAI is None:
                msg = 'OpenAI provider requires dependency: pip install "pulsing[agent]" (includes openai).'
                if _OPENAI_IMPORT_ERROR is not None:
                    msg += f" Import error: {_OPENAI_IMPORT_ERROR}"
                raise ValueError(msg)
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self._client = anthropic.Anthropic(
                api_key=api_key,
                base_url=base_url,
                timeout=httpx.Timeout(600.0, connect=30.0),
            )

    def stream_messages(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> _AnthropicStream | _OpenAIStream:
        if self.provider == "demo":
            from pulsing.agent.loop.demo_llm import _DemoStream

            return _DemoStream(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                system=system,
                tools=tools or [],
            )
        if self.provider == "openai":
            return _OpenAIStream(
                client=self._client,
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                system=system,
                tools=tools or [],
            )
        return _AnthropicStream(
            client=self._client,
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            system=system,
            tools=tools or [],
        )

    @staticmethod
    def error_message(exc: Exception) -> str:
        return str(getattr(exc, "message", None) or exc)

    def is_authentication_error(self, exc: Exception) -> bool:
        if self.provider == "demo":
            return False
        if self.provider == "openai" and openai is not None:
            return isinstance(exc, openai.AuthenticationError)
        return isinstance(exc, anthropic.AuthenticationError)

    def is_retryable_error(self, exc: Exception) -> bool:
        if self.provider == "demo":
            return False
        if isinstance(
            exc,
            (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError),
        ):
            return True
        if self.provider == "openai" and openai is not None:
            return isinstance(
                exc,
                (
                    openai.RateLimitError,
                    openai.APIConnectionError,
                    openai.InternalServerError,
                ),
            )
        return isinstance(
            exc,
            (
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            ),
        )

    def is_api_error(self, exc: Exception) -> bool:
        if self.provider == "demo":
            return False
        if self.provider == "openai" and openai is not None:
            return isinstance(exc, openai.APIError)
        return isinstance(exc, anthropic.APIError)
