# SPDX-License-Identifier: Apache-2.0
"""Parse LLM tool calls and build tool-result messages for custom agent frameworks.

Forge executes tools; this module bridges **provider wire formats** (OpenAI,
Anthropic) and Forge's ``ToolResult`` so Host code stays small.

Typical loop::

    calls = extract_tool_calls(assistant_message, provider="openai")
    for call in calls:
        result = await runtime.call_tool(call.name, call.arguments)
        messages.append(openai_tool_message(call.id, result))
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pulsing.forge.result import ToolResult

Provider = Literal["auto", "openai", "anthropic"]


@dataclass(frozen=True)
class ParsedToolCall:
    """Normalized tool invocation from an LLM response."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str | None = None


def parse_tool_arguments(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    """Parse tool arguments from JSON string or dict; never raises."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    text = str(raw).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def extract_tool_calls_openai(message: dict[str, Any]) -> list[ParsedToolCall]:
    """Extract from an OpenAI-style assistant message (``tool_calls`` field)."""
    out: list[ParsedToolCall] = []
    for entry in message.get("tool_calls") or []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") or {}
        if not isinstance(fn, dict):
            fn = {}
        raw_args = fn.get("arguments", "")
        out.append(
            ParsedToolCall(
                id=str(entry.get("id", "")),
                name=str(fn.get("name", "")),
                arguments=parse_tool_arguments(raw_args),
                raw_arguments=str(raw_args) if raw_args else None,
            )
        )
    return [c for c in out if c.name]


def extract_tool_calls_anthropic(content: list[Any]) -> list[ParsedToolCall]:
    """Extract from Anthropic ``content`` blocks (``type: tool_use``)."""
    out: list[ParsedToolCall] = []
    for block in content or []:
        if isinstance(block, dict):
            if block.get("type") != "tool_use":
                continue
            name = str(block.get("name", ""))
            args = block.get("input", {})
            call_id = str(block.get("id", ""))
        else:
            if getattr(block, "type", None) != "tool_use":
                continue
            name = str(getattr(block, "name", ""))
            args = getattr(block, "input", {})
            call_id = str(getattr(block, "id", ""))
        out.append(
            ParsedToolCall(
                id=call_id,
                name=name,
                arguments=dict(args) if isinstance(args, dict) else {},
            )
        )
    return [c for c in out if c.name]


def extract_tool_calls(
    message: dict[str, Any],
    *,
    provider: Provider = "auto",
) -> list[ParsedToolCall]:
    """Unified extractor for assistant messages or Anthropic-shaped dicts.

    * ``provider="openai"`` — expects ``message["tool_calls"]``.
    * ``provider="anthropic"`` — expects ``message["content"]`` as block list.
    * ``provider="auto"`` — OpenAI if ``tool_calls`` present, else Anthropic blocks.
    """
    if provider == "openai":
        return extract_tool_calls_openai(message)
    if provider == "anthropic":
        content = message.get("content", [])
        return extract_tool_calls_anthropic(
            content if isinstance(content, list) else []
        )

    if message.get("tool_calls"):
        return extract_tool_calls_openai(message)
    content = message.get("content")
    if isinstance(content, list):
        calls = extract_tool_calls_anthropic(content)
        if calls:
            return calls
    return []


def openai_tool_message(call_id: str, result: ToolResult) -> dict[str, Any]:
    """Build an OpenAI ``role: tool`` message from a Forge result."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": result.content,
    }


def anthropic_tool_result_block(call_id: str, result: ToolResult) -> dict[str, Any]:
    """Build an Anthropic ``tool_result`` content block."""
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": result.content,
    }
    if result.is_error:
        block["is_error"] = True
    return block


def anthropic_tool_results_message(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap tool-result blocks in a user message (Anthropic API shape)."""
    return {"role": "user", "content": blocks}


def to_openai_tools(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Anthropic-style ``{name, description, input_schema}`` to OpenAI tools."""
    out: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            out.append(tool)
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema")
                    or tool.get("parameters")
                    or {},
                },
            }
        )
    return out


def to_anthropic_tools(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OpenAI ``tools`` to Anthropic ``tools`` list."""
    out: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("name") and tool.get("input_schema") is not None:
            out.append(
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool["input_schema"],
                }
            )
            continue
        fn = tool.get("function") or {}
        if not isinstance(fn, dict):
            continue
        out.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {},
            }
        )
    return [t for t in out if t.get("name")]


class OpenAIToolCallAccumulator:
    """Accumulate streaming OpenAI ``delta.tool_calls`` chunks into parsed calls."""

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, str]] = {}

    def feed_chunk(self, chunk: dict[str, Any]) -> None:
        """Feed one streaming chunk (``choices[0].delta`` shape or full chunk)."""
        choices = chunk.get("choices") or []
        if not choices:
            return
        delta = choices[0].get("delta") or {}
        for tool_call in delta.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            index = int(tool_call.get("index", 0) or 0)
            entry = self._calls.setdefault(
                index, {"id": "", "name": "", "arguments": ""}
            )
            if tool_call.get("id"):
                entry["id"] = str(tool_call["id"])
            fn = tool_call.get("function") or {}
            if isinstance(fn, dict):
                if fn.get("name"):
                    entry["name"] = str(fn["name"])
                if fn.get("arguments"):
                    entry["arguments"] += str(fn["arguments"])

    def finish(self) -> list[ParsedToolCall]:
        """Return accumulated calls after the stream ends."""
        out: list[ParsedToolCall] = []
        for index in sorted(self._calls):
            entry = self._calls[index]
            raw_args = entry.get("arguments", "")
            out.append(
                ParsedToolCall(
                    id=entry.get("id", ""),
                    name=entry.get("name", ""),
                    arguments=parse_tool_arguments(raw_args),
                    raw_arguments=raw_args or None,
                )
            )
        return [c for c in out if c.name]

    def reset(self) -> None:
        self._calls.clear()


def forge_tool_definitions(
    names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Anthropic-shaped tool defs for Forge tools (``name`` / ``description`` / ``input_schema``).

    Full set (32 tools) requires ``pulsing.agent`` templates; falls back to a
    minimal filesystem + shell subset when unavailable.
    """
    try:
        from pulsing.agent.loop.forge_tools import all_forge_tool_templates

        templates = all_forge_tool_templates()
        defs = [t.to_api_schema() for t in templates]
    except ImportError:
        from pulsing.forge.tool_definitions import minimal_forge_tool_definitions

        defs = minimal_forge_tool_definitions()

    if names is None:
        return defs
    wanted = set(names)
    return [d for d in defs if d.get("name") in wanted]
