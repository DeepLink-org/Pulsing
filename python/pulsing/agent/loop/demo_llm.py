# SPDX-License-Identifier: Apache-2.0
"""Offline demo LLM — scripted replies and tool calls for ``pulsing agent demo``."""

from __future__ import annotations

import re
import uuid
from typing import Any, Iterator

from pulsing.agent.loop.llm_client import LLMMessage, LLMUsage

_DEMO_PEERS = ("bard", "smith", "sage", "guide")


def _is_tool_result_user_message(msg: dict[str, Any]) -> bool:
    content = msg.get("content")
    if not isinstance(content, list) or not content:
        return False
    return all(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            if _is_tool_result_user_message(msg):
                continue
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
            return " ".join(parts).strip()
    return ""


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for t in tools or []:
        name = t.get("name")
        if name:
            out.add(str(name))
    return out


def plan_demo_turn(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    # After tool results, finish with a short text reply (avoid infinite tool loops).
    if messages and _is_tool_result_user_message(messages[-1]):
        original = _last_user_text(messages)
        snippet = re.sub(r"\s+", " ", original)[:100]
        return {
            "kind": "text",
            "text": f"(demo) Finished tool run for: {snippet or 'your request'}.",
        }

    text = _last_user_text(messages)
    lower = text.lower()
    allowed = _tool_names(tools)

    if any(k in lower for k in ("glob", "files", "list project", "directory")):
        if "Glob" in allowed:
            return {
                "kind": "tool",
                "name": "Glob",
                "input": {"pattern": "*", "path": "."},
            }

    if any(k in lower for k in ("read", "readme", "summary")):
        if "Read" in allowed:
            return {
                "kind": "tool",
                "name": "Read",
                "input": {"file_path": "README.md"},
            }

    if any(k in lower for k in ("quest", "puzzle", "unit-test", "questreport")):
        if "QuestReport" in allowed:
            return {
                "kind": "tool",
                "name": "QuestReport",
                "input": {
                    "quest_id": "unit-tests",
                    "status": "in_progress",
                    "note": "demo chatter",
                },
            }

    if "messageclusteragent" in lower or "coordinate" in lower or "peer" in lower:
        if "MessageClusterAgent" in allowed:
            target = next((p for p in _DEMO_PEERS if p in lower), "smith")
            return {
                "kind": "tool",
                "name": "MessageClusterAgent",
                "input": {
                    "agent": target,
                    "message": "Demo ping — reply in one short sentence.",
                    "wait": False,
                },
            }

    snippet = re.sub(r"\s+", " ", text)[:100]
    return {"kind": "text", "text": f"(demo) Noted: {snippet or '(empty)'}"}


class _DemoStream:
    """Anthropic-shaped stream context for :class:`LLMClient`."""

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
    ) -> None:
        _ = model, max_tokens, system
        self._plan = plan_demo_turn(messages, tools)
        self._text = (
            self._plan.get("text", "") if self._plan.get("kind") == "text" else ""
        )
        self.text_stream: Iterator[str] = iter(())

    def __enter__(self) -> _DemoStream:
        if self._text:
            self.text_stream = iter([self._text])
        else:
            self.text_stream = iter(())
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return False

    def close(self) -> None:
        pass

    def get_final_message(self) -> LLMMessage:
        if self._plan.get("kind") == "tool":
            content = [
                {
                    "type": "tool_use",
                    "id": f"demo-{uuid.uuid4().hex[:12]}",
                    "name": self._plan["name"],
                    "input": dict(self._plan.get("input") or {}),
                },
            ]
            stop = "tool_use"
        else:
            content = [{"type": "text", "text": self._text}]
            stop = "end_turn"
        return LLMMessage(
            content=content,
            usage=LLMUsage(input_tokens=1, output_tokens=max(1, len(self._text) // 4)),
            stop_reason=stop,
        )
