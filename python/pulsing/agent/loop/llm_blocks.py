# SPDX-License-Identifier: Apache-2.0
"""Shared LLM tool-use block helpers."""

from __future__ import annotations

from typing import Any

from pulsing.agent.loop.tool_base import ToolResult


class AbortedError(Exception):
    """Turn aborted via ``abort()`` (standard abort semantics)."""


def tool_result_dict(tid: str, result: ToolResult) -> dict[str, Any]:
    d: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tid,
        "content": result.content,
    }
    if result.is_error:
        d["is_error"] = True
    return d


def block_type(block: Any) -> str | None:
    if isinstance(block, dict):
        return str(block.get("type")) if block.get("type") is not None else None
    return getattr(block, "type", None)


def block_name(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("name", ""))
    return str(getattr(block, "name", ""))


def block_id(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("id", ""))
    return str(getattr(block, "id", ""))


def block_input(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        value = block.get("input", {})
    else:
        value = getattr(block, "input", {})
    return dict(value) if isinstance(value, dict) else {}
