# SPDX-License-Identifier: Apache-2.0
"""Nested Forge tool calls from a Python code cell."""

from __future__ import annotations

import json
from typing import Any, Callable

from pulsing.forge.code_mode.protocol import PUBLIC_TOOL_NAME, WAIT_TOOL_NAME
from pulsing.forge.result import ToolResult

# Tools a cell may not invoke (avoid recursion / host-only orchestration).
_CODE_MODE_BLOCKED = frozenset({PUBLIC_TOOL_NAME, WAIT_TOOL_NAME})


def default_nested_tool_names() -> frozenset[str]:
    from pulsing.forge.integrated import (
        FORGE_ISOLATED_TOOL_NAMES,
        FORGE_HOST_TOOL_NAMES,
    )

    allowed = (FORGE_ISOLATED_TOOL_NAMES | FORGE_HOST_TOOL_NAMES) - _CODE_MODE_BLOCKED
    return frozenset(allowed)


class ToolsBridge:
    """``tools.call(name, args)`` surface exposed inside exec cells."""

    def __init__(
        self,
        call_tool: Callable[[str, dict[str, Any]], ToolResult],
        *,
        allowed: frozenset[str] | None = None,
    ) -> None:
        self._call_tool = call_tool
        self._allowed = allowed or default_nested_tool_names()

    def call(self, name: str, args: dict[str, Any] | str | None = None) -> Any:
        tool = str(name).strip()
        if tool not in self._allowed:
            raise PermissionError(f"tool not available in code mode: {tool}")
        payload: dict[str, Any]
        if args is None:
            payload = {}
        elif isinstance(args, str):
            payload = {"input": args}
        elif isinstance(args, dict):
            payload = dict(args)
        else:
            raise TypeError("tool args must be a dict, string, or omitted")
        result = self._call_tool(tool, payload)
        if result.is_error:
            raise RuntimeError(result.content or f"{tool} failed")
        if result.structured is not None:
            return result.structured
        text = result.content or ""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def _invoke(args: dict[str, Any] | str | None = None) -> Any:
            return self.call(name, args)

        return _invoke
