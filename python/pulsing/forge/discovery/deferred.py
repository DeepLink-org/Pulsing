# SPDX-License-Identifier: Apache-2.0
"""Tools registered at runtime after ``tool_search``."""

from __future__ import annotations

import json
from typing import Any, Protocol

from pulsing.forge.tool_schema import json_schema_object


class ForgeToolLike(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def input_schema(self) -> dict[str, Any]: ...

    def is_read_only(self) -> bool: ...

    def execute(self, **kwargs: Any): ...


class DeferredForgeTool:
    """Schema-only tool activated after ``tool_search`` (execution via Forge host/worker)."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        *,
        read_only: bool = False,
    ) -> None:
        self._name = name
        self._description = description
        self._parameters = parameters
        self._read_only = read_only

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def input_schema(self) -> dict:
        props = dict(self._parameters.get("properties") or {})
        required = list(self._parameters.get("required") or [])
        return json_schema_object(props, required or None)

    def is_read_only(self) -> bool:
        return self._read_only

    def execute(self, **kwargs: Any):
        raise RuntimeError(f"{self._name} runs in Forge worker/host")


def parse_tool_search_result(content: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return []
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return []
    return [t for t in tools if isinstance(t, dict) and t.get("name")]


def activate_discovered_tools(
    tools_by_name: dict[str, Any],
    content: str,
    *,
    register: Any | None = None,
) -> list[str]:
    """Register ``tool_search`` hits. ``register(tool)`` optional (e.g. LlmChat.register_tool)."""
    specs = parse_tool_search_result(content)
    if not specs:
        return []

    activated: list[str] = []
    for spec in specs:
        name = str(spec.get("name", "")).strip()
        if not name or name in tools_by_name:
            continue
        tool = DeferredForgeTool(
            name=name,
            description=str(spec.get("description", "")),
            parameters=dict(
                spec.get("parameters") or {"type": "object", "properties": {}}
            ),
        )
        tools_by_name[name] = tool
        if register is not None:
            register(tool)
        activated.append(name)
    return activated
