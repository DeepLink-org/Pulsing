# SPDX-License-Identifier: Apache-2.0
"""Deferred tool entries for tool_search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TOOL_SEARCH_DEFAULT_LIMIT = 8
TOOL_SEARCH_MAX_LIMIT = 100
TOOL_SEARCH_MAX_QUERY_CHARS = 2000


def normalize_tool_search_limit(limit: Any) -> int:
    """Positive limits only; non-positive or unparsable values use the default."""
    if limit is None:
        return TOOL_SEARCH_DEFAULT_LIMIT
    try:
        n = int(limit)
    except (TypeError, ValueError):
        return TOOL_SEARCH_DEFAULT_LIMIT
    if n <= 0:
        return TOOL_SEARCH_DEFAULT_LIMIT
    return min(n, TOOL_SEARCH_MAX_LIMIT)


@dataclass
class DeferredToolEntry:
    name: str
    description: str
    parameters: dict[str, Any]
    search_text: str
    defer_loading: bool = True
    namespace: str | None = None
    plugin_id: str | None = None
    source: str | None = None

    @classmethod
    def from_function(
        cls,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DeferredToolEntry:
        params = parameters or {"type": "object", "properties": {}}
        search_text = f"{name} {name.replace('_', ' ')} {description}"
        return cls(
            name=name,
            description=description,
            parameters=params,
            search_text=search_text,
            **kwargs,
        )

    def to_loadable_json(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "defer_loading": self.defer_loading,
            "namespace": self.namespace,
            "plugin_id": self.plugin_id,
            "source": self.source,
        }
