# SPDX-License-Identifier: Apache-2.0
"""Tool protocol: ``ToolResult`` and abstract ``Tool`` (standard tool API)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


def tool_result_from_worker_value(raw: Any) -> ToolResult:
    """Map isolated worker dict (``content`` / ``is_error``) to :class:`ToolResult`."""
    if isinstance(raw, dict) and "content" in raw:
        return ToolResult(
            content=str(raw["content"]),
            is_error=bool(raw.get("is_error", False)),
        )
    if hasattr(raw, "content") and hasattr(raw, "is_error"):
        return ToolResult(
            content=str(raw.content),
            is_error=bool(getattr(raw, "is_error", False)),
        )
    return ToolResult(content=str(raw), is_error=False)


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def input_schema(self) -> dict: ...

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult: ...

    def get_activity_description(self, **kwargs) -> str | None:
        return None

    def is_read_only(self) -> bool:
        return False

    def to_api_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
