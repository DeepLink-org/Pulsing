# SPDX-License-Identifier: Apache-2.0
"""Picklable tool call result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    structured: dict[str, Any] | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"content": self.content, "is_error": self.is_error}
        if self.structured is not None:
            out["structured"] = self.structured
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolResult:
        return cls(
            content=str(data.get("content") or ""),
            is_error=bool(data.get("is_error", False)),
            structured=data.get("structured"),
        )
