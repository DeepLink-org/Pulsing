# SPDX-License-Identifier: Apache-2.0
"""Codex-aligned Code Mode wire types (Python cell, no V8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

PUBLIC_TOOL_NAME = "exec"
WAIT_TOOL_NAME = "wait"

DEFAULT_EXEC_YIELD_TIME_MS = 10_000
DEFAULT_WAIT_YIELD_TIME_MS = 10_000
DEFAULT_MAX_OUTPUT_TOKENS = 10_000


@dataclass(frozen=True)
class CellId:
    value: str

    def __str__(self) -> str:
        return self.value


class ContentItemType(str, Enum):
    TEXT = "text"


@dataclass
class ContentItem:
    type: Literal["text"] = "text"
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "text": self.text}


@dataclass
class ParsedExecSource:
    source: str
    segments: list[str] = field(default_factory=list)
    yield_time_ms: int = DEFAULT_EXEC_YIELD_TIME_MS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS


@dataclass
class RuntimeResponse:
    kind: Literal["yielded", "terminated", "result"]
    cell_id: CellId
    content_items: list[ContentItem] = field(default_factory=list)
    error_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "cell_id": str(self.cell_id),
            "content_items": [c.to_dict() for c in self.content_items],
        }
        if self.error_text is not None:
            out["error_text"] = self.error_text
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RuntimeResponse:
        items = [
            ContentItem(type="text", text=str(item.get("text") or ""))
            for item in (raw.get("content_items") or [])
            if isinstance(item, dict)
        ]
        return cls(
            kind=raw.get("kind") or "result",
            cell_id=CellId(str(raw.get("cell_id") or "unknown")),
            content_items=items,
            error_text=raw.get("error_text"),
        )

    def model_message(self) -> str:
        """Human-readable summary aligned with Codex exec/wait responses."""
        if self.kind == "yielded":
            prefix = f"Script running with cell ID {self.cell_id}"
        elif self.kind == "terminated":
            prefix = f"Script terminated (cell ID {self.cell_id})"
        else:
            prefix = f"Script completed (cell ID {self.cell_id})"
        body = "".join(c.text for c in self.content_items)
        if self.error_text:
            return f"{prefix}\nError: {self.error_text}\n{body}".strip()
        return f"{prefix}\n{body}".strip() if body else prefix


@dataclass
class WaitArgs:
    cell_id: str
    yield_time_ms: int = DEFAULT_WAIT_YIELD_TIME_MS
    max_tokens: int | None = None
    terminate: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WaitArgs:
        raw_yield_ms = raw.get("yield_time_ms")
        yield_time_ms = (
            DEFAULT_WAIT_YIELD_TIME_MS if raw_yield_ms is None else int(raw_yield_ms)
        )
        if yield_time_ms < 0:
            raise ValueError("yield_time_ms must be non-negative")

        raw_max_tokens = raw.get("max_tokens")
        max_tokens = None if raw_max_tokens is None else int(raw_max_tokens)
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be positive")

        return cls(
            cell_id=str(raw.get("cell_id") or ""),
            yield_time_ms=yield_time_ms,
            max_tokens=max_tokens,
            terminate=bool(raw.get("terminate")),
        )
