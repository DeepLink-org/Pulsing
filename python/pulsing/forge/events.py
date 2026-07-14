# SPDX-License-Identifier: Apache-2.0
"""Forge → Host point-to-point events (Codex event-bus equivalent)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ForgeEventKind(str, Enum):
    EXEC_OUTPUT_DELTA = "exec_output_delta"
    TOOL_BEGIN = "tool_begin"
    TOOL_END = "tool_end"
    PLAN_UPDATED = "plan_updated"
    NEW_CONTEXT = "new_context"
    USER_INPUT_REQUEST = "user_input_request"
    EXEC_APPROVAL_REQUEST = "exec_approval_request"
    REQUEST_PERMISSIONS = "request_permissions"


@dataclass
class ForgeEvent:
    """Pickle-friendly envelope for actor tell/ask transport."""

    kind: str
    payload: dict[str, Any]
    source: str | None = None
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ForgeEvent:
        return cls(
            kind=str(raw.get("kind", "")),
            payload=dict(raw.get("payload") or {}),
            source=raw.get("source"),
            ts=float(raw.get("ts") or time.time()),
        )

    @classmethod
    def exec_output_delta(
        cls, *, session_id: int, stream: str, chunk: str
    ) -> ForgeEvent:
        return cls(
            kind=ForgeEventKind.EXEC_OUTPUT_DELTA.value,
            payload={"session_id": session_id, "stream": stream, "chunk": chunk},
        )

    @classmethod
    def tool_begin(
        cls, tool: str, arguments: dict[str, Any] | None = None
    ) -> ForgeEvent:
        return cls(
            kind=ForgeEventKind.TOOL_BEGIN.value,
            source=tool,
            payload={"arguments": dict(arguments or {})},
        )

    @classmethod
    def tool_end(
        cls, tool: str, *, is_error: bool, content_preview: str = ""
    ) -> ForgeEvent:
        preview = content_preview[:500]
        return cls(
            kind=ForgeEventKind.TOOL_END.value,
            source=tool,
            payload={"is_error": is_error, "content_preview": preview},
        )

    @classmethod
    def exec_approval_request(cls, request: dict[str, Any]) -> ForgeEvent:
        return cls(
            kind=ForgeEventKind.EXEC_APPROVAL_REQUEST.value,
            payload=dict(request),
        )

    @classmethod
    def request_permissions(cls, args: dict[str, Any]) -> ForgeEvent:
        return cls(
            kind=ForgeEventKind.REQUEST_PERMISSIONS.value,
            payload=dict(args),
        )
