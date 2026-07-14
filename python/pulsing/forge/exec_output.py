# SPDX-License-Identifier: Apache-2.0
"""Codex-compatible exec output helpers."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

DEFAULT_SHELL_TIMEOUT_MS = 10_000
DEFAULT_YIELD_TIME_MS = 250
MIN_YIELD_TIME_MS = 250
MAX_YIELD_TIME_MS = 30_000
DEFAULT_MAX_OUTPUT_TOKENS = 10_000
SHELL_MAX_BYTES = 256 * 1024
MAX_STDIN_BYTES = 1024 * 1024
RUNNING_EXIT_SENTINEL = -(2**31)


class ExecStream(str, Enum):
    STDOUT = "stdout"
    STDERR = "stderr"
    PTY = "pty"


@dataclass
class ExecOutputDelta:
    session_id: int
    stream: ExecStream
    chunk: str


@dataclass
class ExecCommandOutput:
    chunk_id: str
    wall_time_seconds: float
    output: str
    exit_code: int | None = None
    session_id: int | None = None
    original_token_count: int | None = None

    @classmethod
    def build(
        cls,
        output: str,
        wall_time_seconds: float,
        exit_code: int | None,
        session_id: int | None,
    ) -> ExecCommandOutput:
        return cls(
            chunk_id=str(uuid.uuid4()),
            wall_time_seconds=wall_time_seconds,
            output=output,
            exit_code=exit_code,
            session_id=session_id,
            original_token_count=max(1, len(output) // 4),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


def _tail_at(s: str, keep: int) -> str:
    # Python str slicing is codepoint-safe; keep parity with Rust helper naming.
    return s[-keep:] if keep > 0 else ""


class Utf8ChunkDecoder:
    """Incrementally decode PTY/pipe byte chunks without splitting codepoints."""

    def __init__(self) -> None:
        self._pending = bytearray()

    def decode(self, data: bytes) -> str:
        if data:
            self._pending.extend(data)
        try:
            out = self._pending.decode("utf-8")
            self._pending.clear()
            return out
        except UnicodeDecodeError as e:
            valid_len = e.start
            out = self._pending[:valid_len].decode("utf-8", errors="replace")
            leftover = len(self._pending) - valid_len
            if leftover > 4:
                rest = self._pending[valid_len:].decode("utf-8", errors="replace")
                self._pending.clear()
                return out + rest
            del self._pending[:valid_len]
            return out

    def finish(self) -> str:
        if not self._pending:
            return ""
        out = self._pending.decode("utf-8", errors="replace")
        self._pending.clear()
        return out


class OutputBuffer:
    def __init__(self, max_bytes: int = SHELL_MAX_BYTES) -> None:
        self._max_bytes = max_bytes
        self._data = ""

    def push(self, chunk: str) -> None:
        if not chunk:
            return
        self._data += chunk
        if len(self._data) > self._max_bytes:
            keep = self._max_bytes // 2
            self._data = f"...[output truncated]...\n{_tail_at(self._data, keep)}"

    def snapshot(self) -> str:
        return self._data

    def truncate_to_tokens(self, max_tokens: int) -> None:
        est = max(1, len(self._data) // 4)
        if est <= max_tokens:
            return
        ratio = max_tokens / est
        keep = int(len(self._data) * ratio)
        self._data = f"...[token limit]...\n{_tail_at(self._data, keep)}"


def clamp_yield_ms(raw: int | None) -> int:
    return max(MIN_YIELD_TIME_MS, min(raw or DEFAULT_YIELD_TIME_MS, MAX_YIELD_TIME_MS))


def shell_timeout_ms(args: dict[str, Any]) -> int:
    raw_ms = args.get("timeout_ms")
    if raw_ms is not None:
        return max(1, int(raw_ms))
    raw_sec = args.get("timeout_sec")
    if raw_sec is not None:
        return max(1, int(raw_sec) * 1000)
    return DEFAULT_SHELL_TIMEOUT_MS
