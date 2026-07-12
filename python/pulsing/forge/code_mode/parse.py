# SPDX-License-Identifier: Apache-2.0
"""Parse exec cell source + optional ``# @exec:`` pragma (Codex-compatible shape)."""

from __future__ import annotations

import json
import re
from typing import Any

from pulsing.forge.code_mode.protocol import (
    DEFAULT_EXEC_YIELD_TIME_MS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    ParsedExecSource,
)

_CODE_MODE_PRAGMA_PREFIX = re.compile(
    r"^[ \t]*// @exec:\s*(?P<body>\{.*\})\s*(?:\r?\n)", re.MULTILINE
)
_PY_EXEC_PRAGMA_PREFIX = re.compile(
    r"^[ \t]*#\s*@exec:\s*(?P<body>\{.*\})\s*(?:\r?\n)", re.MULTILINE
)
_YIELD_CONTROL_LINE = re.compile(r"^\s*yield_control\s*\(\s*\)\s*(?:#.*)?$")


def split_yield_segments(source: str) -> list[str]:
    """Split top-level source at standalone ``yield_control()`` lines (L2 resume)."""
    lines = source.splitlines(keepends=True)
    segments: list[str] = []
    buf: list[str] = []
    for line in lines:
        buf.append(line)
        if _YIELD_CONTROL_LINE.match(line.rstrip("\r\n")):
            segments.append("".join(buf))
            buf = []
    if buf:
        segments.append("".join(buf))
    return segments or [source]


def parse_exec_source(raw: str) -> ParsedExecSource:
    text = raw if raw.endswith("\n") else raw + "\n"
    meta: dict[str, Any] = {}
    body = text

    for pattern in (_PY_EXEC_PRAGMA_PREFIX, _CODE_MODE_PRAGMA_PREFIX):
        match = pattern.search(text)
        if match:
            try:
                meta = json.loads(match.group("body"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid @exec pragma JSON: {exc}") from exc
            body = text[match.end() :]
            break

    yield_ms = int(meta.get("yield_time_ms") or DEFAULT_EXEC_YIELD_TIME_MS)
    max_tokens = int(meta.get("max_output_tokens") or DEFAULT_MAX_OUTPUT_TOKENS)
    if yield_ms < 0:
        raise ValueError("yield_time_ms must be non-negative")
    if max_tokens < 1:
        raise ValueError("max_output_tokens must be positive")

    stripped = body.strip("\n")
    if not stripped.strip():
        raise ValueError("exec source is empty")
    return ParsedExecSource(
        source=stripped,
        segments=split_yield_segments(stripped),
        yield_time_ms=yield_ms,
        max_output_tokens=max_tokens,
    )
