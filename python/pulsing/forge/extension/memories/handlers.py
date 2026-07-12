# SPDX-License-Identifier: Apache-2.0
"""Handlers for ``memories.*`` namespace tools (Codex wire shapes)."""

from __future__ import annotations

import json
from typing import Any

from pulsing.forge.context import ToolCallContext
from pulsing.forge.extension.memories.backend import (
    MemoriesBackendError,
    SearchMatchMode,
)
from pulsing.forge.extension.memories.local_backend import (
    LocalMemoriesStore,
    default_ad_hoc_filename,
)
from pulsing.forge.extension.memories.path_utils import (
    validate_optional_scoped_path,
    validate_read_path,
)
from pulsing.forge.result import ToolResult


def _store(ctx: ToolCallContext) -> LocalMemoriesStore:
    return ctx.memories


def _ok_response(payload: dict[str, Any]) -> ToolResult:
    return ToolResult(
        content=json.dumps(payload, indent=2, ensure_ascii=False),
        structured=payload,
    )


def _err(exc: Exception) -> ToolResult:
    return ToolResult(content=str(exc), is_error=True)


def handle_memories_list(
    *,
    ctx: ToolCallContext,
    path: str | None = None,
    cursor: str | None = None,
    max_results: int | None = None,
    **_: Any,
) -> ToolResult:
    try:
        rel_path = validate_optional_scoped_path(path)
    except MemoriesBackendError as exc:
        return _err(exc)
    try:
        response = _store(ctx).list_memories(
            path=rel_path, cursor=cursor, max_results=max_results
        )
    except MemoriesBackendError as exc:
        return _err(exc)
    except OSError as exc:
        return _err(exc)
    return _ok_response(response.to_dict())


def handle_memories_read(
    *,
    ctx: ToolCallContext,
    path: str = "",
    line_offset: int | None = None,
    max_lines: int | None = None,
    max_tokens: int | None = None,
    **_: Any,
) -> ToolResult:
    try:
        rel_path = validate_read_path(path)
    except MemoriesBackendError as exc:
        return _err(exc)
    try:
        response = _store(ctx).read_memory(
            path=rel_path,
            line_offset=int(line_offset or 1),
            max_lines=max_lines,
            max_tokens=int(max_tokens or 0),
        )
    except MemoriesBackendError as exc:
        return _err(exc)
    except OSError as exc:
        return _err(exc)
    return _ok_response(response.to_dict())


def handle_memories_search(
    *,
    ctx: ToolCallContext,
    queries: list[str] | None = None,
    query: str = "",
    match_mode: Any = None,
    path: str | None = None,
    cursor: str | None = None,
    context_lines: int | None = None,
    case_sensitive: bool | None = None,
    normalized: bool | None = None,
    max_results: int | None = None,
    **_: Any,
) -> ToolResult:
    qlist = list(queries) if queries else ([query] if query else [])
    rel_path = str(path).strip() if path else None
    if rel_path == "":
        rel_path = None
    try:
        response = _store(ctx).search_memories(
            queries=qlist,
            match_mode=SearchMatchMode.from_wire(match_mode),
            path=rel_path,
            cursor=cursor,
            context_lines=int(context_lines or 0),
            case_sensitive=bool(case_sensitive if case_sensitive is not None else True),
            normalized=bool(normalized or False),
            max_results=max_results,
        )
    except MemoriesBackendError as exc:
        return _err(exc)
    return _ok_response(response.to_dict())


def handle_memories_add_ad_hoc_note(
    *,
    ctx: ToolCallContext,
    filename: str = "",
    path: str = "",
    note: str = "",
    text: str = "",
    content: str = "",
    **_: Any,
) -> ToolResult:
    body = note or text or content
    name = (filename or path).strip() or default_ad_hoc_filename(body[:40])
    try:
        out = _store(ctx).add_ad_hoc_note(filename=name, note=body)
    except MemoriesBackendError as exc:
        return _err(exc)
    except OSError as exc:
        return _err(exc)
    return _ok_response(out)
