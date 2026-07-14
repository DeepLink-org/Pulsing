# SPDX-License-Identifier: Apache-2.0
"""Dispatch Codex Extension namespace tools."""

from __future__ import annotations

from typing import Any

from pulsing.forge.context import ToolCallContext
from pulsing.forge.extension.protocol import (
    EXTENSION_TOOL_NAMES,
    MEMORIES_ADD_AD_HOC_NOTE_TOOL,
    MEMORIES_LIST_TOOL,
    MEMORIES_READ_TOOL,
    MEMORIES_SEARCH_TOOL,
    SKILLS_LIST_TOOL,
    SKILLS_READ_TOOL,
    WEB_RUN_TOOL,
    WEB_SEARCH_TOOL,
)
from pulsing.forge.extension.memories.handlers import (
    handle_memories_add_ad_hoc_note,
    handle_memories_list,
    handle_memories_read,
    handle_memories_search,
)
from pulsing.forge.extension.skills.handlers import (
    handle_skills_list,
    handle_skills_read,
)
from pulsing.forge.extension.web_run.handlers import handle_web_run
from pulsing.forge.extension.web_search.handlers import handle_web_search
from pulsing.forge.result import ToolResult

_HANDLERS = {
    WEB_RUN_TOOL: handle_web_run,
    SKILLS_LIST_TOOL: handle_skills_list,
    SKILLS_READ_TOOL: handle_skills_read,
    MEMORIES_LIST_TOOL: handle_memories_list,
    MEMORIES_READ_TOOL: handle_memories_read,
    MEMORIES_SEARCH_TOOL: handle_memories_search,
    MEMORIES_ADD_AD_HOC_NOTE_TOOL: handle_memories_add_ad_hoc_note,
    WEB_SEARCH_TOOL: handle_web_search,
}


def dispatch_extension_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    ctx: ToolCallContext,
) -> ToolResult:
    impl = _HANDLERS.get(name)
    if impl is None:
        return ToolResult(content=f"unknown extension tool: {name}", is_error=True)
    return impl(ctx=ctx, **dict(arguments))


__all__ = ["EXTENSION_TOOL_NAMES", "dispatch_extension_tool"]
