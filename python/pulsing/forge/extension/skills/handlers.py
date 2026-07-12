# SPDX-License-Identifier: Apache-2.0
"""Handlers for ``skills.list`` / ``skills.read``."""

from __future__ import annotations

import json
from typing import Any

from pulsing.forge.context import ToolCallContext
from pulsing.forge.extension.skills.catalog import list_skills, read_skill
from pulsing.forge.result import ToolResult


def handle_skills_list(*, ctx: ToolCallContext, **_: Any) -> ToolResult:
    try:
        entries = [e.to_dict() for e in list_skills(ctx.cwd)]
    except OSError:
        entries = []
    return ToolResult(
        content=json.dumps({"skills": entries}, indent=2, ensure_ascii=False),
        structured={"skills": entries},
    )


def handle_skills_read(
    *,
    ctx: ToolCallContext,
    name: str = "",
    path: str = "",
    **_: Any,
) -> ToolResult:
    target_name = name.strip()
    target_path = path.strip()
    if not target_name and not target_path:
        return ToolResult(content="name or path is required", is_error=True)
    try:
        text = read_skill(cwd=ctx.cwd, name=target_name, path=target_path)
    except FileNotFoundError as exc:
        return ToolResult(content=str(exc), is_error=True)
    except OSError as exc:
        return ToolResult(content=str(exc), is_error=True)
    return ToolResult(content=text)
