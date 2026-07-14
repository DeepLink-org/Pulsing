# SPDX-License-Identifier: Apache-2.0
"""QuestReport tool — agents update puzzle status in cluster.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pulsing.agent.loop.tool_base import Tool, ToolResult
from pulsing.agent.loop.tools_pkg import _json_schema_object
from pulsing.agent.workspace.quest import (
    QUEST_STATUSES,
    format_quest_brief,
    update_quest_status,
)


class QuestReportTool(Tool):
    @property
    def name(self) -> str:
        return "QuestReport"

    @property
    def description(self) -> str:
        return (
            "Report progress on a workspace quest (puzzle). "
            "Updates status in .pulsing/cluster.json."
        )

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "quest_id": {
                    "type": "string",
                    "description": "Quest id (puzzle id from cluster.json).",
                },
                "status": {
                    "type": "string",
                    "enum": sorted(QUEST_STATUSES),
                    "description": "New quest status.",
                },
                "note": {
                    "type": "string",
                    "description": "Optional progress note.",
                },
            },
            ["quest_id", "status"],
        )

    def is_read_only(self) -> bool:
        return False

    def execute(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("QuestReport runs on Agent.")


async def tool_quest_report(agent: Any, kwargs: dict[str, Any]) -> ToolResult:
    qid = str(kwargs.get("quest_id") or kwargs.get("id") or "").strip()
    status = str(kwargs.get("status") or "").strip().lower()
    note = str(kwargs.get("note") or "").strip()
    if not qid:
        return ToolResult(content="QuestReport: quest_id required.", is_error=True)
    if not status:
        return ToolResult(content="QuestReport: status required.", is_error=True)
    root = Path(agent._cwd)
    reporter = agent._cluster_short_name or "agent"
    try:
        updated = update_quest_status(
            root,
            qid,
            status=status,
            note=note,
            reporter=reporter,
        )
    except KeyError as e:
        return ToolResult(content=str(e), is_error=True)
    except ValueError as e:
        return ToolResult(content=str(e), is_error=True)
    except OSError as e:
        return ToolResult(content=f"QuestReport failed: {e!r}", is_error=True)
    payload = {
        "quest_id": qid,
        "status": updated.get("status"),
        "summary": format_quest_brief(qid, updated),
    }
    if note:
        payload["note"] = note
    return ToolResult(content=json.dumps(payload, ensure_ascii=False))
