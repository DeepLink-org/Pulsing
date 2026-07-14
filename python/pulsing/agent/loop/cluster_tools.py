# SPDX-License-Identifier: Apache-2.0
"""Anthropic tool schemas for cluster peer discovery and messaging."""

from __future__ import annotations

from typing import Any

from pulsing.agent.loop.tool_base import Tool, ToolResult
from pulsing.agent.loop.tools_pkg import _json_schema_object


class ListClusterAgentsTool(Tool):
    @property
    def name(self) -> str:
        return "ListClusterAgents"

    @property
    def description(self) -> str:
        return (
            "List other agents in the local Pulsing cluster (workspace gossip names). "
            "Use before MessageClusterAgent to find peer ids."
        )

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "local_only": {
                    "type": "boolean",
                    "description": "If true, only list agents on this node.",
                },
            },
            [],
        )

    def is_read_only(self) -> bool:
        return False

    def execute(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("ListClusterAgents runs on Agent._cluster_tool.")


class MessageClusterAgentTool(Tool):
    @property
    def name(self) -> str:
        return "MessageClusterAgent"

    @property
    def description(self) -> str:
        return (
            "Send a message to another cluster agent by short name (e.g. coder). "
            "The peer runs an LLM chat round; set wait=true to block for a reply."
        )

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "agent": {
                    "type": "string",
                    "description": "Target agent short name (also target/to/name).",
                },
                "message": {
                    "type": "string",
                    "description": "User-style instruction for the peer agent.",
                },
                "wait": {
                    "type": "boolean",
                    "description": "Block until peer finishes (default false; use true only when reply needed).",
                },
                "timeout": {
                    "type": "number",
                    "description": "Max seconds to wait for peer reply (default 600).",
                },
            },
            ["message"],
        )

    def is_read_only(self) -> bool:
        return False

    def execute(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("MessageClusterAgent runs on Agent._cluster_tool.")
