# SPDX-License-Identifier: Apache-2.0
"""Minimal Forge tool schemas (no agent.loop dependency)."""

from __future__ import annotations

from typing import Any

from pulsing.forge.tool_schema import json_schema_object


def minimal_forge_tool_definitions() -> list[dict[str, Any]]:
    """Core tools for custom agent frameworks without pulling agent.loop."""
    return [
        {
            "name": "update_plan",
            "description": "Publish or revise the task plan visible to the user.",
            "input_schema": json_schema_object(
                {
                    "plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                            },
                            "required": ["step", "status"],
                        },
                    },
                    "explanation": {"type": "string"},
                },
                required=["plan"],
            ),
        },
        {
            "name": "Glob",
            "description": "Find files matching a glob pattern under a directory.",
            "input_schema": json_schema_object(
                {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                required=["pattern"],
            ),
        },
        {
            "name": "Read",
            "description": "Read a text file from the workspace.",
            "input_schema": json_schema_object(
                {"file_path": {"type": "string"}},
                required=["file_path"],
            ),
        },
        {
            "name": "Grep",
            "description": "Search file contents with a regex pattern.",
            "input_schema": json_schema_object(
                {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string"},
                },
                required=["pattern"],
            ),
        },
        {
            "name": "shell_command",
            "description": "Run a shell command in the workspace.",
            "input_schema": json_schema_object(
                {
                    "command": {"type": "string"},
                    "workdir": {"type": "string"},
                },
                required=["command"],
            ),
        },
    ]
