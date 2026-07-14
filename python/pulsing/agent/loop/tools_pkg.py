# SPDX-License-Identifier: Apache-2.0
"""Built-in filesystem / network tools."""

from __future__ import annotations

from typing import Any

from pulsing.agent.loop.permissions import PermissionChecker
from pulsing.agent.loop.tool_base import Tool, ToolResult
from pulsing.agent.loop.tools_impl import (
    impl_bash,
    impl_edit,
    impl_fetch_url,
    impl_glob,
    impl_grep,
    impl_read,
    impl_write,
)


def _json_schema_object(
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict:
    req = list(properties.keys()) if required is None else required
    return {
        "type": "object",
        "properties": properties,
        "required": req,
        "additionalProperties": False,
    }


class ReadTool(Tool):
    @property
    def name(self) -> str:
        return "Read"

    @property
    def description(self) -> str:
        return "Read a UTF-8 text file from disk (size-capped)."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object({"file_path": {"type": "string"}}, ["file_path"])

    def is_read_only(self) -> bool:
        return True

    def get_activity_description(self, **kwargs: Any) -> str | None:
        return f"Read {kwargs.get('file_path', '')}"

    def execute(self, **kwargs: Any) -> ToolResult:
        return impl_read(**kwargs)


class GlobTool(Tool):
    @property
    def name(self) -> str:
        return "Glob"

    @property
    def description(self) -> str:
        return "Glob files under a directory (non-recursive segments via pathlib)."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "Base directory"},
            },
            ["pattern", "path"],
        )

    def is_read_only(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> ToolResult:
        return impl_glob(**kwargs)


class GrepTool(Tool):
    @property
    def name(self) -> str:
        return "Grep"

    @property
    def description(self) -> str:
        return "Search files with a regex (Python ``re``), capped matches."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string", "description": "Optional fnmatch filter"},
            },
            ["pattern", "path"],
        )

    def is_read_only(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> ToolResult:
        return impl_grep(**kwargs)


class EditTool(Tool):
    @property
    def name(self) -> str:
        return "Edit"

    @property
    def description(self) -> str:
        return "Replace exactly one occurrence of old_string with new_string in a file."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        return impl_edit(**kwargs)


class WriteTool(Tool):
    @property
    def name(self) -> str:
        return "Write"

    @property
    def description(self) -> str:
        return "Write UTF-8 text to a file (creates parent directories)."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {"file_path": {"type": "string"}, "content": {"type": "string"}},
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        return impl_write(**kwargs)


class BashTool(Tool):
    @property
    def name(self) -> str:
        return "Bash"

    @property
    def description(self) -> str:
        return (
            "Run a shell command in the isolated worker. "
            "Sandbox policy is configured on the agent (off / restricted / bwrap); "
            "the worker uses subprocess without shell=True."
        )

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "command": {"type": "string"},
                "timeout_sec": {"type": "integer", "minimum": 1, "maximum": 3600},
            },
            ["command"],
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        return impl_bash(**kwargs)


class FetchUrlTool(Tool):
    @property
    def name(self) -> str:
        return "FetchUrl"

    @property
    def description(self) -> str:
        return (
            "HTTP(S) GET for a small text response. Requires env PULSING_CRAFT_FETCH_ALLOW "
            "comma-separated hostname allowlist. Capped size."
        )

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "url": {"type": "string"},
                "max_bytes": {"type": "integer", "minimum": 1024, "maximum": 262144},
            },
            ["url"],
        )

    def is_read_only(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> ToolResult:
        return impl_fetch_url(**kwargs)
