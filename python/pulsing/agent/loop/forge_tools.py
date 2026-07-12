# SPDX-License-Identifier: Apache-2.0
"""LLM schemas for the unified Pulsing Forge tool surface."""

from __future__ import annotations

from typing import Any

from pulsing.agent.loop.tool_base import Tool
from pulsing.agent.loop.tools_pkg import (
    BashTool,
    EditTool,
    GlobTool,
    GrepTool,
    ReadTool,
    WriteTool,
    _json_schema_object,
)
from pulsing.forge.integrated import FORGE_TOOL_NAMES


class ShellCommandTool(Tool):
    @property
    def name(self) -> str:
        return "shell_command"

    @property
    def description(self) -> str:
        return "Run a shell command (Codex-compatible: command, workdir, timeout_ms, sandbox_permissions)."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "command": {"type": "string"},
                "workdir": {"type": "string"},
                "timeout_ms": {"type": "integer", "minimum": 1000},
                "login": {"type": "boolean"},
                "sandbox_permissions": {
                    "type": "string",
                    "enum": ["require_escalated", "with_additional_permissions"],
                },
            },
            ["command"],
        )

    def execute(self, **kwargs: Any):
        raise RuntimeError("shell_command runs in Forge worker")


class ExecCommandTool(Tool):
    @property
    def name(self) -> str:
        return "exec_command"

    @property
    def description(self) -> str:
        return "Start a unified exec session (PTY by default). Returns session_id while running."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "cmd": {"type": "string"},
                "workdir": {"type": "string"},
                "tty": {"type": "boolean"},
                "yield_time_ms": {"type": "integer"},
                "max_output_tokens": {"type": "integer"},
            },
            ["cmd"],
        )

    def execute(self, **kwargs: Any):
        raise RuntimeError("exec_command runs in Forge worker")


class WriteStdinTool(Tool):
    @property
    def name(self) -> str:
        return "write_stdin"

    @property
    def description(self) -> str:
        return "Write to a unified exec session stdin (or send \\x03 to interrupt)."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "session_id": {"type": "integer"},
                "chars": {"type": "string"},
                "yield_time_ms": {"type": "integer"},
            },
            ["session_id", "chars"],
        )

    def execute(self, **kwargs: Any):
        raise RuntimeError("write_stdin runs in Forge worker")


class ApplyPatchTool(Tool):
    @property
    def name(self) -> str:
        return "apply_patch"

    @property
    def description(self) -> str:
        return "Apply a structured patch to the workspace."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object({"patch": {"type": "string"}}, ["patch"])

    def execute(self, **kwargs: Any):
        raise RuntimeError("apply_patch runs in Forge worker")


class ViewImageTool(Tool):
    @property
    def name(self) -> str:
        return "view_image"

    @property
    def description(self) -> str:
        return "Attach a local image for the model (returns structured content_items)."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "path": {"type": "string"},
                "detail": {"type": "string", "enum": ["high", "original"]},
            },
            ["path"],
        )

    def execute(self, **kwargs: Any):
        raise RuntimeError("view_image runs in Forge worker")


class UpdatePlanTool(Tool):
    @property
    def name(self) -> str:
        return "update_plan"

    @property
    def description(self) -> str:
        return "Update the collaborative task plan."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "explanation": {"type": "string"},
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
            },
            ["plan"],
        )

    def execute(self, **kwargs: Any):
        raise RuntimeError("update_plan runs on Forge host")


class NewContextTool(Tool):
    @property
    def name(self) -> str:
        return "new_context"

    @property
    def description(self) -> str:
        return "Request a fresh context window without summarizing history."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object({})

    def execute(self, **kwargs: Any):
        raise RuntimeError("new_context runs on Forge host")


class GetContextRemainingTool(Tool):
    @property
    def name(self) -> str:
        return "get_context_remaining"

    @property
    def description(self) -> str:
        return "Report remaining token budget if configured."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object({})

    def execute(self, **kwargs: Any):
        raise RuntimeError("get_context_remaining runs on Forge host")


class RequestUserInputTool(Tool):
    @property
    def name(self) -> str:
        return "request_user_input"

    @property
    def description(self) -> str:
        return "Ask the user structured questions and wait for answers (Codex-compatible schema)."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "header": {"type": "string"},
                            "question": {"type": "string"},
                            "isOther": {"type": "boolean"},
                            "isSecret": {"type": "boolean"},
                            "options": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                    "required": ["label"],
                                },
                            },
                        },
                        "required": ["id", "header", "question"],
                    },
                },
                "autoResolutionMs": {
                    "type": "integer",
                    "minimum": 60000,
                    "maximum": 240000,
                    "description": "Optional auto-resolve window (ms). On timeout, use first/recommended option per question.",
                },
            },
            ["questions"],
        )

    def execute(self, **kwargs: Any):
        raise RuntimeError("request_user_input runs on Forge host")


class ToolSearchTool(Tool):
    @property
    def name(self) -> str:
        return "tool_search"

    @property
    def description(self) -> str:
        return "Search deferred tools (BM25) before loading them into the session."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 32},
            },
            ["query"],
        )

    def execute(self, **kwargs: Any):
        raise RuntimeError("tool_search runs on Forge host")


class ListAvailablePluginsTool(Tool):
    @property
    def name(self) -> str:
        return "list_available_plugins_to_install"

    @property
    def description(self) -> str:
        return (
            "List Codex-compatible plugins available to install from local plugin dirs."
        )

    @property
    def input_schema(self) -> dict:
        return _json_schema_object({})

    def execute(self, **kwargs: Any):
        raise RuntimeError("list_available_plugins_to_install runs on Forge host")


class RequestPluginInstallTool(Tool):
    @property
    def name(self) -> str:
        return "request_plugin_install"

    @property
    def description(self) -> str:
        return "Request user approval to install a Codex-compatible plugin and register its tools."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "tool_type": {"type": "string", "enum": ["connector", "plugin"]},
                "action_type": {"type": "string", "enum": ["install", "enable"]},
                "tool_id": {"type": "string"},
                "suggest_reason": {"type": "string"},
            },
            ["tool_type", "action_type", "tool_id", "suggest_reason"],
        )

    def execute(self, **kwargs: Any):
        raise RuntimeError("request_plugin_install runs on Forge host")


class RequestPermissionsTool(Tool):
    @property
    def name(self) -> str:
        return "request_permissions"

    @property
    def description(self) -> str:
        return "Request additional filesystem/network permissions (Codex-aligned)."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "reason": {"type": "string"},
                "permissions": {
                    "type": "object",
                    "properties": {
                        "network": {"type": "object"},
                        "file_system": {"type": "object"},
                    },
                },
            },
            ["permissions"],
        )

    def execute(self, **kwargs: Any):
        raise RuntimeError("request_permissions runs on Forge host")


class ListMcpResourcesTool(Tool):
    @property
    def name(self) -> str:
        return "list_mcp_resources"

    @property
    def description(self) -> str:
        return "List MCP resources from configured servers."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object({})

    def execute(self, **kwargs: Any):
        raise RuntimeError("list_mcp_resources runs on Forge host")


class ListMcpResourceTemplatesTool(Tool):
    @property
    def name(self) -> str:
        return "list_mcp_resource_templates"

    @property
    def description(self) -> str:
        return "List MCP resource URI templates from configured servers."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "server": {"type": "string", "description": "MCP server name"},
                "cursor": {"type": "string", "description": "Pagination cursor"},
            },
            ["server"],
        )

    def execute(self, **kwargs: Any):
        raise RuntimeError("list_mcp_resource_templates runs on Forge host")


class ReadMcpResourceTool(Tool):
    @property
    def name(self) -> str:
        return "read_mcp_resource"

    @property
    def description(self) -> str:
        return "Read an MCP resource by URI."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "server": {"type": "string", "description": "MCP server name"},
                "uri": {"type": "string", "description": "Resource URI"},
            },
            ["server", "uri"],
        )

    def execute(self, **kwargs: Any):
        raise RuntimeError("read_mcp_resource runs on Forge host")


class ExecTool(Tool):
    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Run Python source in Code Mode (returns cell_id while running)."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "source": {"type": "string"},
                "yield_time_ms": {"type": "integer"},
                "max_output_tokens": {"type": "integer"},
            },
            ["source"],
        )

    def execute(self, **kwargs: Any):
        raise RuntimeError("exec runs on Forge host")


class WaitTool(Tool):
    @property
    def name(self) -> str:
        return "wait"

    @property
    def description(self) -> str:
        return "Wait for a Code Mode cell or terminate it."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "cell_id": {"type": "string"},
                "yield_time_ms": {"type": "integer"},
                "max_tokens": {"type": "integer"},
                "terminate": {"type": "boolean"},
            },
            ["cell_id"],
        )

    def execute(self, **kwargs: Any):
        raise RuntimeError("wait runs on Forge host")


class WebRunTool(Tool):
    @property
    def name(self) -> str:
        return "web.run"

    @property
    def description(self) -> str:
        return "Client web search / open / find (Codex SearchCommands wire)."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "search_query": {"type": "string"},
                "image_query": {"type": "string"},
                "open": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ref_id": {"type": "string"},
                            "url": {"type": "string"},
                        },
                    },
                },
                "find": {"type": "array", "items": {"type": "object"}},
            },
        )

    def execute(self, **kwargs: Any):
        raise RuntimeError("web.run runs on Forge host")


class SkillsListTool(Tool):
    @property
    def name(self) -> str:
        return "skills.list"

    @property
    def description(self) -> str:
        return "List available agent skills in the workspace."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object({})

    def execute(self, **kwargs: Any):
        raise RuntimeError("skills.list runs on Forge host")


class SkillsReadTool(Tool):
    @property
    def name(self) -> str:
        return "skills.read"

    @property
    def description(self) -> str:
        return "Read a skill prompt by name or path."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "name": {"type": "string"},
                "path": {"type": "string"},
            },
        )

    def execute(self, **kwargs: Any):
        raise RuntimeError("skills.read runs on Forge host")


class MemoriesListTool(Tool):
    @property
    def name(self) -> str:
        return "memories.list"

    @property
    def description(self) -> str:
        return "List memory files under the Codex memories root."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "path": {"type": "string"},
                "cursor": {"type": "string"},
                "max_results": {"type": "integer"},
            },
        )

    def execute(self, **kwargs: Any):
        raise RuntimeError("memories.list runs on Forge host")


class MemoriesReadTool(Tool):
    @property
    def name(self) -> str:
        return "memories.read"

    @property
    def description(self) -> str:
        return "Read a memory file with optional line/token limits."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "path": {"type": "string"},
                "line_offset": {"type": "integer"},
                "max_lines": {"type": "integer"},
                "max_tokens": {"type": "integer"},
            },
            ["path"],
        )

    def execute(self, **kwargs: Any):
        raise RuntimeError("memories.read runs on Forge host")


class MemoriesSearchTool(Tool):
    @property
    def name(self) -> str:
        return "memories.search"

    @property
    def description(self) -> str:
        return "Search memory files by query."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "query": {"type": "string"},
                "queries": {"type": "array", "items": {"type": "string"}},
                "path": {"type": "string"},
                "cursor": {"type": "string"},
                "max_results": {"type": "integer"},
            },
        )

    def execute(self, **kwargs: Any):
        raise RuntimeError("memories.search runs on Forge host")


class MemoriesAddAdHocNoteTool(Tool):
    @property
    def name(self) -> str:
        return "memories.add_ad_hoc_note"

    @property
    def description(self) -> str:
        return "Append an ad-hoc note to the memories store."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "content": {"type": "string"},
                "path": {"type": "string"},
            },
            ["content"],
        )

    def execute(self, **kwargs: Any):
        raise RuntimeError("memories.add_ad_hoc_note runs on Forge host")


class WebSearchTool(Tool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Hosted Provider web search (enable in model config; not executed locally)."
        )

    @property
    def input_schema(self) -> dict:
        return _json_schema_object({"query": {"type": "string"}}, ["query"])

    def is_read_only(self) -> bool:
        return True

    def execute(self, **kwargs: Any):
        raise RuntimeError("web_search runs on the LLM provider")


def all_forge_tool_templates() -> list[Tool]:
    return [
        ReadTool(),
        GlobTool(),
        GrepTool(),
        EditTool(),
        WriteTool(),
        BashTool(),
        ShellCommandTool(),
        ExecCommandTool(),
        WriteStdinTool(),
        ApplyPatchTool(),
        ViewImageTool(),
        UpdatePlanTool(),
        NewContextTool(),
        GetContextRemainingTool(),
        RequestUserInputTool(),
        RequestPermissionsTool(),
        ToolSearchTool(),
        ListAvailablePluginsTool(),
        RequestPluginInstallTool(),
        ListMcpResourcesTool(),
        ListMcpResourceTemplatesTool(),
        ReadMcpResourceTool(),
        ExecTool(),
        WaitTool(),
        WebRunTool(),
        SkillsListTool(),
        SkillsReadTool(),
        MemoriesListTool(),
        MemoriesReadTool(),
        MemoriesSearchTool(),
        MemoriesAddAdHocNoteTool(),
        WebSearchTool(),
    ]


def assert_forge_tool_coverage() -> None:
    from pulsing.forge.tool_coverage import (
        assert_forge_tool_coverage as _assert_registry,
    )

    _assert_registry()
    names = {t.name for t in all_forge_tool_templates()}
    assert names == set(FORGE_TOOL_NAMES), (names, FORGE_TOOL_NAMES)
