# SPDX-License-Identifier: Apache-2.0
"""Codex-aligned discoverable tools (Connector + Plugin)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pulsing.forge.discovery.codex_manifest import load_codex_manifest
from pulsing.forge.discovery.codex_paths import TOOL_SUGGEST_PLUGIN_ALLOWLIST
from pulsing.forge.discovery.marketplace import InstallPolicy, list_marketplaces
from pulsing.forge.discovery.plugin_id import PluginId
from pulsing.forge.discovery.plugin_store import PluginStore

DESCRIPTION_MAX_LEN = 240


class DiscoverableToolType(str, Enum):
    CONNECTOR = "connector"
    PLUGIN = "plugin"


class DiscoverableToolAction(str, Enum):
    INSTALL = "install"
    ENABLE = "enable"


@dataclass
class DiscoverableConnector:
    id: str
    name: str
    description: str | None = None
    install_url: str | None = None
    is_accessible: bool = False

    @property
    def tool_type(self) -> DiscoverableToolType:
        return DiscoverableToolType.CONNECTOR


@dataclass
class DiscoverablePlugin:
    id: str
    name: str
    description: str | None
    remote_plugin_id: str | None
    has_skills: bool
    mcp_server_names: list[str]
    app_connector_ids: list[str]
    installed: bool = False
    marketplace_name: str | None = None
    manifest_path: Any = None  # Path | None — kept for legacy compat

    @property
    def tool_type(self) -> DiscoverableToolType:
        return DiscoverableToolType.PLUGIN

    def to_entry_json(self) -> dict[str, Any]:
        return request_plugin_install_entry(self)


DiscoverableTool = DiscoverableConnector | DiscoverablePlugin


def request_plugin_install_entry(tool: DiscoverableTool) -> dict[str, Any]:
    desc = tool.description
    if desc and len(desc) > DESCRIPTION_MAX_LEN:
        desc = desc[: DESCRIPTION_MAX_LEN - 1] + "…"
    if isinstance(tool, DiscoverableConnector):
        return {
            "id": tool.id,
            "name": tool.name,
            "description": desc,
            "tool_type": DiscoverableToolType.CONNECTOR.value,
            "has_skills": False,
            "mcp_server_names": [],
            "app_connector_ids": [],
        }
    return {
        "id": tool.id,
        "name": tool.name,
        "description": desc,
        "tool_type": DiscoverableToolType.PLUGIN.value,
        "has_skills": tool.has_skills,
        "mcp_server_names": list(tool.mcp_server_names),
        "app_connector_ids": list(tool.app_connector_ids),
    }


def collect_discoverable_plugins(
    *,
    configured_plugin_ids: set[str] | None = None,
    allowlist_only: bool = False,
) -> list[DiscoverablePlugin]:
    store = PluginStore()
    configured = configured_plugin_ids or set()
    discover_all = os.environ.get("FORGE_PLUGIN_DISCOVER_ALL", "").lower() in (
        "1",
        "true",
        "yes",
    )
    out: list[DiscoverablePlugin] = []

    for marketplace in list_marketplaces():
        for entry in marketplace.plugins:
            pid = entry.plugin_id
            if store.is_installed(pid):
                continue
            if entry.installation == InstallPolicy.NOT_AVAILABLE:
                continue
            in_allowlist = pid.id in TOOL_SUGGEST_PLUGIN_ALLOWLIST
            is_configured = pid.id in configured
            if not discover_all and not in_allowlist and not is_configured:
                continue
            manifest = None
            try:
                if entry.source.local_path and entry.source.local_path.is_dir():
                    manifest = load_codex_manifest(entry.source.local_path)
            except (OSError, ValueError):
                manifest = None
            name = manifest.display_name or manifest.name if manifest else entry.name
            description = manifest.description if manifest else None
            out.append(
                DiscoverablePlugin(
                    id=pid.id,
                    name=name,
                    description=description,
                    remote_plugin_id=_remote_plugin_id(pid),
                    has_skills=bool(manifest and manifest.has_skills),
                    mcp_server_names=(
                        list(manifest.mcp_server_names) if manifest else []
                    ),
                    app_connector_ids=(
                        list(manifest.app_connector_ids) if manifest else []
                    ),
                    installed=False,
                    marketplace_name=marketplace.name,
                )
            )
    return out


def collect_discoverable_connectors() -> list[DiscoverableConnector]:
    """Connector catalog requires ChatGPT Apps API — stub for Codex wire compatibility."""
    return []


def collect_discoverable_tools(
    *,
    configured_plugin_ids: set[str] | None = None,
) -> list[DiscoverableTool]:
    plugins = collect_discoverable_plugins(configured_plugin_ids=configured_plugin_ids)
    connectors = collect_discoverable_connectors()
    return [*connectors, *plugins]


def find_discoverable_tool(
    tool_type: DiscoverableToolType,
    tool_id: str,
    *,
    configured_plugin_ids: set[str] | None = None,
) -> DiscoverableTool | None:
    for tool in collect_discoverable_tools(configured_plugin_ids=configured_plugin_ids):
        if tool.tool_type == tool_type and tool.id == tool_id:
            return tool
    return None


def _remote_plugin_id(pid: PluginId) -> str | None:
    if pid.marketplace_name.endswith("-remote"):
        return f"plugins~Plugin_{pid.plugin_name}"
    return None


def build_plugin_install_elicitation_meta(
    tool: DiscoverablePlugin,
    *,
    suggest_reason: str,
    action_type: DiscoverableToolAction = DiscoverableToolAction.INSTALL,
) -> dict[str, Any]:
    """Codex `tool_suggestion` elicitation meta (tools/request_plugin_install.rs)."""
    meta: dict[str, Any] = {
        "codex_approval_kind": "tool_suggestion",
        "persist": "always",
        "tool_type": DiscoverableToolType.PLUGIN.value,
        "suggest_type": action_type.value,
        "suggest_reason": suggest_reason,
        "tool_id": tool.id,
        "tool_name": tool.name,
    }
    if tool.remote_plugin_id:
        meta["remote_plugin_id"] = tool.remote_plugin_id
    if tool.app_connector_ids:
        meta["app_connector_ids"] = tool.app_connector_ids
    return meta
