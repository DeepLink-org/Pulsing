# SPDX-License-Identifier: Apache-2.0
"""Codex-aligned plugin install orchestration."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from pulsing.forge.discovery.codex_manifest import load_codex_manifest
from pulsing.forge.discovery.discoverable import (
    DiscoverableToolAction,
    DiscoverableToolType,
    find_discoverable_tool,
)
from pulsing.forge.discovery.entries import DeferredToolEntry
from pulsing.forge.discovery.marketplace import (
    find_installable_plugin,
    materialize_plugin_source,
)
from pulsing.forge.discovery.plugin_id import PluginId
from pulsing.forge.discovery.plugin_store import (
    PluginInstallResult,
    PluginStore,
    enable_plugin,
)


@dataclass
class CodexInstallOutcome:
    completed: bool
    user_confirmed: bool
    tool_type: DiscoverableToolType
    action_type: DiscoverableToolAction
    tool_id: str
    tool_name: str
    suggest_reason: str
    tools_registered: int = 0
    install_result: PluginInstallResult | None = None
    deferred_tools: list[DeferredToolEntry] | None = None

    def to_result_json(self) -> dict:
        return {
            "completed": self.completed,
            "user_confirmed": self.user_confirmed,
            "tool_type": self.tool_type.value,
            "action_type": self.action_type.value,
            "tool_id": self.tool_id,
            "tool_name": self.tool_name,
            "suggest_reason": self.suggest_reason,
            "tools_registered": self.tools_registered,
        }


def install_plugin_from_marketplace(plugin_id: str) -> PluginInstallResult:
    entry = find_installable_plugin(plugin_id)
    pid = entry.plugin_id
    source = materialize_plugin_source(entry)
    store = PluginStore()
    try:
        result = store.install(source, pid)
    finally:
        if source.name.startswith("forge-plugin-src-"):
            shutil.rmtree(source, ignore_errors=True)
    enable_plugin(pid)
    return result


def deferred_tools_for_installed_plugin(plugin_id: str) -> list[DeferredToolEntry]:
    pid = PluginId.parse(plugin_id)
    root = PluginStore().active_plugin_root(pid)
    if root is None:
        return []
    manifest = load_codex_manifest(root)
    return manifest.deferred_tools()


def verify_plugin_install_completed(plugin_id: str) -> bool:
    pid = PluginId.parse(plugin_id)
    if pid.marketplace_name.endswith("-remote"):
        return True
    return PluginStore().is_installed(pid)


def execute_plugin_install(
    *,
    tool_type: DiscoverableToolType,
    action_type: DiscoverableToolAction,
    tool_id: str,
    suggest_reason: str,
    user_confirmed: bool,
) -> CodexInstallOutcome:
    if action_type != DiscoverableToolAction.INSTALL:
        raise ValueError(
            'plugin install requests currently support only action_type="install"'
        )
    if not suggest_reason.strip():
        raise ValueError("request_plugin_install requires non-empty suggest_reason")
    if not user_confirmed:
        tool = find_discoverable_tool(tool_type, tool_id)
        if tool is None:
            raise ValueError(f"unknown plugin {tool_id!r}")
        return CodexInstallOutcome(
            completed=True,
            user_confirmed=False,
            tool_type=tool_type,
            action_type=action_type,
            tool_id=tool_id,
            tool_name=tool.name,
            suggest_reason=suggest_reason,
            tools_registered=0,
        )

    tool = find_discoverable_tool(tool_type, tool_id)
    if tool is None:
        raise ValueError(f"unknown plugin {tool_id!r}")
    tool_name = tool.name

    if tool_type == DiscoverableToolType.PLUGIN:
        install_result = install_plugin_from_marketplace(tool_id)
        deferred = deferred_tools_for_installed_plugin(tool_id)
        completed = verify_plugin_install_completed(tool_id)
        return CodexInstallOutcome(
            completed=completed,
            user_confirmed=True,
            tool_type=tool_type,
            action_type=action_type,
            tool_id=tool_id,
            tool_name=tool_name,
            suggest_reason=suggest_reason,
            tools_registered=len(deferred),
            install_result=install_result,
            deferred_tools=deferred,
        )

    # Connector install is browser/OAuth — Craft must open install_url separately.
    return CodexInstallOutcome(
        completed=False,
        user_confirmed=True,
        tool_type=tool_type,
        action_type=action_type,
        tool_id=tool_id,
        tool_name=tool_name,
        suggest_reason=suggest_reason,
    )
