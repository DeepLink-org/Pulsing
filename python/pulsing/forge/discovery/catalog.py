# SPDX-License-Identifier: Apache-2.0
"""Deferred tool catalog + Codex plugin loading."""

from __future__ import annotations

from dataclasses import dataclass, field

from pulsing.forge.discovery.bm25 import bm25_scores
from pulsing.forge.discovery.discoverable import (
    DiscoverablePlugin,
    DiscoverableTool,
    DiscoverableToolAction,
    DiscoverableToolType,
    collect_discoverable_tools,
    request_plugin_install_entry,
)
from pulsing.forge.discovery.entries import TOOL_SEARCH_DEFAULT_LIMIT, DeferredToolEntry
from pulsing.forge.discovery.install import (
    deferred_tools_for_installed_plugin,
    execute_plugin_install,
)
from pulsing.forge.discovery.plugin_store import PluginStore, load_plugin_state


class ToolCatalogRefreshError(RuntimeError):
    """refresh_from_codex() could not enumerate the plugin cache/marketplaces."""


@dataclass
class ToolCatalog:
    deferred: list[DeferredToolEntry] = field(default_factory=list)
    discoverable: list[DiscoverablePlugin] = field(default_factory=list)
    discoverable_tools: list[DiscoverableTool] = field(default_factory=list)
    installed_plugin_ids: set[str] = field(default_factory=set)

    def register_deferred(self, entry: DeferredToolEntry) -> None:
        self.deferred = [e for e in self.deferred if e.name != entry.name]
        self.deferred.append(entry)

    def refresh_from_codex(self, extra_dirs: list[str] | None = None) -> None:
        """Rescan marketplaces + the installed-plugin cache.

        Raises `ToolCatalogRefreshError` if the plugin cache itself can't be read
        (permission/disk errors). A single plugin with a corrupt manifest is
        skipped rather than failing the whole refresh.
        """
        _ = extra_dirs
        state = load_plugin_state()
        configured = set(state.get("enabled_plugins") or [])
        store = PluginStore()

        try:
            installed_ids = store.list_installed_plugin_ids()
            discoverable_tools = collect_discoverable_tools(
                configured_plugin_ids=configured
            )
        except OSError as e:
            raise ToolCatalogRefreshError(
                f"failed to refresh plugin catalog: {e}"
            ) from e

        self.installed_plugin_ids = {pid.id for pid in installed_ids}
        self.discoverable_tools = discoverable_tools
        self.discoverable = [
            t for t in self.discoverable_tools if isinstance(t, DiscoverablePlugin)
        ]

        seen: set[str] = set()
        self.deferred = []
        for pid in installed_ids:
            try:
                entries = deferred_tools_for_installed_plugin(pid.id)
            except (OSError, ValueError):
                continue  # corrupt/unreadable plugin manifest — skip, don't fail the refresh
            for entry in entries:
                if entry.name not in seen:
                    seen.add(entry.name)
                    self.register_deferred(entry)

    def load_codex_plugins(self, extra_dirs: list[str] | None = None) -> None:
        self.refresh_from_codex(extra_dirs)

    def search(
        self, query: str, limit: int = TOOL_SEARCH_DEFAULT_LIMIT
    ) -> list[DeferredToolEntry]:
        docs = [e.search_text for e in self.deferred]
        scores = bm25_scores(query, docs)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        out: list[DeferredToolEntry] = []
        for idx, score in ranked:
            if score <= 0:
                break
            if len(out) >= limit:
                break
            out.append(self.deferred[idx])
        return out

    def list_installable_entries(self) -> list[dict]:
        return [request_plugin_install_entry(t) for t in self.discoverable_tools]

    def list_installable(self) -> list[DiscoverablePlugin]:
        return list(self.discoverable)

    def find_discoverable(
        self, tool_type: DiscoverableToolType, tool_id: str
    ) -> DiscoverableTool | None:
        for tool in self.discoverable_tools:
            if tool.tool_type == tool_type and tool.id == tool_id:
                return tool
        return None

    def find_plugin(self, plugin_id: str) -> DiscoverablePlugin | None:
        for p in self.discoverable:
            if p.id == plugin_id:
                return p
        return None

    def install_plugin(
        self,
        plugin_id: str,
        *,
        user_confirmed: bool,
        suggest_reason: str = "Model requested plugin install",
    ) -> list[DeferredToolEntry]:
        if not user_confirmed:
            return []
        outcome = execute_plugin_install(
            tool_type=DiscoverableToolType.PLUGIN,
            action_type=DiscoverableToolAction.INSTALL,
            tool_id=plugin_id,
            suggest_reason=suggest_reason,
            user_confirmed=user_confirmed,
        )
        if not outcome.user_confirmed or outcome.deferred_tools is None:
            return []
        self.refresh_from_codex()
        return list(outcome.deferred_tools)

    def install_plugin_legacy(self, plugin_id: str) -> list[DeferredToolEntry]:
        from pulsing.forge.discovery.install import install_plugin_from_marketplace

        install_plugin_from_marketplace(plugin_id)
        entries = deferred_tools_for_installed_plugin(plugin_id)
        self.refresh_from_codex()
        return entries
