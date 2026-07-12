# SPDX-License-Identifier: Apache-2.0
"""Parse Codex `.codex-plugin/plugin.json` manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pulsing.forge.discovery.codex_paths import PLUGIN_MANIFEST_RELATIVE_PATHS
from pulsing.forge.discovery.entries import DeferredToolEntry


def find_plugin_manifest_path(plugin_root: Path) -> Path | None:
    for rel in PLUGIN_MANIFEST_RELATIVE_PATHS:
        candidate = plugin_root / rel
        if candidate.is_file():
            return candidate
    return None


@dataclass
class CodexPluginManifest:
    name: str
    version: str | None = None
    description: str | None = None
    keywords: list[str] = field(default_factory=list)
    skills_path: str | None = None
    mcp_servers_path: str | None = None
    apps_path: str | None = None
    hooks_path: str | None = None
    display_name: str | None = None
    mcp_server_names: list[str] = field(default_factory=list)
    app_connector_ids: list[str] = field(default_factory=list)
    manifest_path: Path | None = None
    plugin_root: Path | None = None

    @property
    def has_skills(self) -> bool:
        return bool(self.skills_path)

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *, manifest_path: Path, plugin_root: Path
    ) -> CodexPluginManifest:
        interface = (
            raw.get("interface") if isinstance(raw.get("interface"), dict) else {}
        )
        display = interface.get("displayName") or interface.get("display_name")
        desc = (
            raw.get("description")
            or interface.get("shortDescription")
            or interface.get("short_description")
        )
        mcp_path = raw.get("mcpServers") or raw.get("mcp_servers")
        apps_path = raw.get("apps")
        skills = raw.get("skills")
        hooks = raw.get("hooks")
        mcp_names = _extract_mcp_server_names(plugin_root, mcp_path)
        connector_ids = _extract_app_connector_ids(plugin_root, apps_path)
        name = str(raw.get("name") or plugin_root.name)
        version = raw.get("version")
        return cls(
            name=name,
            version=str(version) if version is not None else None,
            description=str(desc) if desc is not None else None,
            keywords=[str(k) for k in (raw.get("keywords") or []) if str(k).strip()],
            skills_path=str(skills) if skills else None,
            mcp_servers_path=str(mcp_path) if mcp_path else None,
            apps_path=str(apps_path) if apps_path else None,
            hooks_path=str(hooks) if hooks else None,
            display_name=str(display) if display else None,
            mcp_server_names=mcp_names,
            app_connector_ids=connector_ids,
            manifest_path=manifest_path,
            plugin_root=plugin_root,
        )

    def deferred_tools(self) -> list[DeferredToolEntry]:
        """Deferred stubs for MCP servers declared by the plugin (tool_search index)."""
        out: list[DeferredToolEntry] = []
        plugin_id = self.name
        for server in self.mcp_server_names:
            ns = f"mcp__{server}"
            entry = DeferredToolEntry.from_function(
                ns,
                f"MCP server {server} from plugin {self.display_name or self.name}",
                {"type": "object", "properties": {}},
                defer_loading=True,
                namespace=ns,
                plugin_id=plugin_id,
                source="codex_mcp_server",
            )
            out.append(entry)
        return out


def load_codex_manifest(plugin_root: Path) -> CodexPluginManifest:
    manifest_path = find_plugin_manifest_path(plugin_root)
    if manifest_path is None:
        raise ValueError(f"missing plugin manifest under {plugin_root}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{manifest_path}: manifest must be a JSON object")
    return CodexPluginManifest.from_dict(
        raw, manifest_path=manifest_path, plugin_root=plugin_root
    )


def _resolve_relative(plugin_root: Path, ref: str | None) -> Path | None:
    if not ref:
        return None
    path = (plugin_root / ref).resolve()
    return path if path.exists() else None


def _extract_mcp_server_names(plugin_root: Path, mcp_ref: Any) -> list[str]:
    path = _resolve_relative(plugin_root, str(mcp_ref) if mcp_ref else None)
    if path is None or not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    servers = raw.get("mcpServers") if isinstance(raw, dict) else raw
    if isinstance(servers, dict):
        return sorted(str(k) for k in servers if str(k).strip())
    return []


def _extract_app_connector_ids(plugin_root: Path, apps_ref: Any) -> list[str]:
    path = _resolve_relative(plugin_root, str(apps_ref) if apps_ref else None)
    if path is None or not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, dict):
        return []
    connectors = raw.get("connectors") or raw.get("apps") or raw
    if isinstance(connectors, list):
        return [
            str(c.get("id", c))
            for c in connectors
            if isinstance(c, dict) and c.get("id")
        ]
    if isinstance(connectors, dict):
        return sorted(str(k) for k in connectors)
    return []
