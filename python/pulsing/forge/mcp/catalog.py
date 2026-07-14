# SPDX-License-Identifier: Apache-2.0
"""MCP catalog loading — mirrors `pulsing-forge` MCP module (plugin + config.toml)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pulsing.forge.discovery.codex_manifest import (
    find_plugin_manifest_path,
    load_codex_manifest,
)
from pulsing.forge.discovery.codex_paths import codex_home, plugins_cache_root
from pulsing.forge.discovery.plugin_store import PluginStore


@dataclass
class McpServerEntry:
    name: str
    config: dict[str, Any]
    source: str
    plugin_id: str | None = None


@dataclass
class McpCatalogSnapshot:
    servers: dict[str, McpServerEntry] = field(default_factory=dict)


def _load_config_toml_servers() -> dict[str, dict[str, Any]]:
    path = codex_home() / "config.toml"
    if not path.is_file():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    block = raw.get("mcp_servers") or {}
    if not isinstance(block, dict):
        return {}
    return {str(k): dict(v) if isinstance(v, dict) else {} for k, v in block.items()}


def parse_plugin_mcp_file(
    plugin_root: Path, mcp_path: Path
) -> dict[str, dict[str, Any]]:
    """Parse `.mcp.json` — both `{mcpServers:{}}` and flat server map shapes."""
    raw = json.loads(mcp_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "mcpServers" in raw:
        servers = raw["mcpServers"]
    elif isinstance(raw, dict):
        servers = raw
    else:
        return {}
    if not isinstance(servers, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, cfg in servers.items():
        if isinstance(cfg, dict):
            out[str(name)] = _normalize_plugin_server_config(plugin_root, dict(cfg))
    return out


def _normalize_plugin_server_config(
    plugin_root: Path, cfg: dict[str, Any]
) -> dict[str, Any]:
    cfg.pop("type", None)
    oauth = cfg.pop("oauth", None)
    if isinstance(oauth, dict):
        oauth = dict(oauth)
        oauth.pop("callbackPort", None)
        if "clientId" in oauth and "client_id" not in oauth:
            oauth["client_id"] = oauth.pop("clientId")
        if oauth:
            cfg["oauth"] = oauth
    cwd = cfg.get("cwd")
    if isinstance(cwd, str) and cwd and not Path(cwd).is_absolute():
        cfg["cwd"] = str((plugin_root / cwd).resolve())
    return cfg


def load_plugin_mcp_servers() -> list[McpServerEntry]:
    entries: list[McpServerEntry] = []
    store = PluginStore()
    for pid in store.list_installed_plugin_ids():
        root = plugins_cache_root() / pid.marketplace / pid.name / pid.version
        manifest_path = find_plugin_manifest_path(root)
        if manifest_path is None:
            continue
        try:
            manifest = load_codex_manifest(root)
        except ValueError:
            continue
        mcp_ref = manifest.mcp_servers_path
        if not mcp_ref:
            continue
        mcp_path = (root / mcp_ref).resolve()
        if not mcp_path.is_file():
            continue
        plugin_id = f"{manifest.name}@{pid.marketplace}"
        for name, cfg in parse_plugin_mcp_file(root, mcp_path).items():
            entries.append(
                McpServerEntry(
                    name=name,
                    config=cfg,
                    source="plugin",
                    plugin_id=plugin_id,
                )
            )
    return entries


def load_mcp_catalog() -> McpCatalogSnapshot:
    """Plugin registrations first; `config.toml` overrides (Codex precedence)."""
    snap = McpCatalogSnapshot()
    for entry in load_plugin_mcp_servers():
        snap.servers[entry.name] = entry
    for name, cfg in _load_config_toml_servers().items():
        snap.servers[name] = McpServerEntry(name=name, config=cfg, source="config")
    return snap
