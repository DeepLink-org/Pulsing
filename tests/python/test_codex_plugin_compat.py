# SPDX-License-Identifier: Apache-2.0
"""Codex plugin marketplace + cache install tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pulsing.forge.discovery.catalog import ToolCatalog
from pulsing.forge.discovery.codex_paths import plugins_cache_root
from pulsing.forge.discovery.discoverable import (
    DiscoverableToolAction,
    DiscoverableToolType,
)
from pulsing.forge.discovery.install import install_plugin_from_marketplace
from pulsing.forge.discovery.plugin_id import PluginId
from pulsing.forge.discovery.plugin_store import PluginStore
from pulsing.forge.handlers import dispatch_tool
from pulsing.forge.context import ToolCallContext
from pulsing.forge.session import LocalToolSession


@pytest.fixture
def codex_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "codex"
    home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(home))
    monkeypatch.setenv("FORGE_PLUGIN_DISCOVER_ALL", "1")
    return home


def _scaffold_marketplace(codex_home: Path) -> str:
    agents = codex_home / ".agents" / "plugins"
    agents.mkdir(parents=True)
    plugin_src = agents / "demo"
    manifest_dir = plugin_src / ".codex-plugin"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "plugin.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "version": "1.0.0",
                "description": "Demo plugin",
                "mcpServers": ".mcp.json",
            }
        ),
        encoding="utf-8",
    )
    (plugin_src / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"demo-mcp": {"command": "echo"}}}),
        encoding="utf-8",
    )

    marketplace_path = agents / "marketplace.json"
    marketplace_path.write_text(
        json.dumps(
            {
                "name": "local-dev",
                "plugins": [
                    {
                        "name": "demo",
                        "source": {"source": "local", "path": "./demo"},
                        "policy": {"installation": "AVAILABLE"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return "demo@local-dev"


def test_marketplace_discover_and_install(codex_home: Path) -> None:
    plugin_id = _scaffold_marketplace(codex_home)
    catalog = ToolCatalog()
    catalog.refresh_from_codex()
    assert any(t.id == plugin_id for t in catalog.discoverable)

    result = install_plugin_from_marketplace(plugin_id)
    assert result.installed_path.is_dir()
    assert PluginStore().is_installed(result.plugin_id)

    cache_root = plugins_cache_root()
    assert (
        cache_root / "local-dev" / "demo" / "1.0.0" / ".codex-plugin" / "plugin.json"
    ).is_file()


def test_plugin_store_sorts_mixed_numeric_and_text_versions(
    codex_home: Path,
) -> None:
    plugin_id = PluginId(plugin_name="demo", marketplace_name="local-dev")
    base = PluginStore().plugin_base_root(plugin_id)
    for version in ["1.0.9", "1.0.dev", "1.0.10"]:
        (base / version).mkdir(parents=True)

    assert PluginStore().active_plugin_version(plugin_id) == "1.0.dev"


def test_request_plugin_install_codex_wire(codex_home: Path) -> None:
    plugin_id = _scaffold_marketplace(codex_home)
    session = LocalToolSession(
        plugin_install=lambda args: True,
    )
    ctx = ToolCallContext(cwd=".", session=session)
    ctx.tool_catalog.refresh_from_codex()

    out = dispatch_tool(
        "request_plugin_install",
        {
            "tool_type": "plugin",
            "action_type": "install",
            "tool_id": plugin_id,
            "suggest_reason": "Need demo MCP tools",
        },
        ctx=ctx,
    )
    assert not out.is_error
    payload = json.loads(out.content)
    assert payload["completed"] is True
    assert payload["user_confirmed"] is True
    assert payload["tool_type"] == "plugin"
    assert payload["tool_id"] == plugin_id


def test_list_available_plugins_codex_wire(codex_home: Path) -> None:
    plugin_id = _scaffold_marketplace(codex_home)
    ctx = ToolCallContext(cwd=".", session=LocalToolSession())
    out = dispatch_tool("list_available_plugins_to_install", {}, ctx=ctx)
    payload = json.loads(out.content)
    entry = payload["tools"][0]
    assert entry["tool_type"] == "plugin"
    assert entry["id"] == plugin_id
    assert set(entry) == {
        "id",
        "name",
        "description",
        "tool_type",
        "has_skills",
        "mcp_server_names",
        "app_connector_ids",
    }


def test_list_available_plugins_refresh_failure_returns_tool_error(
    codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken plugin cache must surface as a ToolResult error, not an uncaught exception."""

    ctx = ToolCallContext(
        cwd=".", session=LocalToolSession()
    )  # initial refresh succeeds

    def _boom(self: PluginStore) -> list[PluginId]:
        raise OSError("permission denied")

    monkeypatch.setattr(PluginStore, "list_installed_plugin_ids", _boom)
    out = dispatch_tool("list_available_plugins_to_install", {}, ctx=ctx)
    assert out.is_error
    assert "permission denied" in out.content


def test_list_available_plugins_skips_corrupt_installed_plugin(
    codex_home: Path,
) -> None:
    """One plugin with a broken manifest must not block listing the rest of the catalog."""
    plugin_id = _scaffold_marketplace(codex_home)
    install_plugin_from_marketplace(plugin_id)
    manifest_path = (
        plugins_cache_root()
        / "local-dev"
        / "demo"
        / "1.0.0"
        / ".codex-plugin"
        / "plugin.json"
    )
    manifest_path.write_text("{not valid json", encoding="utf-8")

    ctx = ToolCallContext(cwd=".", session=LocalToolSession())
    out = dispatch_tool("list_available_plugins_to_install", {}, ctx=ctx)
    assert not out.is_error
    payload = json.loads(out.content)
    assert payload["tools"] == []
