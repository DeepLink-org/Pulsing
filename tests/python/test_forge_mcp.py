# SPDX-License-Identifier: Apache-2.0
"""Tests for Forge MCP catalog loading."""

from __future__ import annotations

import json
from pathlib import Path

from pulsing.forge.discovery.plugin_id import PluginId
from pulsing.forge.discovery.plugin_store import PluginStore
from pulsing.forge.mcp.catalog import load_mcp_catalog, parse_plugin_mcp_file


def test_parse_plugin_mcp_file_shapes(tmp_path: Path) -> None:
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(
        json.dumps({"mcpServers": {"demo": {"command": "echo", "args": ["hi"]}}}),
        encoding="utf-8",
    )
    out = parse_plugin_mcp_file(tmp_path, wrapped)
    assert "demo" in out
    assert out["demo"]["command"] == "echo"

    flat = tmp_path / "flat.json"
    flat.write_text(
        json.dumps({"github": {"command": "npx", "args": ["pkg"]}}), encoding="utf-8"
    )
    out2 = parse_plugin_mcp_file(tmp_path, flat)
    assert "github" in out2


def test_load_mcp_catalog_empty(monkeypatch) -> None:
    monkeypatch.setenv("CODEX_HOME", "/tmp/nonexistent-codex-home-forge-test")
    snap = load_mcp_catalog()
    assert isinstance(snap.servers, dict)


def test_load_mcp_catalog_uses_active_plugin_version(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    plugin_id = PluginId(plugin_name="demo", marketplace_name="local-dev")
    root = PluginStore().plugin_version_root(plugin_id, "1.2.3")
    manifest_dir = root / ".codex-plugin"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "plugin.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "version": "1.2.3",
                "mcpServers": ".mcp.json",
            }
        ),
        encoding="utf-8",
    )
    (root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"demo-server": {"command": "echo"}}}),
        encoding="utf-8",
    )

    snapshot = load_mcp_catalog()

    assert snapshot.servers["demo-server"].plugin_id == plugin_id.id
