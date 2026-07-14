# SPDX-License-Identifier: Apache-2.0
"""Tests for Forge MCP catalog loading."""

from __future__ import annotations

import json
from pathlib import Path

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
