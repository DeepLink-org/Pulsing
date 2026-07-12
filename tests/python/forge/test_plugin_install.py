# SPDX-License-Identifier: Apache-2.0
"""`request_plugin_install` authorization and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pulsing.forge.context import ToolCallContext
from pulsing.forge.discovery.plugin_id import PluginId
from pulsing.forge.discovery.plugin_store import PluginStore
from pulsing.forge.handlers import dispatch_tool
from pulsing.forge.session import LocalToolSession

pytestmark = pytest.mark.forge


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


def _ctx(*, approve: bool) -> ToolCallContext:
    session = LocalToolSession(plugin_install=lambda _args: approve)
    return ToolCallContext(cwd=".", session=session)


def test_plugin_install_confirmed_success(codex_home: Path) -> None:
    plugin_id = _scaffold_marketplace(codex_home)
    ctx = _ctx(approve=True)
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
    assert payload["tool_id"] == plugin_id
    assert payload["tools_registered"] >= 0
    assert PluginStore().is_installed(PluginId.parse(plugin_id))


def test_plugin_install_denied_skips_install(codex_home: Path) -> None:
    plugin_id = _scaffold_marketplace(codex_home)
    ctx = _ctx(approve=False)
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
    assert payload["user_confirmed"] is False
    assert payload["completed"] is True
    assert payload["tools_registered"] == 0
    assert not PluginStore().is_installed(PluginId.parse(plugin_id))


def test_plugin_install_unknown_tool_id(codex_home: Path) -> None:
    _scaffold_marketplace(codex_home)
    ctx = _ctx(approve=True)
    ctx.tool_catalog.refresh_from_codex()

    out = dispatch_tool(
        "request_plugin_install",
        {
            "tool_type": "plugin",
            "action_type": "install",
            "tool_id": "missing@local-dev",
            "suggest_reason": "Need it",
        },
        ctx=ctx,
    )
    assert out.is_error
    assert "unknown plugin" in out.content


def test_plugin_install_rejects_enable_action(codex_home: Path) -> None:
    plugin_id = _scaffold_marketplace(codex_home)
    ctx = _ctx(approve=True)
    ctx.tool_catalog.refresh_from_codex()

    out = dispatch_tool(
        "request_plugin_install",
        {
            "tool_type": "plugin",
            "action_type": "enable",
            "tool_id": plugin_id,
            "suggest_reason": "Need it",
        },
        ctx=ctx,
    )
    assert out.is_error
    assert 'action_type="install"' in out.content


def test_plugin_install_requires_suggest_reason(codex_home: Path) -> None:
    plugin_id = _scaffold_marketplace(codex_home)
    ctx = _ctx(approve=True)
    ctx.tool_catalog.refresh_from_codex()

    out = dispatch_tool(
        "request_plugin_install",
        {
            "tool_type": "plugin",
            "action_type": "install",
            "tool_id": plugin_id,
            "suggest_reason": "   ",
        },
        ctx=ctx,
    )
    assert out.is_error
    assert "non-empty suggest_reason" in out.content


def test_plugin_install_accepts_reason_alias(codex_home: Path) -> None:
    plugin_id = _scaffold_marketplace(codex_home)
    ctx = _ctx(approve=True)
    ctx.tool_catalog.refresh_from_codex()

    out = dispatch_tool(
        "request_plugin_install",
        {
            "plugin_id": plugin_id,
            "reason": "Need demo MCP tools",
        },
        ctx=ctx,
    )
    assert not out.is_error
    payload = json.loads(out.content)
    assert payload["user_confirmed"] is True


def test_list_available_plugins_returns_marketplace_entry(codex_home: Path) -> None:
    plugin_id = _scaffold_marketplace(codex_home)
    ctx = _ctx(approve=True)
    ctx.tool_catalog.refresh_from_codex()

    out = dispatch_tool("list_available_plugins_to_install", {}, ctx=ctx)
    assert not out.is_error
    payload = json.loads(out.content)
    entry = payload["tools"][0]
    assert entry["id"] == plugin_id
    assert entry["tool_type"] == "plugin"
    assert entry["description"] == "Demo plugin"
    assert entry["mcp_server_names"] == ["demo-mcp"]


def test_list_available_plugins_hides_installed(codex_home: Path) -> None:
    plugin_id = _scaffold_marketplace(codex_home)
    ctx = _ctx(approve=True)
    ctx.tool_catalog.refresh_from_codex()
    dispatch_tool(
        "request_plugin_install",
        {
            "tool_type": "plugin",
            "action_type": "install",
            "tool_id": plugin_id,
            "suggest_reason": "Need demo MCP tools",
        },
        ctx=ctx,
    )

    out = dispatch_tool("list_available_plugins_to_install", {}, ctx=ctx)
    assert not out.is_error
    payload = json.loads(out.content)
    assert payload["tools"] == []
