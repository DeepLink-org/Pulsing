"""Codex package generation is offline and never edits Codex user settings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pulsing.mailbox.codex import generate_plugin


def test_generated_package_is_self_contained(tmp_path):
    root = generate_plugin(
        tmp_path / "marketplace with spaces",
        "worker-a",
        ["10.0.0.1:8100", "10.0.0.2:8100"],
        "10.0.0.3:0",
        state_dir=tmp_path / "state",
    )
    catalog = json.loads((root / ".agents/plugins/marketplace.json").read_text())
    assert catalog["plugins"][0]["name"] == "pulsing"
    assert catalog["plugins"][0]["source"]["path"] == "./plugins/pulsing"
    plugin = root / catalog["plugins"][0]["source"]["path"]
    manifest = json.loads((plugin / ".codex-plugin/plugin.json").read_text())
    assert manifest["name"] == "pulsing"
    config = json.loads((plugin / manifest["mcpServers"]).read_text())
    server = config["mcpServers"]["pulsing-mailbox"]
    assert server["command"] == sys.executable
    assert server["args"] == [
        "-m",
        "pulsing.mailbox",
        "mcp",
        "--managed",
        "--identity",
        "worker-a",
        "--addr",
        "10.0.0.3:0",
        "--state-dir",
        str(tmp_path / "state"),
        "--kind",
        "worker",
        "--seed",
        "10.0.0.1:8100",
        "--seed",
        "10.0.0.2:8100",
    ]
    protocol = (
        Path(__file__).parents[3] / "python/pulsing/mailbox/SKILL.md"
    ).read_text()
    skill = (plugin / manifest["skills"] / "pulsing-mailbox/SKILL.md").read_text()
    assert skill.startswith(protocol)
    assert "## Codex workflow" in skill
    assert "no background wakeup" in skill
    assert "base revision" in skill
    assert not any(path.is_symlink() for path in root.rglob("*"))


def test_generation_refuses_existing_output(tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    sentinel = root / "user-data"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        generate_plugin(root, "router", ["127.0.0.1:8100"])
    assert list(root.iterdir()) == [sentinel]
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"identity": "../worker"},
        {"seeds": [""]},
        {"seeds": ["1", "127.0.0.1:8100"]},
        {"kind": "invalid"},
        {"addr": " "},
        {"python": "python3"},
        {"python": "/nonexistent/pulsing/python"},
    ],
)
def test_invalid_configuration_writes_nothing(tmp_path, kwargs):
    root = tmp_path / "not-created"
    options = {"identity": "worker", "seeds": ["127.0.0.1:8100"]} | kwargs
    with pytest.raises(ValueError):
        generate_plugin(root, **options)
    assert not root.exists()


def test_generation_cli(tmp_path):
    root = tmp_path / "cli"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pulsing.mailbox",
            "codex-plugin",
            "--output",
            str(root),
            "--identity",
            "router",
            "--seed",
            "127.0.0.1:8100",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Identity: router" in result.stdout
    assert (root / ".agents/plugins/marketplace.json").is_file()


def test_generated_plugin_bootstraps_without_seed(tmp_path):
    root = generate_plugin(tmp_path / "plugin", "worker", state_dir=tmp_path / "state")
    server = json.loads((root / "plugins/pulsing/.mcp.json").read_text())["mcpServers"][
        "pulsing-mailbox"
    ]
    assert "--managed" in server["args"]
    assert "--addr" not in server["args"]  # Allow runtime PULSING_ADDR.
    assert "--seed" not in server["args"]
    assert server["args"][server["args"].index("--state-dir") + 1] == str(
        tmp_path / "state"
    )
