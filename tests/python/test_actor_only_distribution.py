"""Guard the actor distribution against removed application dependencies."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "pulsing.serving.router",
        "pulsing.forge.runtime",
        "pulsing.agent.llm",
        "pulsing.cli.bench",
    ],
)
def test_removed_application_modules(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_native_module_has_no_application_exports():
    from pulsing import _core

    for name in ("ForgeRuntime", "CacheAwarePolicy", "CacheAwareConfig"):
        assert not hasattr(_core, name)


def test_actor_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "pulsing.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert "actor" in result.stdout
    assert "inspect" in result.stdout
    assert "checkpoint" in result.stdout
    for name in ("forge", "bench", "agent", "gui"):
        assert name not in result.stdout.lower()


def test_workspace_defaults_do_not_require_an_agent(tmp_path: Path):
    from pulsing.workspace import init_workspace

    result = init_workspace(tmp_path)
    assert result.created
    manifest = json.loads((tmp_path / ".pulsing/workspace.json").read_text())
    cluster = json.loads((tmp_path / ".pulsing/cluster.json").read_text())
    assert manifest["template"] == "minimal"
    assert set(cluster) == {"cluster_id", "name"}
    assert (tmp_path / ".pulsing/history/HEAD").is_file()


def test_workspace_rejects_removed_agent_template(tmp_path: Path):
    from pulsing.workspace import init_workspace

    with pytest.raises(ValueError, match="expected minimal"):
        init_workspace(tmp_path, template="agent")
    assert not (tmp_path / ".pulsing").exists()
