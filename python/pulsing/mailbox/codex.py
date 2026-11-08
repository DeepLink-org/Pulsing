"""Generate a self-contained Codex plugin without changing user configuration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .actors import ROLE_KINDS, validate_identity
from .managed import default_state_dir, seed_configuration

PLUGIN_NAME = "pulsing"
MARKETPLACE_NAME = "pulsing-local"


def generate_plugin(
    output: str | Path,
    identity: str,
    seeds: list[str] | None = None,
    addr: str | None = None,
    python: str = sys.executable,
    *,
    state_dir: str | Path | None = None,
    kind: str = "worker",
) -> Path:
    """Write a new marketplace, refusing to overwrite any existing directory.

    Explicit settings are saved without capturing the generator's environment.
    Omitted network settings may use SEED/PULSING_ADDR at MCP startup. The plugin
    can be cached without repository-relative imports, symlinks, shell expansion,
    or a dependency on the generator's cwd.
    """
    validate_identity(identity)
    seeds = list(seeds or [])
    seed_configuration(seeds)
    if kind not in ROLE_KINDS:
        raise ValueError(f"kind must be one of {ROLE_KINDS}")
    if addr is not None and not addr.strip():
        raise ValueError("addr must not be empty")
    executable = Path(python).expanduser()
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError("python must be an absolute path to a Python executable")

    source = Path(__file__).parent
    skill = (
        (source / "SKILL.md").read_text(encoding="utf-8")
        + "\n"
        + (source / "CODEX.md").read_text(encoding="utf-8")
    )
    root = Path(output).expanduser().absolute()
    # mkdir is also the exclusive ownership check; never merge user files.
    root.mkdir(parents=True, exist_ok=False)
    plugin = root / "plugins" / PLUGIN_NAME
    args = [
        "-m",
        "pulsing.mailbox",
        "mcp",
        "--managed",
        "--identity",
        identity,
    ]
    if addr is not None:
        args.extend(["--addr", addr])
    args.extend(
        [
            "--state-dir",
            str(Path(state_dir or default_state_dir()).expanduser().absolute()),
            "--kind",
            kind,
        ]
    )
    for seed in seeds:
        args.extend(["--seed", seed])
    files = {
        plugin / ".codex-plugin" / "plugin.json": {
            "name": PLUGIN_NAME,
            "version": "0.1.0",
            "description": "Coordinate Codex agents across trusted machines with Pulsing mailboxes.",
            "author": {"name": "Pulsing contributors"},
            "license": "Apache-2.0",
            "skills": "./skills/",
            "mcpServers": "./.mcp.json",
        },
        plugin / ".mcp.json": {
            "mcpServers": {
                "pulsing-mailbox": {"command": str(executable), "args": args}
            }
        },
        root / ".agents" / "plugins" / "marketplace.json": {
            "name": MARKETPLACE_NAME,
            "interface": {"displayName": "Pulsing local agents"},
            "plugins": [
                {
                    "name": PLUGIN_NAME,
                    "source": {"source": "local", "path": f"./plugins/{PLUGIN_NAME}"},
                    "policy": {
                        "installation": "AVAILABLE",
                        "authentication": "ON_INSTALL",
                    },
                    "category": "Productivity",
                }
            ],
        },
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    skill_path = plugin / "skills" / "pulsing-mailbox" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(skill, encoding="utf-8")
    return root
