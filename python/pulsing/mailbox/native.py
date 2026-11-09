"""Native Pi extensions and DSH bundles, sharing the existing companion MCP."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .actors import validate_identity
from .managed import seed_configuration

PLUGIN_DIR = Path(__file__).with_name("plugins")


def generate_plugin(
    output: str | Path,
    harness: str,
    *,
    python: str = sys.executable,
    seeds: list[str] | None = None,
    addr: str | None = None,
    runtime_dir: str | Path | None = None,
    instance: str | None = None,
    dispatch: bool = False,
    dispatch_max_turns: int = 8,
) -> Path:
    """Create an installable package; never overwrite files or capture secrets."""
    if harness not in ("pi", "dsh"):
        raise ValueError("native harness must be pi or dsh")
    if dispatch and harness != "pi":
        raise ValueError("automatic dispatch currently requires the Pi plugin")
    if not 1 <= dispatch_max_turns <= 64:
        raise ValueError("dispatch_max_turns must be 1–64")
    executable = Path(python).expanduser()
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError("python must be an absolute path to a Python executable")
    seeds = list(seeds or [])
    seed_configuration(seeds)
    args = []
    if addr is not None:
        if not addr.strip():
            raise ValueError("addr must not be empty")
        args.extend(["--addr", addr])
    if instance is not None:
        validate_identity(instance)
        if len(instance) > 24 or "." in instance:
            raise ValueError("instance must be at most 24 letters, digits, _ or -")
        args.extend(["--instance", instance])
    if runtime_dir is not None:
        args.extend(["--runtime-dir", str(Path(runtime_dir).expanduser().absolute())])
    for seed in seeds:
        args.extend(["--seed", seed])
    root = Path(output).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "name": "pulsing",
        "version": "0.1.0",
        "type": "module",
        "description": "Pulsing actor mailboxes for trusted multi-agent collaboration",
        "license": "Apache-2.0",
        "files": ["*.mjs", "*.json", "*.md", "cordis.patch.yml"],
        "engines": {"node": ">=22.19.0"},
    }
    if harness == "pi":
        manifest["pi"] = {"extensions": ["./pi.mjs"]}
    else:
        manifest["engines"]["node"] = ">=24.9.0"
        manifest.update(
            {
                "main": "./dsh.mjs",
                "exports": {".": "./dsh.mjs", "./package.json": "./package.json"},
                "dsh": {
                    "bundle": {"patch": "./cordis.patch.yml"},
                    "manifestVersion": 1,
                },
            }
        )
        manifest["engines"]["dsh"] = ">=0.2.2-rc.1 <0.3.0"
        (root / "cordis.patch.yml").write_text(
            "- insert:\n    - id: pulsing\n      name: pulsing\n", encoding="utf-8"
        )
    for name in ("bridge.mjs", "host.mjs", f"{harness}.mjs"):
        (root / name).write_bytes((PLUGIN_DIR / name).read_bytes())
    if harness == "pi":
        (root / "dispatch.mjs").write_bytes((PLUGIN_DIR / "dispatch.mjs").read_bytes())
    (root / "PROTOCOL.md").write_bytes(
        Path(__file__).with_name("COMPANION.md").read_bytes()
    )
    (root / "runtime.json").write_text(
        json.dumps(
            {
                "python": str(executable),
                "launchArgs": args,
                "dispatch": dispatch,
                "dispatchMaxTurns": dispatch_max_turns,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "package.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return root


def pi_command(command: list[str]) -> list[str]:
    if len(command) > 1 and command[1] in (
        "install",
        "remove",
        "uninstall",
        "update",
        "list",
        "config",
        "auth",
        "mcp",
    ):
        return command
    return [command[0], "--extension", str(PLUGIN_DIR / "pi.mjs"), *command[1:]]


def dsh_command(command: list[str], runtime_dir: Path) -> list[str]:
    """An invocation-only overlay; no edits to the user's DSH profile."""
    patch = runtime_dir / "pulsing-dsh.patch.json"
    patch.write_text(
        json.dumps(
            [
                {
                    "insert": [
                        {
                            "id": "pulsing-companion",
                            "name": (PLUGIN_DIR / "dsh.mjs").as_uri(),
                        }
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    # DSH only recognizes shorthand "dsh web" when the profile is the FIRST
    # argument. Normalize it before adding flags, keeping inner app args intact.
    arguments = command[1:]
    if arguments and arguments[0] == "plugin":
        return command
    if arguments and not arguments[0].startswith("-"):
        arguments = ["--profile", arguments[0], *arguments[1:]]
    return [command[0], "--patch", str(patch), *arguments]
