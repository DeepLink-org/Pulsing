"""Contract: ``pulsing._core`` public API must match across Path A and Path B.

Path A: maturin / ``pulsing._core`` PyO3 extension.
Path B: ``pulsing-cli`` RustPython native module (run via subprocess when binary exists).

This test introspects symbols and method names so the two paths cannot drift silently.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Canonical public API (classes + module-level callables + constants).
CORE_CLASSES = frozenset(
    {
        "ActorId",
        "ActorRef",
        "ActorSystem",
        "CacheAwareConfig",
        "CacheAwarePolicy",
        "ConnectActorRef",
        "ConsistentHashPolicy",
        "ForgeRuntime",
        "Message",
        "NodeId",
        "PowerOfTwoPolicy",
        "PulsingConnect",
        "RandomPolicy",
        "RoundRobinPolicy",
        "StreamMessage",
        "StreamReader",
        "StreamWriter",
        "SystemConfig",
        "WorkerInfo",
        "ZeroCopyDescriptor",
    }
)

CORE_FUNCTIONS = frozenset(
    {
        "init_distributed_tracing",
        "shutdown_distributed_tracing",
    }
)

CORE_CONSTANTS = frozenset({"__version__"})

# Methods every binding must expose on these types (subset; grow as Path B catches up).
REQUIRED_METHODS: dict[str, frozenset[str]] = {
    "NodeId": frozenset({"generate", "local", "uuid", "is_local"}),
    "ActorId": frozenset({"generate", "from_str"}),
    "SystemConfig": frozenset(
        {"standalone", "with_addr", "with_seeds", "with_head_node", "with_head_addr"}
    ),
    "ActorSystem": frozenset(
        {
            "create",
            "spawn",
            "shutdown",
            "resolve",
            "refer",
            "members",
            "node_id",
            "addr",
        }
    ),
    "ActorRef": frozenset({"ask", "tell", "is_local"}),
    "Message": frozenset({"from_json", "empty", "to_json"}),
}


def _collect_core_surface():
    core = importlib.import_module("pulsing._core")
    names = {n for n in dir(core) if not n.startswith("_")}
    classes = {n for n in names if isinstance(getattr(core, n), type)}
    funcs = {
        n
        for n in names
        if callable(getattr(core, n)) and not isinstance(getattr(core, n), type)
    }
    constants = names - classes - funcs
    return core, classes, funcs, constants


def _missing_methods(core, class_name: str) -> set[str]:
    cls = getattr(core, class_name)
    exposed = {n for n in dir(cls) if not n.startswith("_")}
    return REQUIRED_METHODS[class_name] - exposed


@pytest.mark.parametrize("path_label", ["path_a"])
def test_core_api_surface_path_a(path_label):
    """Path A (current pytest env) must expose the canonical ``_core`` API."""
    core, classes, funcs, constants = _collect_core_surface()

    assert hasattr(core, "__version__")
    assert CORE_CLASSES <= classes, f"missing classes: {CORE_CLASSES - classes}"
    assert CORE_FUNCTIONS <= funcs, f"missing functions: {CORE_FUNCTIONS - funcs}"

    for cls_name, methods in REQUIRED_METHODS.items():
        missing = _missing_methods(core, cls_name)
        assert not missing, f"{cls_name} missing methods: {sorted(missing)}"


def _pulsing_cli_binary() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    for candidate in (
        root / "target" / "debug" / "pulsing",
        root / "target" / "release" / "pulsing",
    ):
        if candidate.is_file():
            return candidate
    return None


@pytest.mark.skipif(
    _pulsing_cli_binary() is None,
    reason="pulsing-cli binary not built (cargo build -p pulsing-cli)",
)
def test_core_api_surface_path_b():
    """Path B binary must expose the same ``_core`` class names (methods tracked separately)."""
    binary = _pulsing_cli_binary()
    repo = Path(__file__).resolve().parents[2]
    script = """
import json, pulsing._core as c
out = {
    "classes": sorted(n for n in dir(c) if not n.startswith("_") and isinstance(getattr(c, n), type)),
    "funcs": sorted(n for n in dir(c) if not n.startswith("_") and callable(getattr(c, n)) and not isinstance(getattr(c, n), type)),
}
print(json.dumps(out))
"""
    env = os.environ.copy()
    env.setdefault("PULSING_REPO_ROOT", str(repo))
    script_path = repo / "target" / "_core_surface_probe.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script, encoding="utf-8")
    proc = subprocess.run(
        [str(binary), "run", str(script_path)],
        capture_output=True,
        env=env,
        cwd=repo,
        timeout=120,
    )
    if proc.returncode != 0:
        pytest.fail(f"pulsing-cli failed:\n{proc.stderr}\n{proc.stdout}")

    import json

    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    classes = set(payload["classes"])
    funcs = set(payload["funcs"])

    assert CORE_CLASSES <= classes, f"Path B missing classes: {CORE_CLASSES - classes}"
    assert CORE_FUNCTIONS <= funcs, f"Path B missing functions: {CORE_FUNCTIONS - funcs}"
