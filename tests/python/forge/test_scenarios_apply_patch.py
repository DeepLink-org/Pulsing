# SPDX-License-Identifier: Apache-2.0
"""L3 apply_patch scenario fixtures (codex-apply-patch portable layout)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from pulsing.testing.forge_harness import local_runtime

pytestmark = [pytest.mark.forge, pytest.mark.forge_l3]

_SCENARIOS = (
    Path(__file__).resolve().parents[3]
    / "vendor"
    / "codex-rs"
    / "apply-patch"
    / "tests"
    / "fixtures"
    / "scenarios"
)

# Tracked gaps vs codex-apply-patch reference (see testing.zh.md).
KNOWN_GAP_SCENARIOS = frozenset(
    {
        "011_add_overwrites_existing_file",
        "015_failure_after_partial_success_leaves_changes",
    }
)


def _snapshot_dir(root: Path) -> dict[str, bytes | str]:
    out: dict[str, bytes | str] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if path.name == ".gitattributes":
            continue
        rel = str(path.relative_to(root))
        if path.is_dir():
            out[rel + "/"] = "dir"
        elif path.is_file():
            out[rel] = path.read_bytes()
    return out


def _run_scenario(scenario: Path, tmp: Path) -> bool:
    inp = scenario / "input"
    if inp.is_dir():
        for src in inp.rglob("*"):
            if src.is_file():
                dst = tmp / src.relative_to(inp)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    patch = (scenario / "patch.txt").read_text(encoding="utf-8")
    rt = local_runtime(tmp)
    rt.call_tool("apply_patch", {"patch": patch})
    return _snapshot_dir(scenario / "expected") == _snapshot_dir(tmp)


@pytest.fixture(scope="module")
def scenario_dirs() -> list[Path]:
    if not _SCENARIOS.is_dir():
        pytest.skip(f"missing scenarios dir: {_SCENARIOS}")
    return sorted(p for p in _SCENARIOS.iterdir() if p.is_dir())


def test_l3_apply_patch_scenarios_coverage(scenario_dirs: list[Path]) -> None:
    assert len(scenario_dirs) >= 20


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(p, id=p.name)
        for p in (sorted(_SCENARIOS.iterdir()) if _SCENARIOS.is_dir() else [])
        if p.is_dir() and p.name not in KNOWN_GAP_SCENARIOS
    ],
)
def test_l3_apply_patch_scenario(scenario: Path) -> None:
    if not _SCENARIOS.is_dir():
        pytest.skip(f"missing scenarios dir: {_SCENARIOS}")
    tmp = Path(tempfile.mkdtemp())
    try:
        assert _run_scenario(scenario, tmp), scenario.name
    finally:
        shutil.rmtree(tmp)


@pytest.mark.parametrize("name", sorted(KNOWN_GAP_SCENARIOS))
def test_l3_apply_patch_known_gaps(name: str) -> None:
    """Documented gaps — xfail until patch engine catches up."""
    scenario = _SCENARIOS / name
    if not scenario.is_dir():
        pytest.skip(f"missing gap scenario {name}")
    tmp = Path(tempfile.mkdtemp())
    try:
        ok = _run_scenario(scenario, tmp)
    finally:
        shutil.rmtree(tmp)
    if ok:
        pytest.fail(f"scenario {name} now passes — remove from KNOWN_GAP_SCENARIOS")
