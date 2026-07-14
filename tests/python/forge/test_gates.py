# SPDX-License-Identifier: Apache-2.0
"""L0–L1 Forge gates: registry + callable smoke."""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.forge.codex_parity import format_report_text, parity_report
from pulsing.forge.rust_runtime import RUST_FORGE_AVAILABLE
from pulsing.testing.forge_harness import (
    LOCAL_PYTHON_TOOLS,
    assert_forge_manifest,
    run_tool_smoke,
    smoke_failures,
)

pytestmark = pytest.mark.forge


@pytest.mark.forge_l0
def test_l0_forge_manifest() -> None:
    assert_forge_manifest()


@pytest.mark.forge_l1
def test_l1_local_python_tools_callable(local_forge, forge_workspace: Path) -> None:
    failures = smoke_failures(
        run_tool_smoke(local_forge, forge_workspace, tools=LOCAL_PYTHON_TOOLS)
    )
    assert not failures, failures


@pytest.mark.forge_l1
def test_l1_hybrid_all_tools_callable(hybrid_forge, forge_workspace: Path) -> None:
    failures = smoke_failures(run_tool_smoke(hybrid_forge, forge_workspace))
    assert not failures, failures


@pytest.mark.forge_l1
def test_l1_conformance_report(capsys) -> None:
    """Print internal conformance scores for CI logs (informational)."""
    report = parity_report(rust_available=RUST_FORGE_AVAILABLE)
    print(format_report_text(report))
    assert report.gates["registry"].pct >= 96.0
