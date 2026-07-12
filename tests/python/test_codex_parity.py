# SPDX-License-Identifier: Apache-2.0
"""Internal conformance gates (legacy entry — see tests/python/forge/)."""

from __future__ import annotations

import pytest

from pulsing.forge.codex_parity import (
    assert_registry_gate,
    format_report_text,
    parity_report,
)
from pulsing.forge.rust_runtime import RUST_FORGE_AVAILABLE

pytestmark = pytest.mark.forge


def test_ccrp_registry_gate() -> None:
    """Every in-scope Codex tool must be registered in Forge."""
    assert_registry_gate()


def test_ccrp_parity_report_snapshot() -> None:
    """Print parity summary; callable gate reflects maturin build."""
    report = parity_report(rust_available=RUST_FORGE_AVAILABLE)
    text = format_report_text(report)
    # Keep visible in pytest -v / CI logs for release notes.
    print(text)
    assert report.gates["registry"].pct == 100.0
    if RUST_FORGE_AVAILABLE:
        assert report.gates["callable"].pct == 100.0, format_report_text(report)
        assert report.certification in ("Bronze+", "Silver", "Gold", "Platinum")
    assert report.certification in (
        "Incomplete",
        "Bronze",
        "Bronze+",
        "Silver",
        "Gold",
        "Platinum",
    )


@pytest.mark.xfail(reason="L3 behavior parity not yet at Gold threshold")
def test_ccrp_behavior_gate_gold() -> None:
    report = parity_report(rust_available=RUST_FORGE_AVAILABLE)
    assert report.gates["behavior"].pct >= 90.0
