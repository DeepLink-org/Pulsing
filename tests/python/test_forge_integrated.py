# SPDX-License-Identifier: Apache-2.0
"""Forge integrated runtime tests."""

from __future__ import annotations

from pulsing.forge.tool_coverage import assert_forge_tool_coverage
from pulsing.forge.integrated import FORGE_HOST_TOOL_NAMES, FORGE_ISOLATED_TOOL_NAMES


def test_forge_tool_coverage() -> None:
    assert_forge_tool_coverage()


def test_forge_tool_partitions() -> None:
    assert not (FORGE_ISOLATED_TOOL_NAMES & FORGE_HOST_TOOL_NAMES)
