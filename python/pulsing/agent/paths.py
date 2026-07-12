# SPDX-License-Identifier: Apache-2.0
"""Agent user config paths and environment variables."""

from __future__ import annotations

import os


def agent_env(name: str, default: str = "") -> str:
    """Read ``PULSING_AGENT_{name}``, with fallback to legacy ``PULSING_CRAFT_{name}``."""
    return (
        os.environ.get(f"PULSING_AGENT_{name}")
        or os.environ.get(f"PULSING_CRAFT_{name}")
        or default
    )
