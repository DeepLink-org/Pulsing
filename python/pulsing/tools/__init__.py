# SPDX-License-Identifier: Apache-2.0
"""Deprecated: use ``pulsing.forge`` (Pulsing Forge)."""

from __future__ import annotations

import warnings

warnings.warn(
    "pulsing.tools is renamed to pulsing.forge; update imports to pulsing.forge",
    DeprecationWarning,
    stacklevel=2,
)

from pulsing.forge import *  # noqa: F403
