# SPDX-License-Identifier: Apache-2.0
"""Locate and run the Rust ``pulsing-forge-repl`` binary."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def find_forge_repl_binary() -> Path | None:
    override = os.environ.get("PULSING_FORGE_REPL_BIN")
    if override:
        path = Path(override)
        if path.is_file():
            return path

    for name in ("pulsing-forge-repl", "pforge"):
        found = shutil.which(name)
        if found:
            return Path(found)

    root = _repo_root()
    for rel in ("target/debug/pulsing-forge-repl", "target/release/pulsing-forge-repl"):
        candidate = root / rel
        if candidate.is_file():
            return candidate
    return None


def try_run_rust_repl(argv: list[str]) -> int | None:
    """Run Rust REPL if binary exists. Returns exit code, or None to use Python."""
    if os.environ.get("PULSING_FORGE_REPL_PYTHON", "").lower() in ("1", "true", "yes"):
        return None
    binary = find_forge_repl_binary()
    if binary is None:
        return None
    completed = subprocess.run([str(binary), *argv], check=False)
    return completed.returncode
