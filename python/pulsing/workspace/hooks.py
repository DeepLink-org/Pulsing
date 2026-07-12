# SPDX-License-Identifier: Apache-2.0
"""Load and run workspace hook scripts from ``.pulsing/hooks/``."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from pulsing.workspace.layout import WorkspaceLayout


def _load_hook(layout: WorkspaceLayout, filename: str):
    path = layout.hooks_dir / filename
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(f"pulsing_hooks_{filename}", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_on_init(ctx: dict[str, Any]) -> None:
    root = Path(ctx["root"])
    mod = _load_hook(WorkspaceLayout(root), "on_init.py")
    if mod is None:
        return
    fn = getattr(mod, "on_init", None)
    if callable(fn):
        fn(ctx)


def run_before_checkpoint(ctx: dict[str, Any]) -> list[str] | None:
    root = Path(ctx["root"])
    mod = _load_hook(WorkspaceLayout(root), "on_checkpoint.py")
    if mod is None:
        return None
    fn = getattr(mod, "before_checkpoint", None)
    if not callable(fn):
        return None
    extra = fn(ctx)
    if extra is None:
        return None
    return [str(p) for p in extra]


def run_after_checkpoint(ctx: dict[str, Any]) -> None:
    root = Path(ctx["root"])
    mod = _load_hook(WorkspaceLayout(root), "on_checkpoint.py")
    if mod is None:
        return
    fn = getattr(mod, "after_checkpoint", None)
    if callable(fn):
        fn(ctx)
