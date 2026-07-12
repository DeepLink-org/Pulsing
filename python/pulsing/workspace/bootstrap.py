# SPDX-License-Identifier: Apache-2.0
"""Bootstrap a Pulsing workspace (``.pulsing/`` + hooks + initial checkpoint)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pulsing.workspace.hooks import run_on_init
from pulsing.workspace.journal import checkpoint
from pulsing.workspace.layout import WorkspaceLayout, workspace_cluster_id

Template = Literal["minimal", "agent"]

_ON_INIT_STUB = '''"""Pulsing workspace hook — called after `pulsing init`."""

from __future__ import annotations

from typing import Any


def on_init(ctx: dict[str, Any]) -> None:
    """Customize a new workspace. *ctx* has keys: root, cluster_id, template."""
    _ = ctx
'''

_ON_CHECKPOINT_STUB = '''"""Pulsing workspace hook — called before each checkpoint."""

from __future__ import annotations

from typing import Any


def before_checkpoint(ctx: dict[str, Any]) -> list[str] | None:
    """Return extra relative paths to include, or None for default scan."""
    _ = ctx
    return None


def after_checkpoint(ctx: dict[str, Any]) -> None:
    _ = ctx
'''

_SCRIPTS_README = """# Pulsing workspace scripts

Place custom Python workflows here and run:

```bash
pulsing run scripts/your_flow.py
```
"""


@dataclass
class InitResult:
    root: Path
    created: bool
    cluster_id: str


def _minimal_cluster(cluster_id: str, name: str) -> dict:
    return {
        "cluster_id": cluster_id,
        "name": name,
        "provider": "anthropic",
        "model": None,
        "auto_approve": False,
        "sandbox": "off",
        "default_agents": [],
        "shared_tool_worker": False,
        "puzzles": {},
    }


def _agent_cluster(cluster_id: str, name: str) -> dict:
    return {
        "cluster_id": cluster_id,
        "name": name,
        "provider": "anthropic",
        "model": None,
        "auto_approve": False,
        "sandbox": "off",
        "default_agents": ["guide"],
        "shared_tool_worker": False,
        "puzzles": {
            "unit-tests": {
                "title": "Unit test suite",
                "kind": "test",
                "path": "tests",
                "blurb": "Run pytest; keep green.",
                "status": "open",
                "assign_to": "",
            },
        },
    }


def _write_hook_stubs(layout: WorkspaceLayout) -> None:
    on_init = layout.hooks_dir / "on_init.py"
    if not on_init.is_file():
        layout.hooks_dir.mkdir(parents=True, exist_ok=True)
        on_init.write_text(_ON_INIT_STUB, encoding="utf-8")
    on_checkpoint = layout.hooks_dir / "on_checkpoint.py"
    if not on_checkpoint.is_file():
        layout.hooks_dir.mkdir(parents=True, exist_ok=True)
        on_checkpoint.write_text(_ON_CHECKPOINT_STUB, encoding="utf-8")


def _write_scripts_readme(layout: WorkspaceLayout) -> None:
    readme = layout.scripts_dir / "README.md"
    if not readme.is_file():
        layout.scripts_dir.mkdir(parents=True, exist_ok=True)
        readme.write_text(_SCRIPTS_README, encoding="utf-8")


def init_workspace(
    root: Path | None = None,
    *,
    template: Template = "agent",
    name: str | None = None,
    force: bool = False,
    seed_npcs: bool = True,
) -> InitResult:
    """Create ``.pulsing/`` layout, hooks, and an initial checkpoint."""
    root = (root or Path.cwd()).resolve()
    layout = WorkspaceLayout(root)
    cluster_id = workspace_cluster_id(root)

    if layout.is_initialized() and not force:
        return InitResult(root=root, created=False, cluster_id=cluster_id)

    display_name = name or (root.name or "workspace")
    now = datetime.now(timezone.utc).isoformat()

    layout.pulsing_dir.mkdir(parents=True, exist_ok=True)
    layout.hooks_dir.mkdir(parents=True, exist_ok=True)
    layout.scripts_dir.mkdir(parents=True, exist_ok=True)
    layout.revisions_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": 1,
        "template": template,
        "name": display_name,
        "cluster_id": cluster_id,
        "created_at": now,
    }
    layout.workspace_file.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    cluster = _agent_cluster(cluster_id, display_name)
    if template == "minimal":
        cluster = _minimal_cluster(cluster_id, display_name)
    layout.cluster_file.write_text(
        json.dumps(cluster, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _write_hook_stubs(layout)
    _write_scripts_readme(layout)

    if seed_npcs and template == "agent":
        from pulsing.agent.npc.loader import seed_npc_defs

        seed_npc_defs(root)

    checkpoint(layout, message="workspace init", author="pulsing")
    run_on_init({"root": str(root), "cluster_id": cluster_id, "template": template})

    return InitResult(root=root, created=True, cluster_id=cluster_id)
