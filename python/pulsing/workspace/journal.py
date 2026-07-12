# SPDX-License-Identifier: Apache-2.0
"""Workspace journal: checkpoint and rollback."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pulsing.workspace.ignore import should_skip
from pulsing.workspace.layout import WorkspaceLayout


@dataclass
class RevisionInfo:
    id: str
    created_at: str
    message: str
    author: str
    file_count: int


def _scan_files(layout: WorkspaceLayout) -> list[Path]:
    paths: list[Path] = []
    for path in layout.root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(layout.root)
        if should_skip(rel):
            continue
        paths.append(rel)
    paths.sort()
    return paths


def current_head(layout: WorkspaceLayout) -> str | None:
    if not layout.head_file.is_file():
        return None
    value = layout.head_file.read_text(encoding="utf-8").strip()
    return value or None


def list_revisions(layout: WorkspaceLayout) -> list[RevisionInfo]:
    out: list[RevisionInfo] = []
    if not layout.revisions_dir.is_dir():
        return out
    for entry in sorted(layout.revisions_dir.iterdir()):
        if not entry.is_dir():
            continue
        manifest_path = entry / "manifest.json"
        if not manifest_path.is_file():
            continue
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        out.append(
            RevisionInfo(
                id=str(data["id"]),
                created_at=str(data["created_at"]),
                message=str(data.get("message", "")),
                author=str(data.get("author", "")),
                file_count=len(data.get("files") or []),
            )
        )
    return out


def _next_revision_id(layout: WorkspaceLayout) -> str:
    revs = list_revisions(layout)
    if not revs:
        return "0001"
    last = int(revs[-1].id)
    return f"{last + 1:04}"


def checkpoint(
    layout: WorkspaceLayout,
    *,
    message: str | None = None,
    author: str | None = None,
) -> dict[str, Any]:
    from pulsing.workspace.hooks import run_before_checkpoint, run_after_checkpoint

    ctx = {
        "root": str(layout.root),
        "message": message or "checkpoint",
    }
    extra = run_before_checkpoint(ctx)

    rev_id = _next_revision_id(layout)
    rev_path = layout.revision_dir(rev_id)
    files_dir = rev_path / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    scanned = _scan_files(layout)
    if extra:
        for rel_str in extra:
            rel = Path(rel_str)
            p = layout.root / rel
            if p.is_file() and rel not in scanned:
                scanned.append(rel)
        scanned.sort()

    records: list[dict[str, Any]] = []
    for rel in scanned:
        src = layout.root / rel
        dest = files_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        data = src.read_bytes()
        records.append(
            {
                "path": rel.as_posix(),
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
        )

    manifest: dict[str, Any] = {
        "id": rev_id,
        "parent": current_head(layout),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": message or "checkpoint",
        "author": author or "user",
        "files": records,
    }
    (rev_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    layout.head_file.parent.mkdir(parents=True, exist_ok=True)
    layout.head_file.write_text(f"{rev_id}\n", encoding="utf-8")
    run_after_checkpoint({**ctx, "revision_id": rev_id})
    return manifest


def rollback(
    layout: WorkspaceLayout, *, revision_id: str | None = None
) -> dict[str, Any]:
    rev_id = revision_id or current_head(layout)
    if not rev_id:
        raise SystemExit("no checkpoint to roll back to")
    rev_path = layout.revision_dir(rev_id)
    manifest_path = rev_path / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"revision {rev_id} not found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files_dir = rev_path / "files"
    for file in manifest.get("files") or []:
        rel = Path(file["path"])
        if should_skip(rel):
            continue
        src = files_dir / rel
        dest = layout.root / rel
        if not src.is_file():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    layout.head_file.write_text(f"{rev_id}\n", encoding="utf-8")
    return manifest
