# SPDX-License-Identifier: Apache-2.0
"""Top-level workspace commands: init, history, checkpoint, rollback."""

from __future__ import annotations

import argparse
from pathlib import Path

from pulsing.workspace.bootstrap import init_workspace
from pulsing.workspace.journal import checkpoint, current_head, list_revisions, rollback
from pulsing.workspace.layout import WorkspaceLayout, require_workspace_root


def _add_init(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("init", help="Bootstrap a Pulsing workspace")
    p.add_argument("dir", nargs="?", type=Path, default=None, help="target directory")
    p.add_argument(
        "--template",
        choices=("minimal", "agent"),
        default="agent",
        help="workspace template (default: agent)",
    )
    p.add_argument("--name", default=None, help="display name in cluster.json")
    p.add_argument("--force", action="store_true", help="re-initialize existing workspace")
    p.set_defaults(func=cmd_init)


def _add_history(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("history", help="List workspace checkpoints")
    p.set_defaults(func=cmd_history)


def _add_checkpoint(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("checkpoint", help="Save a workspace checkpoint")
    p.add_argument("-m", "--message", default=None, help="checkpoint message")
    p.set_defaults(func=cmd_checkpoint)


def _add_rollback(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("rollback", help="Restore files from a checkpoint")
    p.add_argument("revision", nargs="?", default=None, help="revision id (default: HEAD)")
    p.set_defaults(func=cmd_rollback)


def register_workspace_commands(sub: argparse._SubParsersAction) -> None:
    _add_init(sub)
    _add_history(sub)
    _add_checkpoint(sub)
    _add_rollback(sub)


def cmd_init(args: argparse.Namespace) -> None:
    result = init_workspace(
        args.dir,
        template=args.template,
        name=args.name,
        force=args.force,
    )
    if result.created:
        print(f"initialized {result.root}  (cluster_id={result.cluster_id})")
        if args.template == "agent":
            print("  pulsing agent wake   # start agents")
        print("  pulsing history      # list checkpoints")
        print("  pulsing checkpoint   # save workspace snapshot")
    else:
        print(f"already initialized: {result.root}")


def cmd_history(_args: argparse.Namespace) -> None:
    root = require_workspace_root()
    layout = WorkspaceLayout(root)
    head = current_head(layout)
    revs = list_revisions(layout)
    if not revs:
        print("no checkpoints yet — run `pulsing checkpoint`")
        return
    for rev in revs:
        mark = "*" if head == rev.id else " "
        print(f"{mark} {rev.id}  {rev.created_at}  {rev.file_count} files  {rev.message}")


def cmd_checkpoint(args: argparse.Namespace) -> None:
    root = require_workspace_root()
    layout = WorkspaceLayout(root)
    manifest = checkpoint(layout, message=args.message)
    print(
        f"checkpoint {manifest['id']}  ({len(manifest['files'])} files) — {manifest['message']}",
    )


def cmd_rollback(args: argparse.Namespace) -> None:
    root = require_workspace_root()
    layout = WorkspaceLayout(root)
    manifest = rollback(layout, revision_id=args.revision)
    print(f"rolled back to {manifest['id']} — {manifest['message']}")
