# SPDX-License-Identifier: Apache-2.0
"""Top-level help for ``pulsing`` CLI."""


def print_top_level_help() -> None:
    print(
        """usage: pulsing <command> [options]

Workspace:
  init        Bootstrap a Pulsing workspace (``.pulsing/``)
  history     List workspace checkpoints
  checkpoint  Save a workspace snapshot
  rollback    Restore files from a checkpoint

Runtime:
  actor       Start an Actor service (cluster member)
  inspect     Observe cluster via HTTP (non-member)
  examples    List or show built-in examples

Run `pulsing <command> --help` for command-specific help.
"""
    )
