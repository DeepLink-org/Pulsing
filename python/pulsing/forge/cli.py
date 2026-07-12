# SPDX-License-Identifier: Apache-2.0
"""Forge CLI: ``pulsing forge``."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pulsing.forge.repl.cli import main as repl_main


def main_pulsing_forge(argv: Sequence[str] | None = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] in ("-h", "--help", "help"):
        print(
            """usage: pulsing forge <command> [options]

Commands:
  repl    Interactive session REPL (Rust when available, else Python)

Examples:
  pulsing forge repl --cwd .
  pulsing forge repl --trace trace.jsonl --replay-all --verify
"""
        )
        return
    if args[0] != "repl":
        print(f"Unknown forge command: {args[0]!r}", file=sys.stderr)
        print("Run `pulsing forge --help` for usage.", file=sys.stderr)
        raise SystemExit(2)
    repl_main(args[1:])


def main() -> None:
    main_pulsing_forge(sys.argv[1:])
