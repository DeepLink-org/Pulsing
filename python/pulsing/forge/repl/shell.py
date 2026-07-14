# SPDX-License-Identifier: Apache-2.0
"""Interactive REPL loop (Nushell-inspired tables + structured commands)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TextIO

from pulsing.forge.repl.parse import parse_line
from pulsing.forge.repl.session import ForgeReplSession
from pulsing.forge.repl.trace import save_trace

_HELP = """
Forge session REPL — drive ToolRuntime directly (no LLM).

Invocation (Nushell-style):
  call Read {"file_path": "README.md"}
  Read --file_path README.md
  Glob --pattern "*.py"

Meta (: prefix or bare keyword):
  help | :help          this help
  tools                 registered tools (table)
  session | plan | events
  approve auto|ask
  replay [dry] [verify] | replay all
  trace save PATH
  quit

Replay: ``pulsing forge repl --trace file.jsonl`` then ``replay`` step-by-step
"""


def _prompt(session: ForgeReplSession) -> str:
    cwd = session.cwd.name or str(session.cwd)
    mode = session.approval_mode[0].upper()
    return f"forge ⟨{cwd}⟩ {mode}⟩ "


def _print_help(out: TextIO) -> None:
    out.write(_HELP.strip() + "\n")


def _run_call(
    session: ForgeReplSession, tool: str, arguments: dict, out: TextIO
) -> None:
    result = session.call_tool(tool, arguments)
    payload = {
        "tool": tool,
        "is_error": result.is_error,
        "content": result.content[:2000],
    }
    if result.structured is not None:
        payload["structured"] = result.structured
    out.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _handle_meta(session: ForgeReplSession, cmd: str, args: dict, out: TextIO) -> bool:
    """Returns False to exit REPL."""
    rest: list[str] = list(args.get("rest") or [])

    if cmd == "help":
        _print_help(out)
        return True
    if cmd == "quit":
        return False
    if cmd == "tools":
        out.write(session.format_tools_table() + "\n")
        return True
    if cmd == "session":
        out.write(session.format_session_table() + "\n")
        return True
    if cmd == "plan":
        out.write(session.format_plan_table() + "\n")
        return True
    if cmd == "events":
        out.write(session.format_events_table() + "\n")
        return True
    if cmd == "approve":
        if not rest:
            out.write(f"approval mode: {session.approval_mode}\n")
            return True
        mode = rest[0].lower()
        if mode not in ("auto", "ask"):
            out.write("usage: approve auto|ask\n")
            return True
        session.set_approval_mode(mode)  # type: ignore[arg-type]
        out.write(f"approval → {mode}\n")
        return True
    if cmd == "replay":
        dry = "dry" in rest
        verify = "verify" in rest
        if "all" in rest:
            for line in session.replay_all(dry_run=dry, verify=verify):
                out.write(line + "\n")
            return True
        out.write(session.replay_step(dry_run=dry, verify=verify) + "\n")
        return True
    if cmd == "trace":
        if not rest:
            out.write("usage: trace save PATH | trace show\n")
            return True
        sub = rest[0].lower()
        if sub == "show":
            if session.record_path:
                out.write(
                    f"recording → {session.record_path} ({len(session.trace.records)} lines)\n"
                )
            else:
                out.write("(not recording)\n")
        elif sub == "save" and len(rest) > 1:
            session.record_path = Path(rest[1])
            save_trace(session.record_path, session.trace)
            out.write(
                f"saved {len(session.trace.records)} records → {session.record_path}\n"
            )
        else:
            out.write("usage: trace save PATH | trace show\n")
        return True
    if cmd == "call" and rest:
        from pulsing.forge.repl.parse import parse_json_or_flags

        tool = rest[0]
        arguments = parse_json_or_flags(" ".join(rest[1:]))
        _run_call(session, tool, arguments, out)
        return True
    out.write(f"unknown command: {cmd}\n")
    return True


def run_repl(session: ForgeReplSession, *, stdin=None, stdout=None) -> None:
    out: TextIO = stdout or sys.stdout
    inp = stdin or sys.stdin
    _print_help(out)
    if session._loaded_trace is not None:
        n = len(session._loaded_trace.tool_calls())
        out.write(f"loaded trace: {n} tool calls (replay / replay all)\n")

    while True:
        try:
            if inp is sys.stdin:
                line = input(_prompt(session))
            else:
                line = inp.readline()
                if not line:
                    break
        except (EOFError, KeyboardInterrupt):
            out.write("\n")
            break

        cmd, args = parse_line(line)
        if cmd == "noop":
            continue
        if cmd == "unknown":
            out.write(f"unknown: {args.get('line')!r} (help)\n")
            continue
        if cmd == "call":
            _run_call(session, args["tool"], args.get("arguments") or {}, out)
            continue
        if not _handle_meta(session, cmd, args, out):
            break
