# SPDX-License-Identifier: Apache-2.0
"""Forge REPL session — LocalToolRuntime + trace recording."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pulsing.forge.events import ForgeEvent
from pulsing.forge.integrated import FORGE_TOOL_NAMES
from pulsing.forge.p2p_session import P2PToolSession
from pulsing.forge.result import ToolResult
from pulsing.forge.runtime import LocalToolRuntime
from pulsing.forge.session import PlanItem, StepStatus, UpdatePlanArgs
from pulsing.forge.repl.trace import TraceLog, TraceRecord, load_trace, save_trace

ApprovalMode = Literal["auto", "ask"]


@dataclass
class ReplToolSession(P2PToolSession):
    """In-process session with REPL-friendly approval hooks."""

    approval_mode: ApprovalMode = "auto"
    repl_input: Any = field(default=None, repr=False)

    def request_user_input(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from pulsing.forge.session_input import validate_request_user_input

        validate_request_user_input(arguments)
        self._send(
            ForgeEvent(
                kind="user_input_request",
                payload=dict(arguments),
            )
        )
        if self.approval_mode == "auto":
            questions = arguments.get("questions") or []
            if questions and isinstance(questions[0], dict):
                opts = questions[0].get("options") or []
                if opts:
                    return {
                        "answers": {
                            questions[0].get("id", "q0"): opts[0].get("label", opts[0])
                        }
                    }
            return {"answers": {}}
        if self.repl_input is not None:
            return self.repl_input(
                f"user_input: {json.dumps(arguments, ensure_ascii=False)[:200]}"
            )
        return super().request_user_input(arguments)

    def request_plugin_install(self, args: dict[str, Any]) -> bool:
        if self.approval_mode == "auto":
            return True
        if self.repl_input is not None:
            ans = self.repl_input(f"plugin_install {args.get('tool_id')}? [y/N] ")
            return str(ans).strip().lower() in ("y", "yes", "allow")
        return False


@dataclass
class ForgeReplSession:
    cwd: Path
    sandbox_policy: str = "off"
    dangerously_disable_sandbox: bool = False
    approval_mode: ApprovalMode = "auto"
    record_path: Path | None = None

    session: ReplToolSession = field(init=False)
    runtime: LocalToolRuntime = field(init=False)
    trace: TraceLog = field(init=False)
    events: list[ForgeEvent] = field(init=False, default_factory=list)
    _replay_index: int = field(init=False, default=0)
    _loaded_trace: TraceLog | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.cwd = Path(self.cwd).resolve()
        self.session = ReplToolSession(approval_mode=self.approval_mode)
        self.session.repl_input = self._prompt
        self.runtime = LocalToolRuntime(
            cwd=str(self.cwd),
            sandbox_policy=self.sandbox_policy,
            dangerously_disable_sandbox=self.dangerously_disable_sandbox,
            session=self.session,
        )
        self.trace = TraceLog()
        self.events = []

    def load_replay_trace(self, path: str | Path) -> int:
        self._loaded_trace = load_trace(path)
        self._replay_index = 0
        return len(self._loaded_trace.tool_calls())

    def set_approval_mode(self, mode: ApprovalMode) -> None:
        self.approval_mode = mode
        self.session.approval_mode = mode

    def _prompt(self, message: str) -> str:
        try:
            return input(f"{message}").strip()
        except EOFError:
            return ""

    def _capture_events(self) -> None:
        for ev in self.session.drain_pending():
            self.events.append(ev)
            if self.record_path is not None:
                self.trace.append(
                    TraceRecord(
                        seq=0,
                        kind="forge_event",
                        event=ev.to_dict(),
                    )
                )

    def _session_snapshot(self) -> dict[str, Any]:
        return {
            "cwd": str(self.cwd),
            "sandbox_policy": self.sandbox_policy,
            "plan": [p.to_dict() for p in self.session.plan],
            "new_context_requested": self.session.new_context_requested,
            "tokens_remaining": self.session.tokens_remaining(),
        }

    def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        args = dict(arguments or {})
        result = self.runtime.call_tool(name, args)
        self._capture_events()
        if self.record_path is not None:
            self.trace.append(
                TraceRecord(
                    seq=0,
                    kind="tool_call",
                    tool=name,
                    arguments=args,
                    result=result.to_dict(),
                )
            )
            save_trace(self.record_path, self.trace)
        return result

    def replay_step(
        self,
        *,
        dry_run: bool = False,
        verify: bool = False,
    ) -> str:
        if self._loaded_trace is None:
            return "no trace loaded; start with --trace FILE"
        calls = self._loaded_trace.tool_calls()
        if self._replay_index >= len(calls):
            return "replay complete"
        rec = calls[self._replay_index]
        self._replay_index += 1
        tool = rec.tool or "?"
        args = dict(rec.arguments or {})
        if dry_run:
            return (
                f"dry-run #{rec.seq} call {tool} {json.dumps(args, ensure_ascii=False)}"
            )
        out = self.call_tool(tool, args)
        if verify and rec.result is not None:
            exp_err = bool(rec.result.get("is_error"))
            if out.is_error != exp_err:
                return (
                    f"verify FAIL #{rec.seq} {tool}: is_error expected {exp_err} got {out.is_error}\n"
                    f"{out.content[:500]}"
                )
            if (
                not out.is_error
                and rec.result.get("content")
                and out.content != rec.result.get("content")
            ):
                return f"verify WARN #{rec.seq} {tool}: content differs (non-error)"
        flag = "ERR" if out.is_error else "ok"
        preview = out.content[:400].replace("\n", "\\n")
        return f"replay #{rec.seq} {tool} [{flag}] {preview}"

    def replay_all(self, *, dry_run: bool = False, verify: bool = False) -> list[str]:
        lines: list[str] = []
        while True:
            msg = self.replay_step(dry_run=dry_run, verify=verify)
            lines.append(msg)
            if "complete" in msg or msg.startswith("no trace"):
                break
        return lines

    def fork_trace(self, step: int) -> None:
        """Apply session snapshots and tool calls from trace up to ``step`` (replay index)."""
        if self._loaded_trace is None:
            return
        self._replay_index = 0
        for rec in self._loaded_trace.records:
            if rec.seq > step:
                break
            if rec.kind == "session" and rec.session:
                self._apply_session(rec.session)
            elif rec.kind == "tool_call" and rec.seq <= step:
                if rec.tool and rec.tool in FORGE_TOOL_NAMES:
                    self.call_tool(rec.tool, rec.arguments)
                self._replay_index += 1

    def _apply_session(self, snap: dict[str, Any]) -> None:
        plan_raw = snap.get("plan") or []
        items = [
            PlanItem(
                step=str(p.get("step", "")), status=p.get("status", StepStatus.PENDING)
            )
            for p in plan_raw
        ]
        if items:
            self.session.update_plan(UpdatePlanArgs(plan=items))
        if snap.get("new_context_requested"):
            self.session.new_context_requested = True
        if "tokens_remaining" in snap:
            self.session.token_budget = snap.get("tokens_remaining")

    def format_session_table(self) -> str:
        from pulsing.forge.repl.render import render_table

        snap = self._session_snapshot()
        rows = [[k, str(v)] for k, v in snap.items() if k != "plan"]
        if self.session.plan:
            rows.append(["plan", f"{len(self.session.plan)} items"])
        rows.append(["approval", self.approval_mode])
        rows.append(["events", str(len(self.events))])
        if self._loaded_trace:
            total = len(self._loaded_trace.tool_calls())
            rows.append(["replay", f"{self._replay_index}/{total}"])
        return render_table(["field", "value"], rows)

    def format_plan_table(self) -> str:
        from pulsing.forge.repl.render import render_table

        if not self.session.plan:
            return "(empty plan)"
        rows = [[p.step, str(p.status)] for p in self.session.plan]
        return render_table(["step", "status"], rows)

    def format_tools_table(self) -> str:
        from pulsing.forge.repl.render import render_table
        from pulsing.forge.integrated import FORGE_HOST_TOOL_NAMES

        rows = [
            [name, "host" if name in FORGE_HOST_TOOL_NAMES else "isolated"]
            for name in sorted(FORGE_TOOL_NAMES)
        ]
        return render_table(["tool", "zone"], rows)

    def format_events_table(self, limit: int = 12) -> str:
        from pulsing.forge.repl.render import render_table

        recent = self.events[-limit:]
        if not recent:
            return "(no events)"
        rows = []
        for ev in recent:
            preview = json.dumps(ev.payload, ensure_ascii=False)[:60]
            rows.append([ev.kind, ev.source or "", preview])
        return render_table(["kind", "source", "payload"], rows)
