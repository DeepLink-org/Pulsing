#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Embed Pulsing Forge in a **custom agent framework**.

Your framework owns the LLM loop, memory, and UI. Forge provides the tool
runtime (Read, Glob, shell, patch, plan hooks, …) — no Craft required.

Run (in-process, no cluster):

    uv run python examples/python/forge_custom_agent.py

Run with an isolated ToolWorkerActor (separate OS process via Pulsing):

    uv run python examples/python/forge_custom_agent.py --isolated

This example uses a **scripted mock LLM** so it works without API keys.
Replace ``ScriptedLLM`` with your real model client; keep ``ForgeToolBackend``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pulsing as pul
from pulsing.forge import (
    ForgeEnvironment,
    LocalToolSession,
    ParsedToolCall,
    PlanItem,
    StepStatus,
    ToolResult,
    ToolWorkerActor,
    ToolWorkerConfig,
    UpdatePlanArgs,
    forge_tool_definitions,
    openai_tool_message,
    to_openai_tools,
)

# ---------------------------------------------------------------------------
# 1. Your framework: session hooks (plan, user input, approvals)
# ---------------------------------------------------------------------------


class MyToolSession(LocalToolSession):
    """Host-side state that Forge session tools write into."""

    def __init__(self) -> None:
        super().__init__(token_budget=128_000)
        self.log: list[str] = []

    def update_plan(self, args: UpdatePlanArgs) -> None:
        super().update_plan(args)
        steps = ", ".join(f"{p.step}({p.status})" for p in args.plan)
        self.log.append(f"plan updated: {steps}")

    def request_user_input(self, arguments: dict[str, Any]) -> dict[str, Any]:
        # In a real UI, show a modal and wait for the user.
        questions = arguments.get("questions") or []
        self.log.append(f"user_input requested: {len(questions)} question(s)")
        return {"answers": {q.get("id", "confirm"): "yes" for q in questions}}


# ---------------------------------------------------------------------------
# 2. Your framework: Forge as the tool backend
# ---------------------------------------------------------------------------

DEMO_TOOL_NAMES = ["update_plan", "Glob", "Read", "shell_command"]


class ToolBackend(Protocol):
    def tool_schemas(self) -> list[dict[str, Any]]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...


@dataclass
class InProcessForgeBackend:
    """ForgeEnvironment in the same process as your agent loop."""

    cwd: Path
    session: MyToolSession = field(default_factory=MyToolSession)

    def __post_init__(self) -> None:
        env = ForgeEnvironment(
            cwd=str(self.cwd),
            sandbox_policy="off",
            session=self.session,
            auto_approve=True,
        )
        self._runtime = env.runtime()

    def tool_schemas(self) -> list[dict[str, Any]]:
        # Anthropic-shaped defs → OpenAI tools for the LLM client.
        defs = forge_tool_definitions(DEMO_TOOL_NAMES)
        available = set(self._runtime.tool_names())
        defs = [d for d in defs if d.get("name") in available]
        return to_openai_tools(defs)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        return self._runtime.call_tool(name, arguments)

    def close(self) -> None:
        self._runtime.close()


@dataclass
class IsolatedForgeBackend:
    """Forge tools in a ToolWorkerActor (child process). Host stays lightweight."""

    worker: Any  # ActorProxy

    def tool_schemas(self) -> list[dict[str, Any]]:
        return to_openai_tools(forge_tool_definitions(DEMO_TOOL_NAMES))

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        raw = await self.worker.call_tool(name, arguments)
        return ToolResult.from_dict(raw)

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# 3. Your framework: LLM client (mock here; swap for OpenAI/Anthropic/etc.)
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    call: ParsedToolCall


class ScriptedLLM:
    """Deterministic tool-call sequence for the demo."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace
        self._step = 0

    def next_tool_calls(self, _messages: list[dict[str, Any]]) -> list[ToolCall]:
        readme = self._workspace / "README.md"
        if not readme.is_file():
            readme = self._workspace / "README.zh.md"

        script = [
            ToolCall(
                ParsedToolCall(
                    id="call-plan",
                    name="update_plan",
                    arguments={
                        "plan": [
                            {"step": "Scan workspace", "status": "in_progress"},
                            {"step": "Read README", "status": "pending"},
                            {"step": "Run sanity check", "status": "pending"},
                        ],
                        "explanation": "Exploring the repo before answering.",
                    },
                )
            ),
            ToolCall(
                ParsedToolCall(
                    id="call-glob",
                    name="Glob",
                    arguments={"pattern": "README*", "path": str(self._workspace)},
                )
            ),
            ToolCall(
                ParsedToolCall(
                    id="call-read",
                    name="Read",
                    arguments={"file_path": str(readme)},
                )
            ),
            ToolCall(
                ParsedToolCall(
                    id="call-shell",
                    name="shell_command",
                    arguments={
                        "command": "echo forge-ok",
                        "workdir": str(self._workspace),
                    },
                )
            ),
        ]
        if self._step >= len(script):
            return []
        call = script[self._step]
        self._step += 1
        return [call]


# ---------------------------------------------------------------------------
# 4. Your framework: agent loop
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    max_turns: int = 8
    system_prompt: str = (
        "You are a coding agent. Use tools to inspect the workspace, "
        "then summarize what you found."
    )


class SimpleAgentFramework:
    """Minimal Host: messages in memory, tools via Forge, LLM pluggable."""

    def __init__(
        self,
        *,
        backend: ToolBackend,
        llm: ScriptedLLM,
        config: AgentConfig | None = None,
    ) -> None:
        self.backend = backend
        self.llm = llm
        self.config = config or AgentConfig()
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.config.system_prompt},
        ]

    async def arun(self) -> str:
        tools = self.backend.tool_schemas()
        print(f"Registered {len(tools)} tools for the LLM:")
        for t in tools:
            print(f"  - {t['function']['name']}")

        for turn in range(self.config.max_turns):
            calls = self.llm.next_tool_calls(self.messages)
            if not calls:
                return self._final_answer()

            for call in calls:
                tc = call.call
                print(
                    f"\n[turn {turn + 1}] LLM → tool {tc.name}"
                    f"({json.dumps(tc.arguments, ensure_ascii=False)[:120]}…)"
                )
                result = await self.backend.call_tool(tc.name, tc.arguments)
                preview = result.content[:200].replace("\n", "\\n")
                status = "ERROR" if result.is_error else "OK"
                print(f"  Forge → {status}: {preview}")

                self.messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                        ],
                    }
                )
                self.messages.append(openai_tool_message(tc.id, result))

        return self._final_answer()

    def _final_answer(self) -> str:
        session = getattr(self.backend, "session", None)
        if isinstance(session, MyToolSession) and session.plan:
            done = sum(1 for p in session.plan if p.status == StepStatus.COMPLETED)
            plan_summary = f"{done}/{len(session.plan)} plan steps completed"
        else:
            plan_summary = "plan tracked in host session"

        tool_msgs = [m for m in self.messages if m.get("role") == "tool"]
        return (
            f"Agent finished after {len(tool_msgs)} tool result(s). " f"{plan_summary}."
        )


# ---------------------------------------------------------------------------
# 5. Entry
# ---------------------------------------------------------------------------


async def _run_in_process(workspace: Path) -> None:
    session = MyToolSession()
    backend = InProcessForgeBackend(cwd=workspace, session=session)
    try:
        agent = SimpleAgentFramework(
            backend=backend,
            llm=ScriptedLLM(workspace),
        )
        print("== Custom agent framework + Forge (in-process) ==\n")
        answer = await agent.arun()
        print(f"\n== Final ==\n{answer}")
        if session.log:
            print("\n== Host session log ==")
            for line in session.log:
                print(f"  {line}")
        if session.plan:
            print("\n== Plan snapshot ==")
            for item in session.plan:
                print(f"  [{item.status}] {item.step}")
    finally:
        backend.close()


async def _run_isolated(workspace: Path) -> None:
    await pul.init()
    try:
        worker = await ToolWorkerActor.spawn(
            ToolWorkerConfig(cwd=str(workspace), auto_approve=True),
            public=False,
        )
        backend = IsolatedForgeBackend(worker=worker)
        agent = SimpleAgentFramework(
            backend=backend,
            llm=ScriptedLLM(workspace),
        )
        print("== Custom agent framework + Forge (isolated worker) ==\n")
        ping = await worker.ping()
        print(f"Worker: {ping}\n")
        answer = await agent.arun()
        print(f"\n== Final ==\n{answer}")
    finally:
        await pul.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="Run Forge tools inside ToolWorkerActor (Pulsing subprocess)",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Workspace root for tools (default: repo root)",
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()

    if args.isolated:
        asyncio.run(_run_isolated(workspace))
    else:
        asyncio.run(_run_in_process(workspace))


if __name__ == "__main__":
    main()
