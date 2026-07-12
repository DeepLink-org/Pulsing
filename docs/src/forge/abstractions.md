# Abstractions

Forge provides a **standard, sandboxed workspace** for agents — independent of any single LLM product or CLI.

---

## Core concepts

| Concept | Type | Responsibility |
|---------|------|----------------|
| **ForgeEnvironment** | dataclass | Workspace root, sandbox policy, session hooks |
| **Runtime** | `LocalToolRuntime` / `HybridForgeRuntime` / `ToolWorkerActor` | Dispatch `call_tool`, return `ToolResult` |
| **ToolCallContext** | per-call | `cwd`, sandbox, `session`, exec manager |
| **ToolSession** | Host protocol | Plan, user input, token budget |
| **ToolExecutor** | Rust trait | Single-tool implementation |

```text
Host creates ForgeEnvironment
  → env.runtime()
  → runtime.call_tool("shell_command", {...})
  → handler runs in ToolCallContext
  → optional ToolSession callback for UI tools
```

---

## Environment and sandbox

| Field | Meaning |
|-------|---------|
| `cwd` | Root for relative paths |
| `sandbox_policy` | `off` · `restricted` · `bwrap` |
| `dangerously_disable_sandbox` | Explicit dev-only bypass |

Sandbox applies to **Forge-managed tools** only. Host may add VM/container boundaries; `ToolWorkerActor` adds process isolation.

---

## ToolSession boundary

Product-facing tools delegate to Host:

```python
class ToolSession(Protocol):
    def update_plan(self, args: UpdatePlanArgs) -> None: ...
    def request_new_context(self) -> None: ...
    def tokens_remaining(self) -> int | None: ...
    def request_user_input(self, arguments: dict) -> dict: ...
```

| Implementation | Use |
|----------------|-----|
| `LocalToolSession` | Dev/tests with optional callbacks |
| `NullToolSession` | Shell/files only |
| Custom | Craft TUI, web dashboard, enterprise approval |

**Principle**: Forge sends structured requests; Host decides presentation and blocking.

---

## Deployment modes

| Mode | API | When |
|------|-----|------|
| In-process | `ForgeEnvironment.runtime()` | Unit tests, single-process agents |
| Isolated | `ToolWorkerActor` + `pul.spawn(..., new_process=True)` | Child-process sandbox |
| Shared | `resolve("craft/ws/{id}/_tools")` | Multiple agents, one worker |

Same tool names and args in all modes.

---

## Rust / Python split

| Python | Rust `pulsing-forge` |
|--------|----------------------|
| `ForgeEnvironment` | `ToolRuntimeConfig` |
| `HybridForgeRuntime` | `ToolRuntime` + handlers |
| `ToolSession` | `trait ToolSession` |
| Actor bindings, Code Mode, Extension | Hot-path exec / patch / MCP |

Default path: Rust via PyO3; Python fallback for tools without Rust handlers.

---

## Non-goals

- Not a full agent OS (no LLM, no MCP catalog UI)
- Not container orchestration
- Not tied to one patch or shell JSON schema beyond the default 32-tool surface

Implementation: [Engineering notes](../design/forge/engineering.md) · Architecture with Craft: [craft-architecture](../design/forge/craft-architecture.md)
