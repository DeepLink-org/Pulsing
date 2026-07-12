# Pulsing Integration

How Forge uses Pulsing Actor capabilities — and what remains in-process.

---

## Capability matrix

| Pulsing feature | Forge usage | Maturity |
|-----------------|-------------|:--------:|
| `@remote` + spawn | `ToolWorkerActor`, inbox, MCP hub, code registry | ✅ |
| `resolve` (gossip) | Shared worker, inbox, hub, registry | ✅ |
| `ask` | Exec approval, permissions, code cell nested tools | ✅ |
| `tell` | Forge events, inbox → Host side effects | ✅ |
| `new_process` | Isolated `ToolWorkerActor` | ✅ |
| `ForgeBackend` | LOCAL / DEDICATED / SHARED | ✅ |
| Supervision policy | Manual respawn via supervisor | 🟡 |
| Queue / Topic | Not used — events in memory | ⚪ |
| Multi-node placement | Local spawn only | ⚪ |

---

## Message flows

### Tool execution

```text
CraftAgent.call_tool("Read", ...)
  → ForgeBackend
  → ForgeWorkerSupervisor → ToolWorkerActor (child process)
  → ToolResult
```

### Events (streaming)

```text
Worker / Host runtime
  → tell_forge_event(inbox)
  → ForgeEventInbox.on_forge_event
  → tell Host.on_forge_stream_event / on_forge_side_effect
```

### Approvals (blocking)

```text
Worker
  → ask Host.resolve_exec_approval
  → Host UI / PermissionChecker
  → decision dict
```

---

## What stays in-process (by design)

| Component | Reason |
|-----------|--------|
| `HybridForgeRuntime` | PyO3 callbacks, Session state, MCP in Rust |
| `ForgeWorkerSupervisor` | Cannot spawn child from actor mailbox |
| REPL `LocalToolRuntime` | Debug path without cluster |
| Extension handlers | Local I/O, no cross-node need |

This is **intentional layering**: Actor for deployment boundaries, not for every function call.

---

## Developer-facing surface

Most integrators should use:

```python
from pulsing.forge import ForgeEnvironment          # library / tests
from pulsing.forge import ForgeBackend, ...         # Craft-style host
```

Only advanced hosts need `tell_forge_event`, `ensure_forge_actors`, or custom gossip names.

---

## Constraints (read before extending)

1. **No `new_process` spawn inside actor mailbox handlers** — use Host process or delayed patterns.
2. **Isolated actors must pickle cleanly** — defer locks / exec managers to `on_start`.
3. **Separate approval sink from event sink** — inbox receives tells; Host receives asks.

---

## Next steps (P1+)

- Forge events → Pulsing `Queue` for audit and replay
- REPL remote worker mode (`resolve` shared worker)
- Optional multi-node worker placement

Design reference: [craft-architecture](../design/forge/craft-architecture.md)
