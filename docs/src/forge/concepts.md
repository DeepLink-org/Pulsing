# Core Concepts

## Host vs Forge

| Layer | Owns | Does not own |
|-------|------|--------------|
| **Host** | LLM loop, chat history, UI, approvals display | Shell execution, patch apply |
| **Forge** | Tool dispatch, sandbox, MCP runtime, handlers | Model choice, login, product memory |

Integration contract:

```text
Host → runtime.call_tool(name, args) → ToolResult
Host ← ToolSession callbacks ← plan / user_input / permissions
Host ← tell_forge_event ← exec streams, tool_begin/end
```

---

## Three core types

| Type | Role |
|------|------|
| **`ForgeEnvironment`** | Workspace root (`cwd`), sandbox policy, session hooks |
| **`ToolSession`** | Host-implemented product behavior (plan, prompts, token budget) |
| **`ToolResult`** | `{ content, is_error, structured? }` — uniform tool output |

---

## Runtime stack

```text
ForgeEnvironment
  └─ HybridForgeRuntime (default with Rust)
       ├─ Rust ForgeRuntime — most handlers + MCP
       └─ Python LocalToolRuntime — exec/wait, Extension, Code Mode fallback
```

For isolation, swap the bottom layer for **`ToolWorkerActor`** via [`ForgeBackend`](deployment.md).

---

## Tool domains

Tools are grouped by **capability**, not by vendor:

- **Execution** — shell, PTY, unified exec
- **Filesystem** — read, patch, image
- **Session** — plan, context, user input
- **Discovery** — plugins, tool_search
- **MCP** — resources, dynamic MCP tools
- **Code Mode** — `exec` / `wait` cells
- **Extension** — memories, skills, web.run, web_search

See [Tools (32)](tools.md).

---

## Quality verification

Forge validates tool registration, default callability, and core integration paths in CI (`test_hybrid_forge_callable.py`, `test_pulsing_forge.py`). Capability claims should match test results.

```bash
pytest tests/python/test_hybrid_forge_callable.py tests/python/test_pulsing_forge.py -q
```

---

## Related

- [Abstractions](abstractions.md) — API-level detail
- [Pulsing Integration](pulsing-integration.md) — actors and events
