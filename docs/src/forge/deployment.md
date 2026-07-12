# Deployment on Pulsing

Forge can run **entirely in-process** or use **Pulsing Actors** for isolation and cluster deployment.

---

## ForgeBackend modes

Unified entry: `python/pulsing/forge/backend.py`

| Mode | Enum | Behavior |
|------|------|----------|
| **LOCAL** | `ForgeBackendMode.LOCAL` | Host runtime only — no worker |
| **DEDICATED** | `DEDICATED` | One `ToolWorkerActor` per host (via in-process supervisor) |
| **SHARED** | `SHARED` | `resolve("craft/ws/{workspace_id}/_tools")` |

```python
from pulsing.forge import ForgeBackend, ForgeHostConfig, ForgeIsolatedWorker, create_host_runtime

host = create_host_runtime(ForgeHostConfig(cwd=".", auto_approve=False))
worker = ForgeIsolatedWorker.dedicated(ToolWorkerConfig(cwd=".", host_name="craft/ws/x/agent"))
backend = ForgeBackend(host=host, worker=worker, event_sink_name="craft/ws/x/agent/events")
result = await backend.call_tool("Read", {"file_path": "README.md"})
```

Craft uses this path via `tool_host.py` — application code rarely touches spawn directly.

---

## Named Forge actors (Craft bootstrap)

When a Craft agent has a gossip name, `ensure_forge_actors()` spawns:

| Actor | Gossip name | Role |
|-------|-------------|------|
| `ForgeEventInbox` | `{host}/events` | Collect tell events; forward streams to Host |
| `McpHubActor` | `craft/ws/{id}/_mcp_hub` | MCP refresh + tool discovery |
| `CodeCellRegistryActor` | `{host}/code_cells` | Code Mode `exec` / `wait` control plane |
| `ToolWorkerActor` | child process | Isolated tool execution |

Host name vs event sink:

- **`_forge_host_name`** — exec approval / permissions **ask** target
- **`_event_sink_name`** — Forge **tell** target (usually inbox)

---

## Worker supervision

`ForgeWorkerSupervisor` (in-process, **not** `@remote`) wraps `ToolWorkerActor`:

- Spawns child with `pul.spawn(..., new_process=True)`
- Retries once on failure
- Must **not** spawn from inside another actor's mailbox handler

`ToolWorkerActor` defers heavy init to `on_start` so isolated pickle succeeds.

---

## Shared workspace worker

```python
from pulsing.forge import spawn_shared_tool_worker, resolve_shared_tool_worker

await spawn_shared_tool_worker(workspace_id="myws", cwd="/path/to/repo")
worker = await resolve_shared_tool_worker("myws")
```

Gossip name: `craft/ws/myws/_tools` (`naming.shared_tool_worker_name`).

---

## When to use which mode

| Scenario | Mode |
|----------|------|
| Unit tests, REPL | LOCAL |
| Single agent, strong isolation | DEDICATED |
| Many agents, one repo sandbox | SHARED |
| No Pulsing cluster / no agent name | LOCAL or DEDICATED without inbox actors |

---

## Related

- [Pulsing Integration](pulsing-integration.md) — ask/tell patterns
- [Craft architecture](../design/forge/craft-architecture.md) — full integration diagram
