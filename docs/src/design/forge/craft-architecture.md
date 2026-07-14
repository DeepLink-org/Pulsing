# Forge × Craft Architecture

> **User docs**: [Forge deployment](../../forge/deployment.md) · [Pulsing integration](../../forge/pulsing-integration.md)

Review-grade architecture for how Craft (reference Host) consumes Forge on Pulsing Actors.

---

## Layering

```text
CraftAgent     LLM, permissions UI, tool routing
ForgeBackend   Host + isolated worker unified entry
Forge          32 tools, sandbox, MCP, events
Pulsing        ask/tell, spawn, resolve, gossip
```

**Principles**

- Host owns LLM and product UI; Forge owns workspace execution
- Events via P2P `tell` (not in-process queues)
- Isolated tools in `ToolWorkerActor` child process

---

## Key actors (named)

| Actor | Name pattern |
|-------|----------------|
| CraftAgent | `craft/ws/{id}/{short}` |
| ForgeEventInbox | `{host}/events` |
| McpHubActor | `craft/ws/{id}/_mcp_hub` |
| CodeCellRegistryActor | `{host}/code_cells` |
| Shared ToolWorkerActor | `craft/ws/{id}/_tools` |

---

## Further reading

Detailed diagrams, event kinds, and review notes: **`craft-architecture.zh.md`** (Chinese, primary).

Related: [Engineering](engineering.md) · [Craft architecture](craft-architecture.md)
