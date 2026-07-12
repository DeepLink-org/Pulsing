# Pulsing 部署

Forge 可 **纯进程内** 运行，也可通过 **Pulsing Actor** 做隔离与集群部署。

---

## ForgeBackend 三档

统一入口：`python/pulsing/forge/backend.py`

| 模式 | 枚举 | 行为 |
|------|------|------|
| **LOCAL** | `ForgeBackendMode.LOCAL` | 仅 Host runtime，无 worker |
| **DEDICATED** | `DEDICATED` | 每 Host 一个 `ToolWorkerActor`（经进程内 supervisor） |
| **SHARED** | `SHARED` | `resolve("craft/ws/{workspace_id}/_tools")` |

Craft 通过 `tool_host.py` 使用此路径，应用代码通常不直接 spawn。

---

## 命名 Forge Actor（Craft bootstrap）

具 gossip 名的 Craft Agent 在 `on_start` 调用 `ensure_forge_actors()`：

| Actor | Gossip 名 | 职责 |
|-------|-----------|------|
| `ForgeEventInbox` | `{host}/events` | 收集 tell 事件；流式转发 Host |
| `McpHubActor` | `craft/ws/{id}/_mcp_hub` | MCP refresh + 工具发现 |
| `CodeCellRegistryActor` | `{host}/code_cells` | Code Mode 控制面 |
| `ToolWorkerActor` | 子进程 | 隔离工具执行 |

双 sink 约定：

- **`_forge_host_name`** — 审批 / 权限 **ask** 目标
- **`_event_sink_name`** — Forge **tell** 目标（通常为 inbox）

---

## Worker 监管

`ForgeWorkerSupervisor` 为 **进程内类**（非 `@remote`）：

- `pul.spawn(..., new_process=True)` 拉起 worker
- 失败自动 respawn 一次
- **禁止**在 Actor mailbox 处理中 spawn 子进程

`ToolWorkerActor` 重资源延迟到 `on_start`，保证 isolated pickle 成功。

---

## Workspace 共享 Worker

```python
from pulsing.forge import spawn_shared_tool_worker, resolve_shared_tool_worker

await spawn_shared_tool_worker(workspace_id="myws", cwd="/path/to/repo")
worker = await resolve_shared_tool_worker("myws")
```

Gossip 名：`craft/ws/myws/_tools`。

---

## 选型

| 场景 | 模式 |
|------|------|
| 单测、REPL | LOCAL |
| 单 Agent 强隔离 | DEDICATED |
| 多 Agent 共享 sandbox | SHARED |
| 无集群 / 无 agent 名 | LOCAL |

---

## 相关

- [Pulsing 集成](pulsing-integration.zh.md)
- [Craft 一体化架构](../design/forge/craft-architecture.zh.md)
