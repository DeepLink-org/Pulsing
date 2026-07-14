# Pulsing 集成

Forge 如何使用 Pulsing Actor 能力，以及哪些部分刻意保持进程内。

---

## 能力矩阵

| Pulsing 能力 | Forge 用法 | 成熟度 |
|-------------|-----------|:------:|
| `@remote` + spawn | `ToolWorkerActor`、inbox、MCP hub、code registry | ✅ |
| `resolve` | 共享 worker、inbox、hub、registry | ✅ |
| `ask` | 审批、权限、Code Mode 嵌套工具 | ✅ |
| `tell` | Forge 事件、inbox → Host 副作用 | ✅ |
| `new_process` | 隔离 `ToolWorkerActor` | ✅ |
| `ForgeBackend` | LOCAL / DEDICATED / SHARED | ✅ |
| Supervision | supervisor 手工 respawn | 🟡 |
| Queue / Topic | 未用，事件在内存 | ⚪ |
| 多节点 placement | 仅本机 spawn | ⚪ |

---

## 消息路径

### 工具执行

```text
CraftAgent.call_tool → ForgeBackend → Supervisor → ToolWorkerActor → ToolResult
```

### 事件（流式）

```text
Worker/Host → tell inbox → ForgeEventInbox → tell Host（stream / side_effect）
```

### 审批（同步）

```text
Worker → ask Host.resolve_exec_approval → UI → decision
```

---

## 刻意进程内的部分

| 组件 | 原因 |
|------|------|
| `HybridForgeRuntime` | PyO3 回调、Session、Rust MCP |
| `ForgeWorkerSupervisor` | 不能在 mailbox 内 spawn 子进程 |
| REPL | 调试路径 |
| Extension | 本地 I/O |

Actor 用于**部署边界**，不是每个函数调用都 Actor 化。

---

## 对外 API（集成者）

```python
from pulsing.forge import ForgeEnvironment      # 库 / 单测
from pulsing.forge import ForgeBackend, ...     # 类 Craft Host
```

只有高级 Host 才需要直接操作 `tell_forge_event` 或 gossip 命名。

---

## 扩展前必读

1. **不要在 Actor mailbox 里 `new_process` spawn**
2. **隔离 Actor 必须可 pickle** — 重对象放 `on_start`
3. **审批 sink 与事件 sink 分离** — inbox 收 tell，Host 收 ask

---

## 后续（P1+）

- 事件进 Pulsing `Queue`（审计 / replay）
- REPL 远程 worker 模式
- 跨节点 worker placement

设计详图：[Craft 一体化架构](../design/forge/craft-architecture.zh.md)
