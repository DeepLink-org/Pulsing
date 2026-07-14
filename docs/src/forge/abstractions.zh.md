# Pulsing Forge 抽象模型

Forge 的设计目标：**给 Agent 一个标准、可沙箱化的「工作环境」**，而不是绑定某一种 LLM 产品或 CLI。

---

## 1. 核心概念

| 概念 | 类型 | 职责 |
|------|------|------|
| **ForgeEnvironment** | `ForgeEnvironment` | 工作区根目录、沙箱策略、Host 会话钩子 |
| **Runtime** | `LocalToolRuntime` / `ToolWorkerActor` | 按名称分发工具调用，返回 `ToolResult` |
| **ToolCallContext** | 每次调用注入 | `cwd` + sandbox + `session` 快照 |
| **ToolSession** | Host 实现 | Plan、用户输入、context 预算等产品行为 |
| **ToolExecutor** | Rust handler trait | 单一工具的实现（shell、patch、plan…） |

关系：

```text
Host 创建 ForgeEnvironment
  → env.runtime() 得到 Runtime
  → runtime.call_tool("shell_command", {...})
  → Runtime 构造 ToolCallContext 并调用对应 handler
  → 若工具需要 UI（如 request_user_input），handler 回调 ToolSession
```

---

## 2. 环境与沙箱

**Environment** 回答：Agent 在哪个目录、以什么权限操作机器？

| 字段 | 含义 |
|------|------|
| `cwd` | 工具解析相对路径的根目录 |
| `sandbox_policy` | `off` · `restricted` · `bwrap`（平台相关） |
| `dangerously_disable_sandbox` | 显式关闭沙箱（仅开发/受信场景） |

沙箱只约束 **Forge 管理的工具路径**（shell、patch 等），不替代 OS 级容器。Host 仍可在更外层做 VM / 容器隔离；Forge 的 `ToolWorkerActor` 则提供进程级隔离部署。

---

## 3. Session：Host 与 Agent 的边界

部分工具是 **「环境能力」**（读文件、跑命令），部分是 **「产品能力」**（问用户、更新计划、换 context）。

Forge 把产品能力收敛到 **`ToolSession`**，避免在工具库里写 UI：

```python
class ToolSession(Protocol):
    def update_plan(self, args: UpdatePlanArgs) -> None: ...
    def request_new_context(self) -> None: ...
    def tokens_remaining(self) -> int | None: ...
    def request_user_input(self, arguments: dict) -> dict: ...
```

| 实现 | 用途 |
|------|------|
| `LocalToolSession` | 本地开发、测试；内存 plan + 可选 callback |
| `NullToolSession` | 只要 shell/文件、不需要 plan/UI |
| 自定义 | Craft TUI、Web dashboard、企业审批流 |

**原则**：Forge 发出「需要用户确认」的**结构化请求**；Host 决定如何展示与阻塞。

---

## 4. 工具域（Tool Domains）

工具按**能力域**注册，而非按「抄哪家产品」划分：

### Execution

- `shell_command` — 标准 wire 参数（`command`/`timeout_ms`/`login`/`sandbox_permissions`），shell 内 `apply_patch` 自动拦截
- `exec_command` / `write_stdin` — UnifiedExec 会话（`session_id`、`yield_time_ms`、`max_output_tokens`）
- `Bash` — `shell_command` 别名

### Filesystem

- `apply_patch` — 结构化 patch；拒绝隐式裸 patch；支持 `command: ["apply_patch", "..."]`
- `view_image` — 返回 `content_items`（`input_image` + data URL）；`detail`: `high` | `original`
- `Read` / `Glob` / `Grep` / `Edit` / `Write` — 通用文件 helper

### Session

- `update_plan` · `new_context` · `get_context_remaining` · `request_user_input`

新增工具 = 新 `ToolExecutor` + 注册到 `ToolRuntime`，可选是否依赖 `ToolSession`。

---

## 5. 部署模式

| 模式 | API | 适用 |
|------|-----|------|
| **In-process** | `ForgeEnvironment` → `LocalToolRuntime` | 单进程 Agent、单元测试 |
| **Isolated actor** | `ToolWorkerActor` + Pulsing spawn | 子进程 / 集群 worker |
| **Shared worker** | gossip name `craft/ws/{id}/_tools` | 多 Agent 共享同一 sandbox 环境 |

两种模式 **工具名与参数一致**，Host 只换 Runtime 后端。

---

## 6. 与 Agent 框架集成

```text
LangChain / AutoGen / 自研 loop
    │
    ├─ 把 Forge 工具包装成 framework Tool
    │
    └─ 或直接 call_tool("shell_command", {...})
```

Forge **不**管理对话历史、不调用 LLM、不选模型——这些留在 Host。Forge 只保证：**给定 tool name + JSON args，在 Environment 里安全执行并返回文本结果**。

---

## 7. Rust / Python 对齐

| Python | Rust crate `pulsing-forge` |
|--------|----------------------------|
| `ForgeEnvironment` | `ToolRuntimeConfig` + `ToolCallContext` |
| `LocalToolRuntime` | `ToolRuntime` |
| `ToolSession` | `trait ToolSession` |
| `ToolResult` | `ToolResult` |

Python 层用于 Actor worker 绑定与 fallback；**默认执行路径**为 Rust `ForgeRuntime`（PyO3）。一体化架构见 [craft-architecture.md](../design/forge/craft-architecture.md)。

---

## 8. 非目标（明确边界）

- 不是完整 Agent OS（无 LLM、无 memory 产品、无 MCP 目录 UI）
- 不是容器编排（Kubernetes 层由部署解决）
- 不强制某一种 patch 或 shell JSON schema 以外的工具命名——但默认工具面覆盖主流 coding agent 需求

实现与路线图：[design/engineering.md](../design/forge/engineering.md)
