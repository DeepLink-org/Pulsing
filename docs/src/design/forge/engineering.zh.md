# Pulsing Forge 工程说明（`pulsing-forge`）

> **产品文档**（面向用户）：[../../forge/index.md](../../forge/index.md) · [abstractions.md](../../forge/abstractions.md)
>
> **API 速查**：[python/pulsing/forge/README.md](https://github.com/DeepLink-org/pulsing/blob/main/python/pulsing/forge/README.md)
>
> **范围**：本文描述当前实现。目标产品边界、版本化 Session/Event/Evolution 协议以 [Forge 核心架构](core-architecture.zh.md) 为准。

---

## 1. 是什么

**Pulsing Forge** 提供 AI Agent 的**通用工具与环境运行时**：

- 在指定 **workspace** 里执行 shell、应用 patch、读写文件
- 通过 **sandbox** 约束权限（off / restricted / bwrap）
- 通过 **ToolSession** 把 plan、用户输入、token 预算交还给 Host（Forge 不做 UI）

Forge 是 **库**，不是完整 Agent 产品：不包含 LLM 调用、对话管理、登录或云同步。

---

## 2. 命名与生态

| 品牌 | 代码 |
|------|------|
| **Pulsing Forge** | `pulsing.forge` · crate `pulsing-forge` |
| **Pulsing** | Actor 集群与 RPC |
| **Craft** | Multi-Agent 参考应用 |

```text
Pulsing        → 通信与部署
Pulsing Forge  → 工具与环境
Craft          → 产品级 Multi-Agent（消费 Forge）
```

命名定案：[naming.md](./naming.md)

---

## 3. 抽象（实现映射）

| 概念 | Python | Rust |
|------|--------|------|
| Environment | `ForgeEnvironment` | `ToolRuntimeConfig` / `ToolCallContext` |
| Runtime | `LocalToolRuntime` | `ToolRuntime` |
| Host hooks | `ToolSession` | `trait ToolSession` |
| Result | `ToolResult` | `ToolResult` |
| 远程 worker | `ToolWorkerActor` | Rust `ForgeRuntime`（PyO3）；Python fallback |
| PyO3 绑定 | `RustForgeAdapter` | `pulsing._core.ForgeRuntime` |

详见 [abstractions.md](../../forge/abstractions.md)。

---

## 4. 架构

```
Host（Agent loop / Craft / CLI）
  │  LLM + UI + ToolSession impl
  ▼
RustForgeAdapter / ForgeEnvironment / ToolWorkerActor
  ▼
pulsing._core.ForgeRuntime → pulsing-forge ToolRuntime
  ▼
handlers + sandbox + PTY + UnifiedExec
  │
  └─► tell_forge_event → Host.on_forge_event（P2P 事件）
```

**边界**：

- Handler + sandbox = Forge 库（无 Actor、无 UI）
- Session 副作用 = `ToolSession` 回调
- 隔离与 gossip = Pulsing `ToolWorkerActor`
- 事件 = Actor `tell`（见 [craft-architecture.md](./craft-architecture.md)）

**一体化架构（Review）**：[craft-architecture.md](./craft-architecture.md)

---

## 5. 工具域（当前 MVP）

| 域 | 工具 | 状态 |
|----|------|------|
| Execution | `shell_command`, `exec_command`, `write_stdin`, `Bash` | 标准 wire 参数 + patch 拦截 + UnifiedExec 会话 |
| Filesystem | `apply_patch`, `view_image`, `Read/Glob/Grep/Edit/Write` | patch 验证 + 结构化 image 输出 |
| Session | `update_plan`, `new_context`, `get_context_remaining`, `request_user_input` | 经 `ToolSession` |

---

## 6. 仓库布局

```
crates/pulsing-forge/     # Rust handlers + sandbox
python/pulsing/forge/     # Python API + ToolWorkerActor
docs/src/forge/           # 产品介绍与抽象
```

---

## 7. Actor API（Python）

```python
@pul.remote
class ToolWorkerActor:
    def call_tool(self, name: str, arguments: dict) -> dict: ...
```

| 模式 | 用法 |
|------|------|
| 进程内 | `ForgeEnvironment(...).runtime()` |
| 隔离子进程 | `ToolWorkerActor.spawn(..., new_process=True)` |
| 共享 worker | `name="craft/ws/{id}/_tools", public=True` |

---

## 8. 路线图

| Phase | 内容 | 状态 |
|-------|------|------|
| 0 | 文档 + crate 空壳 + vendor | ✅ |
| 1 | Environment 抽象 + 三域工具 MVP | ✅ |
| 2 | UnifiedExec、PTY、tree-sitter、PyO3 ForgeRuntime | ✅ |
| 3 | Craft 集成：Rust worker/host + P2P tell 事件 | ✅ |
| 4 | 移除 Python 双实现、TUI stream、OS sandbox 测试 | 待做 |
| 5 | MCP、execpolicy、VirtualNamedActor | 待做 |

---

## 9. 参考实现（贡献者）

部分 handler（patch 解析、沙箱策略等）在开发阶段参考了业界开源 agent-tool 实践。**产品文档不展开对照表**；贡献者维护 vendor 与 sync 脚本时见仓库内 `vendor/` 与 `scripts/sync-codex-forge.sh`。

禁止引入完整 agent CLI / 登录 / TUI 等产品层 crate。

---

## 10. 相关文档

- [../../forge/index.md](../../forge/index.md) — **主介绍**
- [craft-architecture.md](./craft-architecture.md) — **Forge × Craft 一体化架构（Review）**
- [../../forge/abstractions.md](../../forge/abstractions.md) — 抽象详解
- [testing.md](./testing.md) — **测试体系（L0–L4）**
- [naming.md](./naming.md) — 包名与路径
- [craft-npc-refactor.md](./craft-npc-refactor.md) — Craft 集成（后续）

---

## 11. 验收

- [x] `cargo test -p pulsing-forge`
- [x] `pytest tests/python/test_pulsing_forge.py`
- [x] `ForgeEnvironment` 公开 API
- [x] 产品文档 `docs/forge/`
- [ ] sandbox 集成测试（seatbelt / bwrap）
- [x] Craft 默认 Rust backend（`RustForgeAdapter`）
- [x] Forge 事件统一 `tell_forge_event`
- [ ] 移除 Python handler 双实现（仅 fallback）
