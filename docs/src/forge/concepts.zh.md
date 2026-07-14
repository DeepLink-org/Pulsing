# 核心概念

## Host 与 Forge

| 层 | 负责 | 不负责 |
|----|------|--------|
| **Host** | LLM 循环、对话、UI、审批展示 | 执行 shell、应用 patch |
| **Forge** | 工具分发、沙箱、MCP、handler | 选模型、登录、产品级记忆 |

```text
Host → call_tool(name, args) → ToolResult
Host ← ToolSession 回调 ← plan / 用户输入 / 权限
Host ← tell_forge_event ← exec 流、tool_begin/end
```

---

## 三个核心类型

| 类型 | 作用 |
|------|------|
| **`ForgeEnvironment`** | 工作区根目录、沙箱策略、会话钩子 |
| **`ToolSession`** | Host 实现的产品能力（plan、弹窗、token） |
| **`ToolResult`** | 统一返回 `{ content, is_error, structured? }` |

---

## 运行时栈

```text
ForgeEnvironment
  └─ HybridForgeRuntime（有 Rust 时默认）
       ├─ Rust ForgeRuntime — 多数 handler + MCP
       └─ Python LocalToolRuntime — exec/wait、Extension、Code Mode fallback
```

隔离执行时，通过 [`ForgeBackend`](deployment.zh.md) 换为 **`ToolWorkerActor`**。

---

## 工具域

按**能力**划分，而非按厂商：

- **Execution** — shell、PTY、统一 exec
- **Filesystem** — 读写、patch、图像
- **Session** — plan、context、用户输入
- **Discovery** — 插件、tool_search
- **MCP** — 资源、动态 MCP 工具
- **Code Mode** — `exec` / `wait`
- **Extension** — memories、skills、web.run、web_search

见 [工具清单（32）](tools.zh.md)。

---

## 质量验证

Forge 在 CI 中验证工具注册、默认可调用性与核心集成路径（`test_hybrid_forge_callable.py`、`test_pulsing_forge.py`）。

```bash
pytest tests/python/test_hybrid_forge_callable.py tests/python/test_pulsing_forge.py -q
```

---

## 相关

- [抽象模型](abstractions.zh.md)
- [Pulsing 集成](pulsing-integration.zh.md)
