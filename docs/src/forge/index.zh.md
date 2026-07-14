# Pulsing Forge — Agent 执行环境

> **读者**：想集成 coding agent 的开发者、框架作者、技术决策者
> **版本快照**：2026-05 · **32 工具开箱 callable**（Hybrid + MCP runtime）
> **代码入口**：`pip install pulsing` → `from pulsing.forge import ForgeEnvironment`

---

## 一句话

**Pulsing Forge 是给 AI Agent 用的「工作环境运行时」**——在指定 workspace 里安全地跑 shell、改文件、读图像、维护计划、连接 MCP，而不必在每个项目里重新拼 subprocess、sandbox、工具 schema 和审批流。

Forge **不是**某个 CLI 的换皮，也 **不是**完整 Agent 产品。它是一层可嵌入的库：**Host 管 LLM 与 UI，Forge 管环境 + 工具执行**。

---

## 为什么需要 Forge

几乎每个 coding agent 项目都会重复造同一套轮子：

| 重复劳动 | 典型痛点 |
|----------|----------|
| Shell / Read / patch | 每家参数名不同，错误处理各自为政 |
| 沙箱 | 策略散落，dev 与 prod 行为不一致 |
| Plan / 弹窗 / token 预算 | 与工具层强耦合，难以换 UI |
| 插件与 MCP | handler 有了，runtime 没接上，装完 server 调不动 |
| 本地调试 vs 隔离执行 | 两套代码路径，行为漂移 |

Forge 把这些收敛成 **一套工具面 + 一个 Environment 抽象**，Host 只需实现 `ToolSession`（产品侧回调），其余交给运行时。

---

## 设计哲学

### 1. 环境优先，而非对话优先

Forge 回答的问题是：**「给定 tool name + JSON args，在 workspace 里怎么安全执行？」**

它不管理对话历史、不选模型、不做登录——这些属于 Host。这种切分让 Forge 可以嵌入 LangChain、自研 loop、Craft、企业审批流，而不绑定某一种 LLM 产品形态。

### 2. Host 与 Forge 职责分离

```text
┌──────────────────────────────────────────┐
│  Host（你的产品）                          │
│  LLM loop · TUI/Web · 审批 UI · 记忆产品   │
│  实现 ToolSession：plan / 用户输入 / token │
└────────────────────┬─────────────────────┘
                     │ call_tool(name, args)
┌────────────────────▼─────────────────────┐
│  Forge                                    │
│  沙箱 · execpolicy · patch · MCP · 插件   │
│  返回 ToolResult { content, is_error }    │
└──────────────────────────────────────────┘
```

Forge 发出**结构化请求**（「请批准这条命令」「请回答这个问题」）；Host 决定如何展示、阻塞或自动批准。Forge 内置 UI 是非目标。

### 3. 一套工具面，多种部署形态

同一组 32 个工具名与参数，可在三种模式下运行：

| 模式 | API | 适用 |
|------|-----|------|
| **进程内** | `ForgeEnvironment.runtime()` | 单进程 Agent、单元测试、快速原型 |
| **Rust 高性能路径** | `HybridForgeRuntime`（默认） | maturin 构建后的生产默认 |
| **Actor 隔离** | `ToolWorkerActor` | 子进程 / 集群 worker，与 Pulsing gossip 共享 |

Host 换部署方式时，**LLM 侧 tool schema 不必改**。

### 4. Rust 核心 + Python 生态

执行热路径（shell、patch、文件、审批门、MCP client）在 Rust crate `pulsing-forge` 中实现；Python 提供 Actor 绑定、Code Mode、Extension 与 fallback。默认 **Hybrid dispatch**：Rust 优先，暂无 Rust handler 的工具自动走 Python——保证 **32 工具开箱 callable**。

### 5. 可验证的质量，而非口号

Forge 在 CI 中持续验证工具注册、默认可调用性与核心集成场景（`test_hybrid_forge_callable.py`、`test_pulsing_forge.py` 等）。对外说明能力边界时以测试结果为准，缺口诚实公开。

---

## Forge 是什么 / 不是什么

| ✅ Forge 是 | ❌ Forge 不是 |
|-------------|--------------|
| Agent 工具与环境运行时（库） | 完整 Agent OS 或 Chat 产品 |
| 可沙箱化的 workspace 执行层 | 容器编排（K8s 层由部署解决） |
| 主流 coding agent 工具面 + Claude 互操作 helper | 某单一 CLI / TUI 的复刻 |
| MCP runtime + 插件安装骨架 | MCP 目录 UI 或 OAuth 登录产品 |
| 与 Pulsing Actor 可选集成 | 强依赖分布式集群才能用 |

---

## 架构一览

```text
Host 创建 ForgeEnvironment(cwd, sandbox_policy, session)
         │
         ▼
  HybridForgeRuntime          ← maturin 构建后默认
    ├─ Rust ForgeRuntime      ← 22 handler + MCP runtime
    └─ Python LocalToolRuntime ← 10 个 Host-only 工具 fallback
         │
         ▼
  Handlers（按域）
    Execution · Filesystem · Session · Discovery · MCP · Code Mode · Extension
         │
         ▼
  ToolResult → Host（或 Craft.on_forge_event 事件流）
```

**三个核心类型**（详见 [abstractions.md](./abstractions.md)）：

- **`ForgeEnvironment`** — 工作区根目录、沙箱策略、会话钩子
- **`ToolSession`** — Host 实现的产品能力（plan、用户输入、审批）
- **`ToolResult`** — 统一返回 `{ content, is_error, structured? }`

---

## 工具能力全景（32 个）

Forge 注册 **32 个标准工具**，覆盖主流 coding agent 主路径：

### 隔离执行（11）— 适合 Actor worker

在沙箱进程里跑，不污染 Host 状态：

`Read` · `Glob` · `Grep` · `Edit` · `Write` · `Bash` · `shell_command` · `exec_command` · `write_stdin` · `apply_patch` · `view_image`

### Host 协作（21）— 需要 UI 或全局状态

| 域 | 工具 |
|----|------|
| **Session** | `update_plan` · `new_context` · `get_context_remaining` · `request_user_input` · `request_permissions` |
| **Discovery / 插件** | `tool_search` · `list_available_plugins_to_install` · `request_plugin_install` |
| **MCP** | `list_mcp_resources` · `list_mcp_resource_templates` · `read_mcp_resource` |
| **Code Mode** | `exec` · `wait` |
| **Extension** | `web.run` · `skills.list` · `skills.read` · `memories.*`（4）· `web_search` |

另：**Forge 额外**提供 `Read`/`Glob`/…/`Bash`（Claude 互操作 alias），与 Session 域工具互补。

完整参数见 [包内 README](https://github.com/DeepLink-org/pulsing/blob/main/python/pulsing/forge/README.md)。

---

## 开箱即用：Hybrid + MCP

**2026-05 现状**（对外可宣传的能力）：

1. **`maturin develop` / 正式 wheel 安装后**，`ForgeEnvironment().runtime()` 默认走 `HybridForgeRuntime`
2. **32 工具均可调用**，不会出现 `Unknown tool`（CI：`test_hybrid_forge_callable.py`）
3. **MCP runtime 默认启动**：`list_mcp_resources` 等 wired；无配置 server 时返回空列表，而非 runtime 未初始化
4. **Craft 已接入**：Host 路径 Hybrid、插件安装后 `refresh_mcp()`、隔离 worker 同套工具面

```python
from pulsing.forge import ForgeEnvironment

env = ForgeEnvironment.ephemeral(cwd="/path/to/repo")  # 本地 dev 默认 auto_approve
rt = env.runtime()

rt.call_tool("shell_command", {
    "command": "pytest -q",
    "workdir": "/path/to/repo",
    "timeout_ms": 60_000,
})
rt.call_tool("list_mcp_resources", {})
rt.call_tool("memories.list", {})
```

---

## 成熟度与边界

**当前（2026-05）**

| 维度 | 状态 |
|------|------|
| 32 工具 callable | ✅ Hybrid dispatch + Python fallback |
| MCP runtime | ✅ wired；无 server 时返回空列表 |
| Actor 隔离 worker | ✅ `ToolWorkerActor` + `ForgeBackend` |
| Code Mode L2 | 🟡 控制面 Actor 已接，后台续跑推进中 |
| MCP 动态进 LLM | 🟡 Hub + sync 已接，产品侧持续完善 |

```bash
pytest tests/python/test_hybrid_forge_callable.py tests/python/test_pulsing_forge.py -q
```

### 已知边界（发版说明级）

| 已就绪 | 仍在推进 |
|--------|----------|
| 32 工具 callable、Hybrid dispatch | Code Mode 后台 cell、yield 续跑 |
| MCP runtime live、refresh API | 动态 MCP 工具自动进 LLM schema |
| execpolicy + 审批门（Rust） | 更完整的 policy 与 sandbox 集成 |
| memories Extension 本地 L2 | web.run / skills authority、hosted search |
| Pulsing Actor 隔离 worker | Craft 与 Forge 集成测试持续加强 |

---

## 适用场景

### 适合用 Forge

- 你在做 **coding agent / dev agent**，需要稳定的 shell + 文件 + patch 工具层
- 你想 **标准 coding agent 工具面**，又不想绑定某一家的 Chat 产品壳
- 你需要 **同一套工具** 在本地进程与隔离 worker 之间切换
- 你用 **Pulsing** 做分布式 Agent，需要 gossip 共享的 tool worker
- 你在评估 **Craft** 或想参考其 Multi-Agent 架构（Craft 是 Forge 的参考 Host）

### 可能不需要 Forge

- 只要简单 `subprocess.run` 一两个命令（直接写即可）
- 需要完整开箱 Chat 产品（看 Craft 或自建 Host + Forge）
- 只要 MCP server 开发、不要 agent 工具层（Forge 范围过大）

---

## 5 分钟上手

### 安装

```bash
pip install pulsing
# 从源码开发（Rust 路径 + Hybrid 默认）：
maturin develop   # 或 uv run maturin develop
```

### 最小示例

```python
from pulsing.forge import ForgeEnvironment, LocalToolSession

env = ForgeEnvironment(
    cwd=".",
    session=LocalToolSession(token_budget=128_000),
)
rt = env.runtime()

# 读文件
print(rt.call_tool("Read", {"file_path": "README.md"}))

# 跑测试
print(rt.call_tool("shell_command", {
    "command": "pytest tests/python/test_pulsing_forge.py -q",
    "workdir": ".",
    "timeout_ms": 120_000,
}))
```

### 接入自研 Agent loop

```python
# 伪代码：把 Forge 当 tool backend
FORGE_TOOLS = env.runtime().tool_names()  # 32 个

async def on_llm_tool_call(name: str, arguments: dict):
    result = env.runtime().call_tool(name, arguments)
    return result.content
```

### 接入 Pulsing 隔离 worker

```python
import pulsing as pul
from pulsing.forge import ToolWorkerActor, ToolWorkerConfig

await pul.init()
worker = await ToolWorkerActor.spawn(config=ToolWorkerConfig(cwd="."))
out = await worker.Read(file_path="README.md")
await pul.shutdown()
```

---

## 与 Pulsing 生态的关系

```text
Pulsing          分布式 Actor 运行时（集群、gossip、隔离 spawn）
    │
Pulsing Forge    Agent 工具与环境（import: pulsing.forge）  ← 本文
    │
Craft            Multi-Agent 参考应用（pulsing.craft，Forge 的参考 Host）
```

- **只用 Forge**：`ForgeEnvironment` + `runtime()`，无需启动集群
- **Forge + Pulsing**：`ToolWorkerActor` 隔离执行，事件经 `tell_forge_event` 回 Host
- **Forge + Craft**：开箱 Multi-Agent、审批 UI、LLM schema（Craft 侧 schema 仍在补全）

Forge 可独立嵌入；Pulsing 与 Craft 是「把 Forge 跑满」的推荐路径，不是硬性依赖。

---

## 与其他方案的比较

| 维度 | 自研 subprocess | LangChain Tools | 闭源 Agent CLI | **Pulsing Forge** |
|------|-----------------|-----------------|----------------|-------------------|
| 工具面统一 | ❌ 各自实现 | 框架绑定 | 产品内封闭 | ✅ 32 工具标准面 |
| 沙箱 / execpolicy | 自行维护 | 通常无 | 通常成熟 | 🟡 Rust 门控 + 持续完善 |
| MCP | 自行接 | 视集成而定 | 视产品而定 | ✅ runtime wired |
| 与 Host 解耦 | ✅ | 部分 | ❌ 产品绑定 | ✅ ToolSession 边界 |
| 隔离 / 集群执行 | 自行造 | 通常无 | 单机为主 | ✅ Actor worker |
| 开源可嵌入 | ✅ | ✅ | ❌ | ✅ |

Forge 的定位：**开源、可嵌入、基于 Pulsing Actor** 的 agent 执行层——介于「自己拼 shell」与「绑定完整闭源产品」之间。

---

## 路线图（高层）

1. **MCP 动态注册** — 插件安装 → server 拉起 → 工具进 LLM catalog
2. **Code Mode L2** — 后台 cell、yield 续跑（Python / Actor 控制面）
3. **Rust handler 覆盖** — execpolicy、unified_exec、Extension 深化
4. **Forge × Pulsing** — 事件 Queue、多节点 worker（见 [Pulsing 集成](pulsing-integration.zh.md)）

---

## 文档导航

| 文档 | 适合谁 |
|------|--------|
| **本文** | 对外介绍、选型、推广 |
| [快速开始](getting-started.zh.md) | 安装与首个 `call_tool` |
| [核心概念](concepts.zh.md) | Host / Forge 心智模型 |
| [抽象模型](abstractions.zh.md) | Environment、Session、工具域 |
| [工具清单（32）](tools.zh.md) | 按域划分的工具表 |
| [Pulsing 部署](deployment.zh.md) | ForgeBackend、Actor 拓扑 |
| [Pulsing 集成](pulsing-integration.zh.md) | ask/tell、能力矩阵 |
| [包内 README](https://github.com/DeepLink-org/pulsing/blob/main/python/pulsing/forge/README.md) | API 速查 |
| [Craft 一体化](../design/forge/craft-architecture.zh.md) | 架构 Review |
| [工程说明](../design/forge/engineering.zh.md) | crate 与实现细节 |

---

## 许可与致谢

Apache-2.0。Rust 核心 crate：`pulsing-forge`。部分实现参考了业界开源 agent-tool 实践；第三方声明见 `crates/pulsing-forge/NOTICE`。

**Try it:**

```bash
pip install pulsing && python -c "
from pulsing.forge import ForgeEnvironment
print(ForgeEnvironment.ephemeral().runtime().call_tool('Glob', {'pattern': '*.md', 'path': '.'}))
"
```

有问题或集成需求，欢迎从 [快速开始](getting-started.zh.md) 入手，或查阅 `tests/python/test_hybrid_forge_callable.py` 了解 32 工具 smoke 覆盖。
