# Forge Session REPL

> **状态**：MVP 已实现 · **入口**：`pulsing forge repl`（优先 Rust `pulsing-forge-repl`，否则 Python fallback）
> **关联**：[craft-architecture.md](./craft-architecture.md)

## 快速开始

```bash
pulsing forge repl --cwd .
pulsing forge repl --trace tests/fixtures/forge_traces/read_sample.jsonl --replay-all --verify
pulsing forge repl --record /tmp/session.jsonl
```

Nushell 风格：`Read --file_path README.md` · `/tools` · `/approve ask` · Tab 补全 · 行内灰色 hint

**Shell UX（Rust `pulsing-forge-repl`）**

| 能力 | 说明 |
|------|------|
| Tab 补全 | `/` slash（含 alias）、`@` 文件 mention、meta、工具名、`--flag` |
| Slash 内置 | `/ps` `/stop`（`/clean`） `/diff` `/review` `/fork N` `/compact` `/mcp` `/plugins` `/rollout` `/copy` `/clear` |
| 常用别名 | `/status`→session · `/permissions`/`/approve`→审批 · `/clean`→stop · `/exit`→quit |
| Mention | `@path` 或 `/mention path` → Read 预览 |
| 审批 | `/permissions auto\|ask` 接通 ReplToolSession |

---

## Rust-first vs Python fallback

| | **Rust（经 `pulsing forge repl`）** | **Python（`--python` 或 fallback）** |
|--|-------------------|----------------------------------|
| 主循环 / 终端 | `reedline`（Tab 补全 + hint + ColumnarMenu）+ `comfy-table` | 裸 `input()` |
| 工具 runtime | `ToolRuntime`（22 个 Rust handler） | `LocalToolRuntime`（32 工具全可用） |
| 依赖 | 无 Python 环境冲突 | 需项目 venv / maturin |
| 适用 | 日常调试、CI replay、终端体验 | Extension / exec / Code Mode 等 Python-only 工具 |

**方向**：Shell 主循环与渲染放 Rust（类似 Nushell 思路）；Python REPL 保留作 hybrid dispatch 完成前的 fallback 与全工具覆盖。

---

## 结论

**值得做。** 在 Forge 层提供 **Session REPL**，直接绑定 `ToolSession` + `ToolRuntime`（而非 Craft 全量 Agent loop），同时服务：

1. **Replay** — 按序重放某次 Agent 轨迹里的 tool call + forge event，用于调试与回归
2. **Manual drive** — 人工逐步调用工具、改 session 状态、模拟审批，干预或替代 LLM 决策

Forge REPL 是 **headless、可脚本化** 的 session 控制台（类似 gdb/lldb 对 runtime，而不是聊天 UI）。

---

## 放哪一层

| 层 | 职责 |
|----|------|
| **Forge REPL** | 绑定 `ToolCallContext` / `ToolSession`；执行 `call_tool`；记录/回放 **trace**；Host 回调可插拔（自动审批 / 交互式 y/n） |
| **Craft** | 从 `CraftAgent` **导出 trace**；可选「调试暂停 → 附着 Forge REPL」；不负责重复实现 shell/patch |
| **Pulsing** | 可选：REPL 附着远程 `ToolWorkerActor`（同一 trace 格式） |

```
Agent 轨迹 (Craft)          Forge Session REPL
     │                              │
     ├─ export trace.jsonl ────────►│ replay / fork
     │                              │
     └─ pause & attach ─────────────►│ manual :call / :approve
                                    │
                            ToolRuntime + Session
                            (Local / Rust / Worker RPC)
```

---

## Trace 格式（单一事实源）

每条记录一行 JSON（JSONL），便于 diff 与 CI：

```json
{"seq": 1, "kind": "tool_call", "tool": "Read", "arguments": {"file_path": "a.py"}, "result": {"content": "...", "is_error": false}}
{"seq": 2, "kind": "forge_event", "event": {"kind": "tool_begin", "source": "Read", ...}}
{"seq": 3, "kind": "tool_call", "tool": "shell_command", "arguments": {...}, "result": {...}}
{"seq": 4, "kind": "session", "plan": [...], "tokens_remaining": 120000}
```

**来源**：

- Craft：`tool_host.call_tool` 前后 + `_forge_events` 合并导出
- 未来：第三方会话导出适配器（仅 tool 层，不含 LLM messages）

**Replay 语义**：

| 模式 | 行为 |
|------|------|
| `--dry-run` | 只打印将执行的 call，不改 workspace |
| `--verify` | 重跑并对比 result / event（集成回归） |
| `--from N` | 从第 N 步继续 |
| `--fork N` | 加载 0..N 步 session 快照，之后进入 **interactive**（人工干预） |

---

## REPL 能力（MVP → 完整）

### MVP

- 启动：`pulsing forge repl [--cwd] [--sandbox] [--trace file.jsonl]`
- 命令：
  - `call <tool> <json>` — 调 `runtime.call_tool`
  - `replay [step]` — 逐步重放 trace
  - `plan` / `session` — 看 `ToolSession` 状态
  - `events` — 最近 `ForgeEvent`
  - `approve auto|ask` — 切换 exec/permissions 回调
- 实现：`ForgeReplSession` 包装 `ForgeHostLink` 或 `LocalToolRuntime` + `LocalToolSession`

### 二期

- 附着 **UnifiedExec** / **Code Mode cell**（`:exec attach <session_id>`）
- 附着 **CraftAgent**（debug 模式，暂停 LLM turn）
- Worker RPC 模式（REPL 远程调 `ToolWorkerActor`）
- `replay --verify` 接入 CI（集成回归 fixture）

---

## 回归与验证

| 场景 | REPL 作用 |
|------|-----------|
| 集成回归 | `replay --verify trace/fixtures/*.jsonl` 证明端到端 |
| 人工 sign-off | 专家用 REPL 逐步验证工具行为 |
| 故障复现 | Agent 失败轨迹导出 → 最小复现 → 修 Forge → replay 绿 |

---

## 不做的事

- 不做 Craft 完整对话 TUI（已有/将有的产品 UI 层）
- 不在 REPL 里跑 LLM turn（只驱动 **Forge 工具面**）
- 不默认做 filesystem 时间旅行（replay 默认在同一 cwd；需要时可接 snapshot 插件）

---

## 建议落地顺序

1. **`ForgeReplSession` + JSONL trace 读写**（Forge 内，`python/pulsing/forge/repl/`）
2. **Craft `trace export`**（从现有 `_forge_events` + tool loop 钩子）
3. **`replay --verify`** + 1–2 条 fixture（patch、shell 审批）
4. CLI 入口 + 文档

**前置依赖**：Hybrid dispatch（Rust 路径下 Host 工具可调用），否则 replay 在 maturin 默认构建下会卡在 Python-only 工具。

---

## 维护

改 REPL / trace 格式时同步更新 Craft export 与 `tests/fixtures/forge_traces/`（待建）。
