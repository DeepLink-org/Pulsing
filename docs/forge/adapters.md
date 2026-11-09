# Runtime Adapters — 两层运行时

Forge 场景里的店员由 adapter 决定"谁来回信"：

| adapter | 层 | 依赖 | 用途 |
|---|---|---|---|
| `builtin`(当前默认) | 确定性层 | 无 LLM | CI、回归测试、场景校验 — 同输入必同输出 |
| `codex`(tide) | 潮汐层 | pulsing-mcp + 模型端点 | 真实多 agent 回路(压测验证栈) |
| `script` | 确定性层 | 任意可执行 | 用户自带逻辑: 每回合以唤醒信为 stdin 调用, stdout 每行一封信(JSON) |
| `claude-code` / `mcp` | 潮汐层 | 各自 CLI | 同一 4 工具 MCP 契约 |

## 契约(冻结)

agent 侧接口永远只有 4 个 MCP 工具:
`pulsing_send(to, message, silent)` · `pulsing_broadcast(to_list, message, silent)` ·
`pulsing_drain_mail()` · `pulsing_health()`

纪律(ACL/自回声熔断/广播收拢/自动静默)全部在工具层实现, adapter 与模型无关。
