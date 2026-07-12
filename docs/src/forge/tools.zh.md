# 工具清单（32）

Forge 注册 **32 个标准工具**，覆盖主流 coding agent 主路径。定义见 `python/pulsing/forge/integrated.py`。

---

## 按部署划分

| 分区 | 数量 | 运行位置 |
|------|------:|----------|
| **隔离** | 11 | `ToolWorkerActor`（子进程） |
| **Host** | 21 | `HybridForgeRuntime` / 进程内 |

两套工具名 **不相交**，由 `ForgeBackend` 自动路由。

---

## 隔离（11）

文件、shell、patch（沙箱边界）：

`Read` · `Glob` · `Grep` · `Edit` · `Write` · `Bash` · `shell_command` · `exec_command` · `write_stdin` · `apply_patch` · `view_image`

---

## Host（21）

| 域 | 工具 |
|----|------|
| **Session** | `update_plan`, `new_context`, `get_context_remaining`, `request_user_input`, `request_permissions` |
| **发现** | `tool_search`, `request_plugin_install`, `list_resources` |
| **MCP** | `list_mcp_resources`, `list_mcp_resource_templates`, `mcp__*`（动态） |
| **Code Mode** | `exec`, `wait` |
| **Extension** | `memories.*`, `skills.*`, `web.run`, `web_search`（共 8 个） |

Rust 暂无 handler 的 **10 个 Host 工具**由 Hybrid Python fallback 覆盖。

---

## Claude 互操作

隔离路径上的 `Read`、`Bash` 等 Claude 风格命名与标准工具面互补。

---

## 动态工具

- **插件 / tool_search** — 延迟注册，搜索后激活
- **MCP** — `refresh_mcp()` 后 live 工具经 Hub 同步进 LLM schema

---

## 验证

```bash
pytest tests/python/test_hybrid_forge_callable.py -q
pytest tests/python/craft/test_tools.py -q
pytest tests/python/test_pulsing_forge.py -q
```

---

## 深入

- [工程说明](../design/forge/engineering.zh.md)
- [包内 API 表](https://github.com/DeepLink-org/pulsing/blob/main/python/pulsing/forge/README.md)
