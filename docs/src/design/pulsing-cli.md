# Pulsing CLI

Pulsing 的命令行聚焦 actor 运行时和 workspace 文件管理。

| 命令 | 职责 |
|------|------|
| `pulsing actor TYPE` | 启动用户提供的 actor |
| `pulsing inspect cluster/actors/metrics/watch` | 通过 HTTP 观察集群，不加入集群 |
| `pulsing examples [NAME]` | 浏览内置示例 |
| `pulsing init` | 创建 minimal workspace |
| `pulsing history` | 列出 checkpoint |
| `pulsing checkpoint` | 保存文件快照 |
| `pulsing rollback` | 恢复文件快照 |
| `pulsing run [SCRIPT]` | Rust CLI：执行一次 Python workflow |

Actor 启动参数使用 `--` 分隔：之前为节点和 actor 注册参数，
之后为用户 actor 的构造参数。例如：

```bash
pulsing actor my_app.Counter --addr 0.0.0.0:8000 --name counter -- --initial 10
pulsing inspect actors --endpoint 127.0.0.1:8000
```

Python 入口为 `pulsing.cli.__main__:main`。Rust CLI 通过 RustPython
运行 Python 命令，workspace 初始化和 journal 则使用 `pulsing-workspace`。

Workflow 执行成功后直接退出，不进入交互式 agent 会话。
推理压测、LLM 引导初始化、coding-agent 命令、工具 REPL 和桌面界面已移除。
