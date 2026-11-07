# CLI 运维

## 启动 actor

指定实现 actor API 的、可导入的用户类：

```bash
pulsing actor my_app.Counter --addr 0.0.0.0:8000 --name counter -- --initial 10
```

`--` 之前的参数配置节点与 actor 注册，之后的参数传给 actor 构造函数。
通过 `--seeds 127.0.0.1:8000` 加入已有集群。

## 观察集群

Inspect 使用 HTTP 观察者模式，不加入集群。

```bash
pulsing inspect cluster --seeds 127.0.0.1:8000
pulsing inspect actors --endpoint 127.0.0.1:8000 --detailed
pulsing inspect actors --seeds 127.0.0.1:8000 --json
pulsing inspect metrics --seeds 127.0.0.1:8000
pulsing inspect watch --seeds 127.0.0.1:8000 --interval 2
```

`--timeout` 限制请求时间；`--best_effort` 允许节点不可用时继续。
`pulsing actor list` 已替换为 `pulsing inspect actors`。

## Workspace 文件管理

```bash
pulsing init --template minimal
pulsing checkpoint -m "working state"
pulsing history
pulsing rollback
```

Rust CLI 可通过 `pulsing run [SCRIPT]` 执行一次 Python workflow。
项目不再内置推理服务、benchmark 命令或 agent 交互会话。
