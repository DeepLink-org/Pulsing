# 术语与风格

| 符号 | 含义 |
|------|------|
| `ActorSystem` | 管理 actor 与集群连接的运行时 |
| `Actor` | 消息处理接口 |
| `ActorRef` | 低层消息引用 |
| `ActorProxy` | `@remote` 返回的 Python 方法调用代理 |

CLI 示例使用用户定义的类路径：

```bash
pulsing actor my_app.Counter --addr 0.0.0.0:8000 --name counter -- --initial 10
pulsing inspect actors --endpoint 127.0.0.1:8000
```

`--` 之后的参数属于 actor 构造函数。Inspect 只观察集群，不加入集群。
符号与路径使用反引号，Python 遵循 PEP 8，Rust 使用 `cargo fmt`，
中文文档使用 `.zh.md` 后缀。
