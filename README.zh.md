# Pulsing

[English](README.md)

基于 Rust 构建、面向 Python 的分布式 actor 运行时。
提供位置透明的 actor 调用、流式消息和内置集群发现，不需要 Redis 或 etcd。

## 安装

```bash
pip install pulsing
```

## 第一个 actor

```python
import asyncio
import pulsing as pul


@pul.remote
class Counter:
    def __init__(self, value=0):
        self.value = value

    def inc(self):
        self.value += 1
        return self.value


async def main():
    await pul.init()
    try:
        counter = await Counter.spawn(value=0, name="counter")
        print(await counter.inc())  # 1
    finally:
        await pul.shutdown()


asyncio.run(main())
```

组建集群时，为节点配置 `addr`，通过 `seeds` 加入集群。
使用 `await Counter.resolve("counter")` 查找命名 actor。

## 项目范围

- Actor mailbox、生命周期、监督重启及命名发现。
- 本地与远程请求响应、单向调用及流式响应。
- HTTP/2、TLS 和不解释业务内容的 tensor buffer 传输。
- Python `@remote`、子进程集成、队列及发布订阅。
- CLI actor 启动与集群观察。
- Workspace 初始化、checkpoint、rollback 及 Python workflow。

项目不再内置推理服务器、LLM 客户端、Forge 工具与 coding-agent SDK、
桌面 GUI 和 benchmark 模块。业务服务可以自行构建在 actor API 之上。

## 命令行

```bash
pulsing actor my_app.Counter --addr 0.0.0.0:8000 --name counter
pulsing inspect actors --endpoint 127.0.0.1:8000
pulsing init --template minimal
pulsing checkpoint -m "working state"
pulsing history
pulsing rollback
```

Rust CLI 还支持 `pulsing run [script.py]`，别名为 `workflow`。
Workflow 执行一次后退出，不再进入 LLM 交互会话。

## 开发

需要 Python 3.10+ 和当前稳定版 Rust。RustPython CLI 要求 Rust 1.93 或更新版本。

```bash
uv sync --extra dev
uv run maturin develop
just test
just fmt
just lint
```

`just build-wheel` 构建 Python wheel，`just build-binary` 构建 RustPython CLI。
开发流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 目录

| 路径 | 职责 |
|------|------|
| `crates/pulsing-actor/` | Actor 运行时、发现与传输 |
| `crates/pulsing-py/` | CPython/PyO3 绑定 |
| `crates/pulsing-rpymod/` | RustPython 绑定 |
| `crates/pulsing-bindings-core/` | 绑定共用数据类型 |
| `crates/pulsing-cli/` | Rust CLI |
| `crates/pulsing-workspace/` | Workspace 初始化与历史 |
| `python/pulsing/` | Python SDK 与集成 |
| `examples/` | Actor 与传输示例 |
| `docs/` | 文档 |

## 许可证

Apache-2.0
