# Pulsing

[![CI](https://github.com/reiase/pulsing/actions/workflows/ci.yml/badge.svg)](https://github.com/reiase/pulsing/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)

Pulsing 是一个轻量级分布式 Actor 框架，专为 LLM 推理服务设计。基于 Rust 和 Tokio 构建，提供高性能的异步 Actor 模型，支持集群通信和服务发现。

## 特性

- **零外部依赖**：基于 Tokio 的纯 Rust 实现，无需 NATS/etcd 等外部服务
- **Gossip 协议发现**：内置 SWIM 协议实现节点发现和故障检测
- **位置透明**：ActorRef 支持本地和远程 Actor 的统一访问
- **流式消息**：原生支持流式请求和响应
- **Python 绑定**：通过 PyO3 提供完整的 Python API

## 快速上手

### 安装

```bash
# 从源码安装
pip install maturin
maturin develop

# 或使用 uv
uv pip install -e .
```

### Python 示例

```python
import asyncio
from pulsing.actor import Actor, Message, SystemConfig, create_actor_system

class EchoActor(Actor):
    async def receive(self, msg: Message) -> Message:
        return Message.single("echo", msg.payload)

async def main():
    # 创建 Actor System
    config = SystemConfig.standalone()
    system = await create_actor_system(config)

    # 生成 Actor
    actor_ref = await system.spawn(EchoActor(), "echo")

    # 发送消息
    response = await actor_ref.ask(Message.single("hello", b"world"))
    print(f"Response: {response.payload}")

asyncio.run(main())
```

### 集群通信

```python
# Node 1 - 启动 seed 节点
config = SystemConfig.with_addr("0.0.0.0:8000")
system = await create_actor_system(config)
await system.spawn(WorkerActor(), "worker", public=True)

# Node 2 - 加入集群
config = SystemConfig.with_addr("0.0.0.0:8001").with_seeds(["192.168.1.1:8000"])
system = await create_actor_system(config)

# 查找远程 Actor
remote_ref = await system.find("worker")
response = await remote_ref.ask(Message.single("request", data))
```

## CLI 命令

Pulsing 提供基于 Actor System 的 LLM 推理服务：

```bash
# 启动 Router (OpenAI 兼容 API)
pulsing actor router --addr 0.0.0.0:8000 --http_port 8080 --model_name my-llm

# 启动 Transformers Worker
pulsing actor transformers --model gpt2 --addr 0.0.0.0:8001 --seeds 127.0.0.1:8000

# 启动 vLLM Worker
pulsing actor vllm --model Qwen/Qwen2.5-0.5B --addr 0.0.0.0:8002 --seeds 127.0.0.1:8000

# 启动 vLLM Worker (macOS Metal 支持)
pulsing actor vllm --model Qwen/Qwen3-0.6B --mlx_device gpu --metal_memory_fraction 0.8 --addr 0.0.0.0:8002 --seeds 127.0.0.1:8000

# 运行基准测试
pulsing bench --tokenizer_name gpt2 --url http://localhost:8080
```

### 测试推理服务

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "my-llm", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

## 项目结构

```
Pulsing/
├── crates/                   # Rust crates
│   ├── pulsing-actor/        # Actor System 核心库
│   └── pulsing-bench/        # 性能基准测试工具
├── bindings/                 # PyO3 绑定
│   ├── lib.rs                # Python 模块入口
│   ├── actor.rs              # Actor 绑定
│   └── python_executor.rs    # Python executor
├── python/pulsing/           # Python 包
│   ├── actor/                # Actor System Python API
│   ├── actors/               # LLM serving actors
│   └── cli/                  # 命令行工具
├── examples/                 # 示例代码
│   ├── rust/                 # Rust 示例
│   └── python/               # Python 示例
└── tests/                    # 测试
    ├── rust/                 # Rust 测试说明
    └── python/               # Python 测试
```

## 构建

### 开发构建

```bash
# 使用 maturin
maturin develop

# 或使用 uv
uv pip install -e .
```

### 发布构建

```bash
# 构建 wheel
maturin build --release

# 构建 manylinux 兼容 wheel
pip install maturin ziglang
maturin build --release --zig --compatibility manylinux_2_24
```

## 运行测试

```bash
# Python 测试
pytest tests/python/

# Rust 测试
cargo test --workspace

# 运行特定 crate 的测试
cargo test -p pulsing-actor
```

## 许可证

Apache-2.0
