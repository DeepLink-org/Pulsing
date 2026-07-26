# Pulsing

[![CI](https://github.com/DeepLink-org/pulsing/actions/workflows/ci.yml/badge.svg)](https://github.com/DeepLink-org/pulsing/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)

**[English](README.md)**

**分布式 AI 系统的通信骨干。**

**Actor 运行时。流式优先。零依赖。内置发现。**

Pulsing 是一个用 Rust 构建、为 Python 设计的分布式 Actor 运行时。跨机器连接 AI Agent 和服务——不需要 Redis，不需要 etcd，不需要 YAML。只需 `pip install pulsing`。

🚀 **零外部依赖** — 纯 Rust + Tokio，无需 NATS/etcd/Redis

⚡ **流式优先** — 原生流式响应支持，为 LLM token 生成而设计

🌐 **内置发现** — SWIM/Gossip 协议实现自动集群管理

🔀 **统一 API** — 本地和远程 Actor 使用相同的 `await actor.method()`

## 🚀 5分钟快速体验

### 安装

```bash
pip install pulsing
```

### 第一个 Multi-Agent 应用

```python
import asyncio
import pulsing as pul
from pulsing.agent import runtime

@pul.remote
class Greeter:
    def __init__(self, display_name: str):
        self.display_name = display_name

    def greet(self, message: str) -> str:
        return f"[{self.display_name}] 收到: {message}"

    async def chat_with(self, peer_name: str, message: str) -> str:
        # 使用 Greeter.resolve() 获取有类型的代理
        peer = await Greeter.resolve(peer_name)
        return await peer.greet(f"来自 {self.display_name}: {message}")

async def main():
    async with runtime():
        # 创建两个 Agent
        alice = await Greeter.spawn(display_name="Alice", name="alice")
        bob = await Greeter.spawn(display_name="Bob", name="bob")

        # Agent 间对话
        reply = await alice.chat_with("bob", "你好！")
        print(reply)  # [Bob] 收到: 来自 Alice: 你好！

asyncio.run(main())
```

**就这么简单！** `@pul.remote` 让普通类变成可分布式部署的 Actor，`Greeter.resolve()` 让 Agent 互相发现和通信。

## 💡 我想做...

| 场景 | 示例 | 说明 |
|------|------|------|
| **快速体验** | `examples/quickstart/` | 10 行代码入门 |
| **Multi-Agent 协作** | `examples/agent/pulsing/` | AI 辩论、头脑风暴、角色扮演 |
| **分布式 LLM 推理** | `pulsing actor router/vllm` | GPU 集群推理服务 |
| **集成 AutoGen** | `examples/agent/autogen/` | 一行代码分布式 |
| **集成 LangGraph** | `examples/agent/langgraph/` | 计算图跨节点执行 |
| **Agent 工作区 CLI** | `pulsing agent init` | [Pulsing Agent](examples/agent/workspace-demo.md) — 在仓库内运行多 Agent |
| **Agent 工具与环境** | `examples/python/forge_minimal.py` | [Pulsing Forge](docs/src/forge/index.zh.md) — 沙箱 shell、文件、plan |

## 🔨 Pulsing Forge

**面向 AI Agent 的通用工具与环境运行时** — 在可配置沙箱里执行 shell、改文件、维护计划；可嵌入任意 Agent 框架，也可通过 Pulsing Actor 隔离部署。

```python
from pulsing.forge import ForgeEnvironment

env = ForgeEnvironment(cwd=".")
env.runtime().call_tool("shell_command", {"cmd": "pytest -q", "workdir": "."})
```

文档：[Forge 章节](docs/src/forge/index.zh.md) · 包 README：[python/pulsing/forge/README.md](python/pulsing/forge/README.md)

## 🤖 Pulsing Agent

**工作区级多 Agent SDK + CLI** — 初始化 `.pulsing/` 工作区，在集群上唤醒 Agent，配合 Forge 工具协作。

```bash
pip install pulsing[agent]
pulsing agent init
pulsing agent wake --agents guide
pulsing agent say guide "运行 pytest"
```

文档：[工作区示例](examples/agent/workspace-demo.md) · SDK：`from pulsing.agent import Agent, spawn_agent`

## 🎯 核心能力

### 1. Multi-Agent 协作

多个 AI Agent 并行工作、互相通信：

```python
from pulsing.agent import agent, runtime, llm

@agent(role="研究员", goal="深入分析问题")
class Researcher:
    async def analyze(self, topic: str) -> str:
        client = await llm()
        return await client.ainvoke(f"分析: {topic}")

@agent(role="评审", goal="评估方案质量")
class Reviewer:
    async def review(self, proposal: str) -> str:
        client = await llm()
        return await client.ainvoke(f"评审: {proposal}")

async with runtime():
    researcher = await Researcher.spawn(name="researcher")
    reviewer = await Reviewer.spawn(name="reviewer")

    # 并行工作，互相协作
    analysis = await researcher.analyze("AI 发展趋势")
    feedback = await reviewer.review(analysis)
```

```bash
# 运行 MBTI 人格讨论示例
python examples/agent/pulsing/mbti_discussion.py --mock --group-size 6

# 运行并行创意生成示例
python examples/agent/pulsing/parallel_ideas_async.py --mock --n-ideas 5
```

### 2. 一行代码分布式

本地开发，无缝扩展到集群：

```python
# 单机模式（开发调试）
async with runtime():
    agent = await MyAgent.spawn(name="agent")

# 分布式模式（生产部署）—— 只需加地址
async with runtime(addr="0.0.0.0:8001"):
    agent = await MyAgent.spawn(name="agent")

# 其他节点自动发现
async with runtime(addr="0.0.0.0:8002", seeds=["node1:8001"]):
    agent = await resolve("agent")  # 跨节点透明调用
```

### 3. LLM 推理服务

开箱即用的 GPU 集群推理：

```bash
# 启动 Router（OpenAI 兼容 API）
pulsing actor pulsing.serving.Router --addr 0.0.0.0:8000 --http_port 8080 --model_name my-llm

# 启动 vLLM Worker（可多个）
pulsing actor pulsing.serving.VllmWorker --model Qwen/Qwen2.5-0.5B --addr 0.0.0.0:8002 --seeds 127.0.0.1:8000

# 测试
curl http://localhost:8080/v1/chat/completions \
  -d '{"model": "my-llm", "messages": [{"role": "user", "content": "Hello"}]}'
```

### 4. Agent 框架集成

已有 AutoGen/LangGraph 代码？一行迁移：

```python
# AutoGen: 替换运行时
from pulsing.autogen import PulsingRuntime
runtime = PulsingRuntime(addr="0.0.0.0:8000")

# LangGraph: 包装计算图
from pulsing.langgraph import with_pulsing
distributed_app = with_pulsing(app, seeds=["gpu-server:8001"])
```

### 5. TensorDict 快速传输

`TensorMessage` 用一段不透明 metadata 和多段连续 CPU buffer 传输 TensorDict。PulsingQueue
负责把 CUDA Tensor 搬到 CPU，并在交给 Pulsing 前保证每个 Tensor 连续；Pulsing 不解析
Tensor 的 dtype、shape 或 stride，这些信息由 PulsingQueue 编码到 metadata。

```python
from array import array

import pulsing as pul

cpu_buffer = array("f", range(6))
message = pul.TensorMessage(
    metadata=b"...",              # dtype / shape / TensorDict 结构
    buffers=[memoryview(cpu_buffer).cast("B")],  # 必须连续
    version=1,
)

# @remote Actor 收到 TensorMessage 时调用该专用入口。
@pul.remote
class Storage:
    async def receive_tensor(self, message: pul.TensorMessage):
        return message
```

明文远程连接默认使用同一监听端口上的 raw TCP 数据面：长连接池复用连接，发送端用
`write_vectored` 直接提交 header、metadata 和各原始 buffer，接收端把每个 payload
直接读入最终由 Python Tensor 持有的分配。按应用 payload 的 CPU copy 口径，主路径只有
`应用 buffer -> TCP kernel buffer` 和 `TCP kernel buffer -> 最终接收 buffer` 两次，
不会先拼成一个大包，也不会在接收后再复制一次。远程调用的 `ask`/`tell` 完成前，发送方
不得修改或 resize 源 buffer；完成后远端已经持有独立接收分配。若目标 actor 位于同一进程，
双方共享原 storage，接收方仍持有引用期间原地修改会彼此可见；需要快照语义时由调用方先
`clone()`。返回的接收 buffer 生命周期由 `TensorMessage`/下游 Tensor 引用管理。

TLS 连接，或设置 `PULSING_TENSOR_TRANSPORT=http2` 时，使用兼容 HTTP/2 路径。该路径会
打包和聚合 payload，copy 次数高于 raw TCP。可通过 `pulsing.tensor_transport_stats()`
检查 `active_copy_model`、raw frame/byte 计数和 HTTP/2 fallback 计数。接收边界可用以下
环境变量限制，超限会在分配大 payload 前拒绝：

- `PULSING_MAX_TENSOR_WIRE_BYTES`：单帧总大小，默认 64 GiB
- `PULSING_MAX_TENSOR_METADATA_BYTES`：metadata 大小，默认 64 MiB
- `PULSING_MAX_TENSOR_BUFFERS`：buffer 数量，默认 65536

同节点共享内存后端已在 `TensorTransport` 抽象中预留，当前尚未实现。

完整内容见 [TensorMessage 快速传输设计](docs/src/design/tensor-message-transport.zh.md)
和 [可运行的 TCP 示例](examples/python/tensor_message_fast_path.py)。

## 📚 示例导航

```
examples/
├── quickstart/              # ⭐ 5分钟入门
│   └── hello_agent.py       #    第一个 Agent
├── agent/
│   ├── pulsing/             # ⭐⭐ Multi-Agent 应用
│   │   ├── mbti_discussion.py      # MBTI 人格讨论
│   │   └── parallel_ideas_async.py # 并行创意生成
│   ├── autogen/             # AutoGen 集成
│   └── langgraph/           # LangGraph 集成
├── python/                  # ⭐⭐ 基础示例
│   ├── ping_pong.py         #    Actor 基础
│   ├── cluster.py           #    集群通信
│   ├── tensor_message_fast_path.py # Tensor 传输
│   └── ...
└── rust/                    # Rust 示例
```

## 🔧 技术特性

- **零外部依赖**：纯 Rust + Tokio，无需 NATS/etcd/Redis
- **Gossip 协议**：内置 SWIM 协议实现节点发现和故障检测
- **位置透明**：本地和远程 Actor 使用相同的 API
- **流式消息**：原生支持流式请求和响应（适配 LLM）
- **类型安全**：Rust Behavior API 提供编译时消息类型检查

## 📦 项目结构

```
Pulsing/
├── crates/                   # Rust 核心
│   ├── pulsing-actor/        #   Actor System
│   └── pulsing-py/           #   Python 绑定
├── python/pulsing/           # Python 包
│   ├── actor/                #   Actor API
│   ├── agent/                #   Agent 工具箱
│   ├── autogen/              #   AutoGen 集成
│   └── langgraph/            #   LangGraph 集成
├── examples/                 # 示例代码
└── docs/                     # 文档
```

## 🛠️ 开发

### 前置依赖

- [Rust](https://rustup.rs/) ≥ 1.75
- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/)（推荐的包管理器）
- [just](https://github.com/casey/just)（任务运行器：`cargo install just` 或 `brew install just`）

### 快速搭建

```bash
# 1. 安装 Python 依赖
uv sync --extra dev

# 2. 编译 Rust 核心并安装（修改 Rust 代码后需重新执行）
uv run maturin develop
```

### 常用命令

```bash
just dev          # 编译并安装（开发模式）
just test         # 运行全部测试（Rust + Python）
just test-python  # 仅运行 Python 测试
just fmt          # 格式化代码
just lint         # 代码检查
just check        # 提交前完整检查（格式 + lint + 测试）
just cov          # 生成覆盖率报告
```

详细开发指南请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 License

Apache-2.0
