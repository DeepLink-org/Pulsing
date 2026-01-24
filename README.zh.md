# Pulsing

[![CI](https://github.com/DeepLink-org/pulsing/actions/workflows/ci.yml/badge.svg)](https://github.com/DeepLink-org/pulsing/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)

**[English](README.md)**

**Pulsing 是一个分布式 actor 框架，为构建分布式系统提供通信骨干，并为 AI 应用提供专门支持。**

🚀 **零外部依赖** — 纯 Rust + Tokio，无需 NATS/etcd/Redis

🌐 **自动发现** — 内置 Gossip 协议管理集群

🔀 **位置透明** — 本地和远程 Actor 使用相同 API

⚡ **流式支持** — 原生支持 LLM 流式响应

🤖 **Agent 友好** — 开箱即用集成 AutoGen、LangGraph

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
pulsing actor pulsing.actors.Router --addr 0.0.0.0:8000 --http_port 8080 --model_name my-llm

# 启动 vLLM Worker（可多个）
pulsing actor pulsing.actors.VllmWorker --model Qwen/Qwen2.5-0.5B --addr 0.0.0.0:8002 --seeds 127.0.0.1:8000

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

```bash
# 开发构建
maturin develop

# 运行测试
pytest tests/python/
cargo test --workspace
```

## 📄 License

Apache-2.0
