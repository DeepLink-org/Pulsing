# Pulsing 示例

## 🚀 从这里开始

```bash
# 安装
pip install pulsing

# 第一个示例（30秒）
python examples/quickstart/hello_agent.py
```

## 📚 示例导航

### ⭐ 快速入门 (`quickstart/`)

| 示例 | 说明 | 难度 |
|------|------|------|
| `hello_agent.py` | 第一个 Multi-Agent 应用 | ⭐ |
| `ai_chat_room.py` | 4个 AI 讨论话题 | ⭐ |

```bash
python examples/quickstart/hello_agent.py
python examples/quickstart/ai_chat_room.py --topic "AI 会取代人类吗？"
```

### ⭐⭐ Multi-Agent 应用 (`agent/pulsing/`)

| 示例 | 说明 | 特点 |
|------|------|------|
| `mbti_discussion.py` | MBTI 人格讨论投票 | 角色扮演、辩论、投票 |
| `parallel_ideas_async.py` | 并行创意生成 | 竞争提交、截止时间 |

```bash
# MBTI 讨论（推荐！）
python examples/agent/pulsing/mbti_discussion.py --mock --group-size 6 --rounds 2

# 并行创意
python examples/agent/pulsing/parallel_ideas_async.py --mock --n-ideas 5 --timeout 30
```

### ⭐⭐ 框架集成 (`agent/autogen/`, `agent/langgraph/`)

| 框架 | 示例 | 说明 |
|------|------|------|
| AutoGen | `agent/autogen/simple.py` | 单进程 |
| AutoGen | `agent/autogen/distributed.py` | 分布式 |
| LangGraph | `agent/langgraph/simple.py` | 单进程 |
| LangGraph | `agent/langgraph/distributed.py` | 分布式 |

```bash
# AutoGen
cd examples/agent/autogen && ./run_distributed.sh

# LangGraph
cd examples/agent/langgraph && ./run_distributed.sh
```

### ⭐⭐ Actor 基础 (`python/`)

| 示例 | 说明 |
|------|------|
| `ping_pong.py` | Actor 基础通信 |
| `message_patterns.py` | RPC、流式消息 |
| `named_actors.py` | 服务发现 |
| `cluster.py` | 集群通信 |
| `remote_actor_example.py` | @remote 装饰器 |

```bash
python examples/python/ping_pong.py
python examples/python/cluster.py --port 8000
```

### ⭐⭐ CLI 工具 (`inspect/`)

| 示例 | 说明 |
|------|------|
| `demo.sh` | 完整演示脚本（推荐） |
| `start_demo.sh` | 仅启动服务脚本 |
| `demo_service.py` | 多节点演示服务 |
| `README.md` | Inspect CLI 使用指南 |

**一键运行完整演示：**

```bash
./examples/inspect/demo.sh
```

这个脚本会自动：
- 启动 3 个节点
- 运行所有 inspect 子命令
- 展示各种用法和输出
- 按 Ctrl+C 自动清理

**或手动启动服务：**

```bash
# 启动服务（3个终端）
python examples/inspect/demo_service.py --port 8000
python examples/inspect/demo_service.py --port 8001 --seed 127.0.0.1:8000
python examples/inspect/demo_service.py --port 8002 --seed 127.0.0.1:8000

# 使用 inspect 查看集群状态
pulsing inspect cluster --seeds 127.0.0.1:8000
pulsing inspect actors --seeds 127.0.0.1:8000
pulsing inspect metrics --seeds 127.0.0.1:8000
pulsing inspect watch --seeds 127.0.0.1:8000
```

### ⭐⭐⭐ Rust 示例 (`rust/`)

| 示例 | 说明 |
|------|------|
| `ping_pong.rs` | 基础通信 |
| `behavior_counter.rs` | Behavior API |
| `behavior_fsm.rs` | 状态机 |
| `cluster.rs` | 集群 |

```bash
cargo run --example ping_pong -p pulsing-actor
cargo run --example behavior_counter -p pulsing-actor
cargo run --example behavior_fsm -p pulsing-actor
```

## 🎯 我想做...

| 场景 | 推荐示例 |
|------|----------|
| 先跑起来看看 | `quickstart/hello_agent.py` |
| 多 Agent 对话 | `quickstart/ai_chat_room.py` |
| AI 辩论/讨论 | `agent/pulsing/mbti_discussion.py` |
| 并行任务竞争 | `agent/pulsing/parallel_ideas_async.py` |
| 集群部署 | `python/cluster.py` |
| 学习 CLI 工具 | `inspect/demo_service.py` |
| 接入 AutoGen | `agent/autogen/` |
| 接入 LangGraph | `agent/langgraph/` |
| 学习 Rust API | `rust/behavior_*.rs` |

## 💡 小贴士

### 模拟模式 vs LLM 模式

大多数示例支持 `--mock` 参数：

```bash
# 模拟模式（无需 API Key，快速体验）
python examples/agent/pulsing/mbti_discussion.py --mock

# LLM 模式（需要 API Key）
export OPENAI_API_KEY=your_key
python examples/agent/pulsing/mbti_discussion.py --topic "AI 的未来"
```

### 分布式模式

```python
# 单机（默认）
async with runtime():
    ...

# 分布式
async with runtime(addr="0.0.0.0:8001"):
    ...

# 加入集群
async with runtime(addr="0.0.0.0:8002", seeds=["node1:8001"]):
    ...
```
