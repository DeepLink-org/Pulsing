# Pulsing 示例

示例聚焦 actor 通信、发现、生命周期和传输。

```bash
pip install pulsing
python examples/quickstart/hello_agent.py
```

| 路径 | 内容 |
|------|------|
| `quickstart/hello_agent.py` | 两个 actor 互相调用 |
| `quickstart/ai_chat_room.py` | 使用本地文本模板的多 actor 协作，无模型依赖 |
| `quickstart/function_to_fleet.py` | 多 worker 任务分发 |
| `python/ping_pong.py` | 基础通信 |
| `python/message_patterns.py` | RPC 与流式消息 |
| `python/named_actors.py` | 服务发现 |
| `python/cluster.py` | 多节点集群 |
| `python/tensor_message_fast_path.py` | 连续 CPU buffer 传输 |
| `python/subprocess_example.py` | 子进程兼容 API |
| `python/isolated_actor_spawn.py` | 隔离子进程 actor |
| `agent/pulsing/runtime_lifecycle_example.py` | 生命周期管理 |
| `agent/autogen/` | 可选 AutoGen 集成 |
| `agent/langgraph/` | 可选 LangGraph 集成 |
| `inspect/` | CLI 集群观察 |
| `rust/` | Rust actor 与 Behavior API |

Rust 示例：

```bash
cargo run --example ping_pong -p pulsing-actor
cargo run --example behavior_counter -p pulsing-actor
```
