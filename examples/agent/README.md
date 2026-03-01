# Pulsing Agent Framework Integration

Pulsing 为主流 Agent 框架提供分布式运行时支持，让您的 Agent 应用轻松扩展到多机集群。

## 设计理念

**不替代，而是增强。**

- 用户的 Agent 代码几乎不需要改动
- API 与原框架完全兼容
- 一行代码即可获得分布式能力

## 支持的框架

| 框架 | 集成方式 | 说明 |
|------|----------|------|
| [AutoGen](autogen/) | `PulsingRuntime` | 替代默认运行时 |
| [LangGraph](langgraph/) | `with_pulsing()` | 包装编译后的图 |

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Agent Code                          │
│              (AutoGen / LangGraph / ...)                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  Pulsing Integration                        │
│          PulsingRuntime / with_pulsing()                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  Pulsing Actor System                       │
│     Gossip Protocol │ Location Transparency │ Streaming     │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  Node 1  │◄─►│  Node 2  │◄─►│  Node 3  │
    │   GPU    │   │   CPU    │   │   CPU    │
    └──────────┘   └──────────┘   └──────────┘
```

## 快速开始

### AutoGen

```python
from pulsing.integrations.autogen import PulsingRuntime

# 替代 SingleThreadedAgentRuntime
runtime = PulsingRuntime(addr="0.0.0.0:8000")
await runtime.start()

# API 完全一致
await runtime.register_factory("agent", lambda: MyAgent())
await runtime.send_message("Hello", AgentId("agent", "default"))
```

### LangGraph

```python
from pulsing.integrations.langgraph import with_pulsing

app = graph.compile()

# 一行代码实现分布式
distributed_app = with_pulsing(
    app,
    node_mapping={"llm": "langgraph_node_llm"},
    seeds=["gpu-server:8001"],
)

await distributed_app.ainvoke(input)
```

## 示例

```bash
# AutoGen
cd autogen && ./run_distributed.sh

# LangGraph
cd langgraph && ./run_distributed.sh
```

## 典型场景

1. **LLM + Tool 分离**
   - LLM 节点 → GPU 服务器
   - Tool 节点 → CPU 服务器

2. **多 Agent 协作**
   - 不同 Agent 运行在不同节点
   - 自动服务发现和负载均衡

3. **弹性扩缩容**
   - 按需添加/移除节点
   - 无需修改代码
