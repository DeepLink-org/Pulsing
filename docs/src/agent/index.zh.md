# Agent 框架集成

Pulsing 原生支持主流 Agent 框架，让您的应用轻松从单进程扩展到分布式集群。

## 设计理念

**不替代，而是增强。**

- 用户代码几乎无需改动
- API 与原框架完全兼容
- 一行代码实现分布式

## 可选方案

| 方案 | 集成方式 | 说明 |
|------|----------|------|
| [**Pulsing 原生**](native.zh.md) | `@agent` | 轻量工具箱，支持元信息 |
| [AutoGen](autogen.zh.md) | `PulsingRuntime` | 替代默认运行时 |
| [LangGraph](langgraph.zh.md) | `with_pulsing()` | 包装编译后的图 |

## Pulsing 原生 Agent

从零构建多智能体应用时，使用 Pulsing 原生的 `@agent` 装饰器：

```python
from pulsing.actor import resolve
from pulsing.agent import agent, runtime, llm, list_agents

@agent(role="研究员", goal="深入分析")
class Researcher:
    async def analyze(self, topic: str) -> str:
        client = await llm()
        return await client.ainvoke(f"分析: {topic}")

async with runtime():
    r = await Researcher.spawn(name="researcher")
    result = await r.analyze("AI 趋势")

    # 访问元信息用于可视化
    for name, meta in list_agents().items():
        print(f"{name}: {meta.role}")
```

**核心特点：**

- `@agent` = `@remote` + 元信息（无魔法）
- 元信息用于可视化/调试
- 完全掌控 LLM 调用

→ [了解更多 Pulsing 原生 Agent](native.zh.md)

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    您的 Agent 代码                          │
│              (AutoGen / LangGraph / ...)                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Pulsing 集成层                           │
│          PulsingRuntime / with_pulsing()                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  Pulsing Actor System                       │
│       Gossip 协议 │ 位置透明 │ 流式消息                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  节点 1  │◄─►│  节点 2  │◄─►│  节点 3  │
    │   GPU    │   │   CPU    │   │   CPU    │
    └──────────┘   └──────────┘   └──────────┘
```

## 快速对比

### AutoGen

```python
from pulsing.autogen import PulsingRuntime

# 替代 SingleThreadedAgentRuntime
runtime = PulsingRuntime(addr="0.0.0.0:8000")
await runtime.start()
await runtime.register_factory("agent", lambda: MyAgent())
```

### LangGraph

```python
from pulsing.langgraph import with_pulsing

app = graph.compile()
distributed_app = with_pulsing(
    app,
    node_mapping={"llm": "langgraph_node_llm"},
    seeds=["gpu-server:8001"],
)
```

## 典型场景

1. **LLM + Tool 分离** - LLM 在 GPU，工具在 CPU
2. **多 Agent 协作** - Agent 分布在不同节点
3. **弹性扩缩容** - 无需修改代码即可增删节点

## 运行示例

```bash
# AutoGen
cd examples/agent/autogen && ./run_distributed.sh

# LangGraph
cd examples/agent/langgraph && ./run_distributed.sh
```
