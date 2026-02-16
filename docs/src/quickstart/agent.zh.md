# 教程：分布式 Agent

将 Pulsing 与 **AutoGen** 和 **LangGraph** 集成，跨机器分布你的 Agent。

**你将学到：**

- 使用 `PulsingRuntime` 分布 AutoGen Agent
- 使用 `with_pulsing()` 分布 LangGraph 节点
- LLM 在 GPU 上运行，工具在 CPU 上运行

```
┌─────────────────────────────────────────────────────────────┐
│                    你的 Agent 代码                          │
│              (AutoGen / LangGraph)                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Pulsing Actor System                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  节点 1  │   │  节点 2  │   │  节点 3  │
    │   GPU    │   │   CPU    │   │   CPU    │
    │   LLM    │   │   工具   │   │   工具   │
    └──────────┘   └──────────┘   └──────────┘
```

---

## 前置条件

```bash
pip install pulsing

# AutoGen
pip install autogen-agentchat autogen-ext

# LangGraph
pip install langgraph langchain-openai
```

---

## 方案 A：AutoGen

### 步骤 1：定义 Agent

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

model = OpenAIChatCompletionClient(model="gpt-4o-mini")
agent = AssistantAgent("assistant", model_client=model)
```

### 步骤 2：使用 PulsingRuntime

用 `PulsingRuntime` 替换 `SingleThreadedAgentRuntime`：

```python
from pulsing.integrations.autogen import PulsingRuntime

# 单进程（默认）
runtime = PulsingRuntime()

# 分布式模式
runtime = PulsingRuntime(addr="0.0.0.0:8000")
```

### 步骤 3：注册并运行

```python
await runtime.start()
await runtime.register_factory("assistant", lambda: agent)

# 发送消息
result = await runtime.send_message(
    "Hello, agent!",
    AgentId("assistant", "default")
)
```

### 完整示例

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_core import AgentId
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pulsing.integrations.autogen import PulsingRuntime

async def main():
    model = OpenAIChatCompletionClient(model="gpt-4o-mini")
    agent = AssistantAgent("assistant", model_client=model)

    runtime = PulsingRuntime(addr="0.0.0.0:8000")
    await runtime.start()
    await runtime.register_factory("assistant", lambda: agent)

    result = await runtime.send_message(
        "2 + 2 等于多少?",
        AgentId("assistant", "default")
    )
    print(result)

asyncio.run(main())
```

---

## 方案 B：LangGraph

### 步骤 1：构建图

```python
from langgraph.graph import StateGraph

def llm_node(state):
    # LLM 调用
    return {"messages": [...]}

def tool_node(state):
    # 工具执行
    return {"messages": [...]}

graph = StateGraph(State)
graph.add_node("llm", llm_node)
graph.add_node("tool", tool_node)
graph.add_edge("llm", "tool")
app = graph.compile()
```

### 步骤 2：用 Pulsing 包装

```python
from pulsing.integrations.langgraph import with_pulsing

distributed_app = with_pulsing(
    app,
    node_mapping={
        "llm": "langgraph_node_llm",    # → GPU 服务器
        "tool": "langgraph_node_tool",  # → CPU 服务器
    },
    seeds=["gpu-server:8001"],
)
```

### 步骤 3：运行

```python
result = await distributed_app.ainvoke({"messages": [...]})
```

---

## 运行示例

```bash
# AutoGen 分布式
cd examples/agent/autogen && ./run_distributed.sh

# LangGraph 分布式
cd examples/agent/langgraph && ./run_distributed.sh
```

---

## 使用场景

| 场景 | 配置 |
|------|------|
| **LLM + 工具** | LLM 在 GPU 节点，工具在 CPU 节点 |
| **多 Agent** | 每个 Agent 在不同节点 |
| **弹性扩展** | 无需代码改动即可增减节点 |

---

## 下一步

- [Agent：AutoGen](../agent/autogen.zh.md) — AutoGen 集成详解
- [Agent：LangGraph](../agent/langgraph.zh.md) — LangGraph 集成详解
- [指南：远程 Actor](../guide/remote_actors.zh.md) — 集群设置
