# Pulsing 原生 Agent

Pulsing 提供轻量的原生 Agent 工具箱，用于构建多智能体应用，与 Actor 模型完全兼容。

## 设计理念

**简单透明。**

- `@agent` = `@remote` + 元信息（无魔法）
- 元信息用于可视化/调试，不会自动注入 prompt
- 完全掌控 LLM 调用和对话流程

## 核心 API

### `@agent` 装饰器

`@agent` 装饰器等同于 `@remote`，但附加元信息用于可视化和调试：

```python
from pulsing.actor import remote, resolve
from pulsing.agent import agent, runtime, llm, get_agent_meta, list_agents

# @remote: 基础 Actor
@remote
class Worker:
    async def work(self):
        return "done"

# @agent: 带元信息的 Actor
@agent(role="研究员", goal="深入分析", domain="AI")
class Researcher:
    async def analyze(self, topic: str) -> str:
        client = await llm()
        resp = await client.ainvoke(f"分析: {topic}")
        return resp.content
```

### 元信息访问

```python
async with runtime():
    r = await Researcher.spawn(name="researcher")

    # 通过名称获取元信息
    meta = get_agent_meta("researcher")
    print(meta.role)       # "研究员"
    print(meta.goal)       # "深入分析"
    print(meta.tags)       # {"domain": "AI"}

    # 列出所有 Agent
    for name, meta in list_agents().items():
        print(f"{name}: {meta.role}")
```

### `@remote` vs `@agent`

| 特性 | `@remote` | `@agent` |
|------|-----------|----------|
| 功能 | Actor 化 | Actor 化 + 元信息 |
| 用途 | 通用 | 可视化/调试 |
| 元信息 | 无 | `role`, `goal`, `backstory`, `tags` |
| 性能 | 基准 | 几乎一致 |

## 运行时管理

```python
from pulsing.agent import runtime, cleanup

async with runtime():
    # 创建和使用 Agent
    agent = await MyAgent.spawn(name="agent")
    await agent.work()

# 可选：清理全局状态
cleanup()
```

### 分布式模式

```python
# 节点 A
async with runtime(addr="0.0.0.0:8001"):
    await JudgeActor.spawn(name="judge")

# 节点 B（自动发现节点 A）
async with runtime(addr="0.0.0.0:8002", seeds=["node_a:8001"]):
    judge = await resolve("judge")  # 跨节点透明调用
    await judge.submit(idea)
```

## LLM 集成

```python
from pulsing.agent import llm

async def analyze(topic: str):
    # 获取 LLM 客户端（懒加载单例）
    client = await llm(temperature=0.8)
    resp = await client.ainvoke(f"分析: {topic}")
    return resp.content
```

**环境变量：**

- `OPENAI_API_KEY`: API 密钥（必需）
- `OPENAI_BASE_URL`: 自定义 API 地址（可选）
- `LLM_MODEL`: 默认模型名（可选，默认 `gpt-4o-mini`）

## 工具函数

### JSON 解析

```python
from pulsing.agent import parse_json, extract_field

# 安全 JSON 解析，带默认值
data = parse_json('{"key": "value"}', default={})

# 提取特定字段
value = extract_field(response, "answer", default="unknown")
```

## 完整示例

```python
import asyncio
from pulsing.actor import remote, resolve
from pulsing.agent import agent, runtime, llm, parse_json, list_agents

@remote
class Moderator:
    """使用 @remote 的协调者（基础 Actor）"""

    def __init__(self, topic: str):
        self.topic = topic
        self.opinions = []

    async def collect_opinion(self, agent_name: str, opinion: str):
        self.opinions.append({"agent": agent_name, "opinion": opinion})
        return {"received": True}

    async def summarize(self):
        return {"topic": self.topic, "opinions": self.opinions}

@agent(role="分析师", goal="提供洞见", domain="tech")
class Analyst:
    """使用 @agent 的分析师（带元信息的 Actor）"""

    def __init__(self, name: str, moderator: str, mock: bool = True):
        self.name = name
        self.moderator_name = moderator
        self.mock = mock

    async def analyze(self, topic: str):
        if self.mock:
            opinion = f"[{self.name}] 对 {topic} 的分析：前景看好"
        else:
            client = await llm()
            resp = await client.ainvoke(f"简要分析: {topic}")
            opinion = resp.content

        # 提交给协调者
        moderator = await resolve(self.moderator_name)
        await moderator.collect_opinion(self.name, opinion)
        return opinion

async def main():
    async with runtime():
        # 创建协调者
        moderator = await Moderator.spawn(topic="AI 趋势", name="moderator")

        # 创建分析师
        for i in range(3):
            name = f"analyst_{i}"
            await Analyst.spawn(
                name=name,
                moderator="moderator",
                mock=True,
                name=name,
            )

        # 显示 Agent 元信息
        print("已注册的 Agent:")
        for name, meta in list_agents().items():
            print(f"  {name}: {meta.role}")

        # 运行分析
        for i in range(3):
            analyst = await resolve(f"analyst_{i}")
            await analyst.analyze("AI 趋势")

        # 获取总结
        result = await moderator.summarize()
        print(f"总结: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 示例

参见 `examples/agent/pulsing/` 目录的完整示例：

| 示例 | 说明 |
|------|------|
| `mbti_discussion.py` | 基于 MBTI 人格的多智能体讨论 |
| `parallel_ideas_async.py` | 并行创意生成与竞争提交 |
| `runtime_lifecycle_example.py` | 运行时生命周期管理 |

```bash
# 运行 MBTI 讨论（模拟模式）
python examples/agent/pulsing/mbti_discussion.py --mock --group-size 6

# 运行并行创意（模拟模式）
python examples/agent/pulsing/parallel_ideas_async.py --mock --n-ideas 5
```

## 下一步

- [AutoGen 集成](autogen.zh.md) — 在 Pulsing 中使用 AutoGen
- [LangGraph 集成](langgraph.zh.md) — 在 Pulsing 中使用 LangGraph
- [远程 Actor](../guide/remote_actors.zh.md) — 集群部署
