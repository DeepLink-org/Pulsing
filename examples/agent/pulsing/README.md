# Pulsing Multi-Agent 示例

本目录包含使用 Pulsing Actor 框架实现的多智能体示例。

## 核心 API

```python
import pulsing as pul
from pulsing.agent import agent, runtime, llm, parse_json, get_agent_meta

# @pul.remote: 基础 Actor 装饰器
@pul.remote
class MyActor:
    async def work(self): ...

# @agent: 带元信息的 Actor（用于可视化/调试）
@agent(role="研究员", goal="分析问题")
class Researcher:
    async def analyze(self, topic: str): ...

# 运行
async with runtime():
    actor = await MyActor.spawn(name="actor")
    result = await actor.work()

    # 通过名称获取其他 Actor
    peer = await MyActor.resolve("actor")
```

## 示例列表

### 1. MBTI 人格讨论 (`mbti_discussion.py`)

基于 MBTI 人格类型的多智能体讨论与投票，演示 `@pul.remote` 与 `@agent` 的区别。

```
         ┌─────────────────────────────────────┐
         │         ModeratorActor              │
         │      (使用 @pul.remote，协调流程)    │
         └──────────────┬──────────────────────┘
                        │ resolve()
    ┌───────────────────┼───────────────────────┐
    ▼                   ▼                       ▼
┌────────┐         ┌────────┐             ┌────────┐
│ INTJ   │ ←────→  │ ENFP   │  ←────→     │ ISFJ   │
│ Agent  │  辩论   │ Agent  │   辩论      │ Agent  │
│(@agent)│         │(@agent)│             │(@agent)│
└────────┘         └────────┘             └────────┘
```

**特点：**
- `ModeratorActor` 使用 `@pul.remote`（普通 Actor）
- `MBTIAgent` 使用 `@agent`（附带元信息）
- 可通过 `get_agent_meta()` 和 `list_agents()` 获取元信息

```bash
# 模拟模式
python mbti_discussion.py --mock --group-size 6 --rounds 2

# LLM 模式
export OPENAI_API_KEY=xxx
python mbti_discussion.py --topic "AI是否应该有情感"
```

### 2. 并行创意生成 (`parallel_ideas_async.py`)

多个 Agent 并行生成方案，竞争提交给 Judge。

```
    ┌───────────┐      ┌───────────┐      ┌───────────┐
    │ IdeaAgent │ ←──→ │ IdeaAgent │ ←──→ │ IdeaAgent │
    │  (性能)    │ 协作 │  (安全)    │ 协作 │  (产品)    │
    └─────┬─────┘      └─────┬─────┘      └─────┬─────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼ submit()
                       ┌───────────┐
                       │   Judge   │
                       │ (截止时间) │
                       └───────────┘
```

**特点：**
- 完全异步并行执行
- Agent 间可通过 `ClassName.resolve()` 协作
- 竞争性提交 + 截止时间机制

```bash
# 模拟模式
python parallel_ideas_async.py --mock --n-ideas 5 --timeout 30

# LLM 模式
export OPENAI_API_KEY=xxx
python parallel_ideas_async.py --question "如何提升 API 吞吐？"
```

## 安装

```bash
cd Pulsing
pip install -e .

# 使用 LLM 模式需要
pip install langchain-openai
```

## `@pul.remote` vs `@agent`

| 特性 | `@pul.remote` | `@agent` |
|------|---------------|----------|
| 功能 | Actor 化 | Actor 化 + 元信息 |
| 用途 | 通用 | 需要可视化/调试时 |
| 元信息 | 无 | `role`, `goal`, `backstory`, `tags` |
| 性能开销 | 无 | 几乎无 |

```python
# @pul.remote: 直接使用
@pul.remote
class Worker:
    async def work(self): ...

# @agent: 附加元信息
@agent(role="分析师", goal="深入分析", domain="AI")
class Analyst:
    async def analyze(self): ...

# 获取元信息
meta = get_agent_meta("analyst")
print(meta.role)   # "分析师"
print(meta.tags)   # {"domain": "AI"}

# 列出所有 Agent
for name, meta in list_agents().items():
    print(f"{name}: {meta.role}")
```

## 分布式部署

```python
# 节点 A
async with runtime(addr="0.0.0.0:8001"):
    await JudgeActor.spawn(name="judge")

# 节点 B（自动发现节点 A）
async with runtime(addr="0.0.0.0:8002", seeds=["node_a:8001"]):
    judge = await JudgeActor.resolve("judge")  # 跨节点透明调用
    await judge.submit(idea)
```
