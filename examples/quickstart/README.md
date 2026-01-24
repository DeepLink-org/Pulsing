# 🚀 Pulsing 快速入门

**5分钟让你的 AI 们开始对话！**

## 第一步：安装

```bash
pip install pulsing
```

## 第二步：运行示例

### 示例 1: Hello Agent（入门）

最简单的 Multi-Agent 应用，两个 Agent 互相打招呼：

```bash
python examples/quickstart/hello_agent.py
```

输出：
```
==================================================
🎉 Pulsing Multi-Agent 快速入门
==================================================
✨ [Alice] 已上线
✨ [Bob] 已上线

--- Agent 互相打招呼 ---

👋 [Alice] 正在向 [Bob] 打招呼...
📨 [Bob] 收到消息: 嗨，我是 Alice！
💬 [Alice] 收到回复: 你好！我是 Bob

👋 [Bob] 正在向 [Alice] 打招呼...
📨 [Alice] 收到消息: 嗨，我是 Bob！
💬 [Bob] 收到回复: 你好！我是 Alice

==================================================
✅ 完成！你已经创建了第一个 Multi-Agent 应用
==================================================
```

### 示例 2: AI 聊天室（有趣）

4 个不同性格的 AI 讨论一个话题：

```bash
python examples/quickstart/ai_chat_room.py
```

自定义话题：

```bash
python examples/quickstart/ai_chat_room.py --topic "远程办公是否会成为主流？" --rounds 5
```

---

## ⚡ 进阶示例

### 示例 3: Function → Fleet（横向扩展）

**一行代码把函数变成可横向扩展的服务**——同一份代码，workers 从 1→N，吞吐线性提升：

```bash
# 8 个 worker 处理 200 个任务
python examples/quickstart/function_to_fleet.py

# 调整 worker 数量
WORKERS=16 ITEMS=500 python examples/quickstart/function_to_fleet.py
```

输出：
```
==================================================
⚡ Function → Fleet Result
==================================================
   Workers:     8
   Tasks:       200
   Duration:    0.52s
   Throughput:  384.6 qps
==================================================
✅ Same code, more workers = higher throughput
==================================================
```

**核心代码**（仅 27 行）：
```python
@remote
class Worker:
    async def run(self, x: int) -> int:
        await asyncio.sleep(0.02)  # simulate I/O
        return x * x

async with runtime():
    ws = [await Worker.spawn(name=f"w{i}") for i in range(n)]
    res = await asyncio.gather(*(ws[i % n].run(i) for i in range(m)))
```

### 示例 4: Chaos-proof（故障自愈）

**Actor 崩溃后自动重启**——30% 崩溃率下，任务仍然全部完成：

```bash
python examples/quickstart/chaos_proof.py
```

输出：
```
==================================================
🛡️  Chaos-proof Result
==================================================
   Total tasks:   50
   Succeeded:     50
   Retries:       23
   Crash rate:    30%
==================================================
✅ All succeeded! Actor auto-restarted on crash.
==================================================
```

**核心代码**：
```python
@remote(restart_policy="on_failure", max_restarts=50)
class FlakyWorker:
    def work(self, x: int) -> int:
        if random.random() < 0.3:  # 30% 概率崩溃
            raise RuntimeError("boom")
        return x + 1
```

**亮点**：30% 崩溃率，框架自动重启 Actor，调用方简单重试，最终 100% 完成。

---

## 核心概念（10秒理解）

```python
import pulsing as pul
from pulsing.agent import runtime

# 1. @pul.remote 让类变成可分布式部署的 Agent
@pul.remote
class MyAgent:
    def hello(self):
        return "Hello!"

# 2. runtime() 管理 Agent 生命周期
async with runtime():
    # 3. spawn() 创建 Agent 实例
    agent = await MyAgent.spawn(name="my_agent")

    # 4. 直接调用方法（自动变成远程调用）
    result = await agent.hello()

    # 5. MyAgent.resolve() 通过名字找到已有 Agent
    same_agent = await MyAgent.resolve("my_agent")
```

## 下一步

| 我想... | 去看... |
|---------|---------|
| 看更复杂的 Multi-Agent | `examples/agent/pulsing/mbti_discussion.py` |
| 学习分布式部署 | `examples/python/cluster.py` |
| 集成 AutoGen | `examples/agent/autogen/` |
| 集成 LangGraph | `examples/agent/langgraph/` |

```bash
# MBTI 人格讨论（推荐！）
python examples/agent/pulsing/mbti_discussion.py --mock --group-size 6

# 并行创意生成
python examples/agent/pulsing/parallel_ideas_async.py --mock --n-ideas 5
```

## 常见问题

### Q: 如何接入真实 LLM？

```bash
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=https://api.openai.com/v1  # 可选

# 去掉 --mock 参数
python examples/agent/pulsing/mbti_discussion.py --topic "AI 的未来"
```

### Q: 如何分布式部署？

```python
# 节点 1
async with runtime(addr="0.0.0.0:8001"):
    await MyAgent.spawn(name="agent")

# 节点 2（自动发现节点 1）
async with runtime(addr="0.0.0.0:8002", seeds=["node1:8001"]):
    agent = await MyAgent.resolve("agent")  # 跨节点调用
```

### Q: @remote vs @agent 有什么区别？

- `@remote`：基础 Actor，功能完整
- `@agent`：= `@remote` + 元信息（role/goal），用于可视化和调试

```python
from pulsing.agent import agent, get_agent_meta

@agent(role="研究员", goal="深入分析")
class Researcher:
    ...

# 可以查询元信息
meta = get_agent_meta("researcher")
print(meta.role)  # "研究员"
```
