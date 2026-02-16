# 快速开始

用三个步骤、约 **10 分钟** 从零到**分布式 Actor**：第一个 Actor、有状态 Actor，再到同一套代码跑在两个节点上。

---

## 安装

```bash
pip install pulsing
```

---

## 1. 第一个 Actor（约 2 分钟）

定义一个类，加上 `@pul.remote`，然后 spawn 并调用。

```python
import asyncio
import pulsing as pul

@pul.remote
class Greeter:
    def greet(self, name: str) -> str:
        return f"Hello, {name}!"

async def main():
    await pul.init()
    greeter = await Greeter.spawn()
    print(await greeter.greet("World"))  # Hello, World!
    await pul.shutdown()

asyncio.run(main())
```

`@pul.remote` 把类变成分布式 Actor；`spawn()` 创建实例，方法调用就是普通的 `await`。

---

## 2. 有状态 Actor（约 3 分钟）

Actor 自带状态。下面这个计数器维护一个值，并暴露 `inc` 和 `get`。

```python
import asyncio
import pulsing as pul

@pul.remote
class Counter:
    def __init__(self, value: int = 0):
        self.value = value

    def inc(self, n: int = 1) -> int:
        self.value += n
        return self.value

    def get(self) -> int:
        return self.value

async def main():
    await pul.init()
    counter = await Counter.spawn(value=0)
    print(await counter.inc())   # 1
    print(await counter.inc(2))  # 3
    print(await counter.get())   # 3
    await pul.shutdown()

asyncio.run(main())
```

同样的思路：一个 Actor 实例、私有状态、通过方法调用发消息。无共享内存、无锁。

---

## 3. 分布式：同一套代码，两个节点（约 5 分钟）

在两个进程里跑同一种 Actor。**只有初始化不同**：第一个节点绑定地址，第二个节点用 `seeds` 加入集群。

**节点 1（seed）：**

```python
import asyncio
import pulsing as pul

@pul.remote
class Counter:
    def __init__(self, value: int = 0):
        self.value = value
    def inc(self, n: int = 1) -> int:
        self.value += n
        return self.value

async def main():
    await pul.init(addr="0.0.0.0:8000")
    await Counter.spawn(value=0, name="counter")
    await asyncio.Event().wait()  # 保持运行

asyncio.run(main())
```

**节点 2（加入集群后 resolve 并调用）：**

```python
import asyncio
import pulsing as pul

@pul.remote
class Counter:
    def __init__(self, value: int = 0):
        self.value = value
    def inc(self, n: int = 1) -> int:
        self.value += n
        return self.value

async def main():
    await pul.init(addr="0.0.0.0:8001", seeds=["127.0.0.1:8000"])
    counter = await Counter.resolve("counter")
    print(await counter.inc(10))  # 10 — 同一套 API，远程 Actor
    await pul.shutdown()

asyncio.run(main())
```

**变化只有：** `init(addr=..., seeds=...)` 和用 `Counter.resolve("counter")` 代替 `spawn()`。其余代码不变 —— **位置透明**。

---

## 下一步：选择你的路径

<div class="grid cards" markdown>

-   :material-robot:{ .lg .middle } **LLM 推理服务**

    ---

    构建支持流式输出和 OpenAI 兼容 API 的可扩展推理后端。

    [:octicons-arrow-right-24: ~10 分钟](llm_inference.zh.md)

-   :material-account-group:{ .lg .middle } **分布式 Agent**

    ---

    集成 AutoGen 和 LangGraph，跨机器分布 Agent。

    [:octicons-arrow-right-24: ~10 分钟](agent.zh.md)

-   :material-swap-horizontal:{ .lg .middle } **与 Ray 配合使用**

    ---

    通过 `pul.mount()` 将 Ray Actor 接入 Pulsing 网络。为 Ray 集群增加流式和发现能力。

    [:octicons-arrow-right-24: ~5 分钟](migrate_from_ray.zh.md)

</div>

---

## 深入了解

| 目标 | 链接 |
|------|------|
| 命名 Actor 与 ask/tell | [Actor 模式](patterns.zh.md) |
| 组建集群（Gossip / Head / Ray） | [集群组网](cluster_networking.zh.md) |
| Actor 基础与模式 | [Actor 指南](../guide/actors.zh.md) |
| 何时用 ask / tell / streaming | [通信范式](../guide/communication_patterns.zh.md) |
| 集群搭建与 resolve | [远程 Actor](../guide/remote_actors.zh.md) |
| 运维与巡检 | [运维操作](../guide/operations.zh.md) |
