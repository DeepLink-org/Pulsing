# 快速开始

本指南帮助您快速安装 Pulsing 并掌握核心概念。

## 安装

### 前置条件

- **Python 3.10+**
- **Rust 工具链** (用于构建原生扩展)
- **Linux/macOS**

### 从源码安装

```bash
git clone https://github.com/reiase/pulsing.git
cd pulsing

# 安装 Rust (如果尚未安装)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 构建和安装
pip install maturin
maturin develop
```

### 从 PyPI 安装

```bash
pip install pulsing
```

---

## 什么是 Actor？

Actor 是具有私有状态的隔离计算单元，顺序处理消息，本地和远程 Actor 使用相同的 API。

```mermaid
graph LR
    A[发送者] -->|消息| B[Actor 邮箱]
    B --> C[Actor]
    C -->|响应| A

    style A fill:#6366F1,color:#fff
    style B fill:#818CF8,color:#fff
    style C fill:#818CF8,color:#fff
```

---

## 第一个 Actor（30秒）

### 方式一：原生异步 API（推荐）

```python
import asyncio
from pulsing.actor import init, shutdown, remote

@remote
class Counter:
    def __init__(self, value=0):
        self.value = value

    def inc(self):
        self.value += 1
        return self.value

async def main():
    await init()
    counter = await Counter.spawn(value=0)
    print(await counter.inc())  # 1
    print(await counter.inc())  # 2
    await shutdown()

asyncio.run(main())
```

### 方式二：Ray 兼容 API（轻松迁移）

```python
from pulsing.compat import ray

ray.init()

@ray.remote
class Counter:
    def __init__(self, value=0):
        self.value = value

    def inc(self):
        self.value += 1
        return self.value

counter = Counter.remote(value=0)
print(ray.get(counter.inc.remote()))  # 1
print(ray.get(counter.inc.remote()))  # 2

ray.shutdown()
```

**任意 Python 对象**都可以作为消息——字符串、字典、列表或自定义类。

---

## 有状态的 Actor

```python
class Counter(Actor):
    def __init__(self):
        self.value = 0

    async def receive(self, msg):
        if msg == "inc":
            self.value += 1
            return self.value
        if msg == "get":
            return self.value
```

---

## API 对比

| API | 风格 | 适用场景 |
|-----|------|----------|
| `pulsing.actor` | 异步 (`await`) | 新项目，高性能 |
| `pulsing.compat.ray` | 同步 (Ray 风格) | Ray 迁移，快速上手 |

### @remote 装饰器（原生 API）

```python
from pulsing.actor import init, shutdown, remote

@remote
class Counter:
    def __init__(self, initial=0):
        self.value = initial

    def inc(self, n=1):
        self.value += n
        return self.value

async def main():
    await init()
    counter = await Counter.spawn(initial=10)
    print(await counter.inc(5))   # 15
    await shutdown()
```

---

## 集群通信

Pulsing 使用 SWIM gossip 协议——无需外部服务！

**节点 1（种子节点）：**
```python
config = SystemConfig.with_addr("0.0.0.0:8000")
system = await create_actor_system(config)
await system.spawn("worker", MyActor(), public=True)
```

**节点 2（加入集群）：**
```python
config = SystemConfig.with_addr("0.0.0.0:8001").with_seeds(["node1:8000"])
system = await create_actor_system(config)

worker = await system.resolve_named("worker")
result = await worker.ask("do_work")  # API 完全相同！
```

---

## 核心概念

| 概念 | 描述 |
|------|------|
| **Actor** | 具有私有状态的隔离单元 |
| **消息** | 任意 Python 对象 |
| **@remote** | 原生异步装饰器 (via `pulsing.actor`) |
| **ray.remote** | Ray 兼容装饰器 (via `pulsing.compat.ray`) |
| **集群** | SWIM 协议自动发现 |

---

## 下一步

- [Actor 指南](../guide/actors.zh.md) - 高级模式
- [Agent 框架](../agent/index.zh.md) - AutoGen 和 LangGraph 集成
- [示例](../examples/index.zh.md) - 真实用例
