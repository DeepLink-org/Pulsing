# 分布式计数器（模式）

本页描述一个常见模式：启动一个**具名计数器 Actor**，其它节点通过名称找到它并远程更新。

如果你需要可运行的基线，可以先看 `examples/python/named_actors.py` 和 `examples/python/cluster.py`，再套用下面的模式。

## 模式说明

1. 启动 seed 节点，并创建一个**命名 Actor**（可被 resolve 发现）
2. 启动其它节点加入集群，通过名称 **resolve**
3. 使用 `ask` 远程更新状态并获取返回值

## 示例草图

```python
import asyncio
import pulsing as pul


class Counter:
    def __init__(self):
        self.v = 0

    async def receive(self, msg):
        if msg.get("action") == "inc":
            self.v += msg.get("n", 1)
            return {"v": self.v}
        if msg.get("action") == "get":
            return {"v": self.v}
        return {}


async def seed():
    system = await pul.actor_system(addr="0.0.0.0:8000")
    # 命名 actor 自动可被 resolve 发现
    await system.spawn(Counter(), name="global_counter")
    await asyncio.Event().wait()


async def worker():
    system = await pul.actor_system(
        addr="0.0.0.0:8001",
        seeds=["127.0.0.1:8000"]
    )
    await asyncio.sleep(1.0)
    ref = await system.resolve("global_counter")
    resp = await ref.ask({"action": "inc", "n": 1})
    print(resp)


asyncio.run(asyncio.gather(seed(), worker()))
```
