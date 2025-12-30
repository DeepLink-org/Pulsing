# 分布式计数器（模式）

本页描述一个常见模式：启动一个**具名计数器 Actor**，其它节点通过名称找到它并远程更新。

如果你需要可运行的基线，可以先看 `examples/python/named_actors.py` 和 `examples/python/cluster.py`，再套用下面的模式。

## 模式说明

1. 启动 seed 节点，并创建一个**public 的具名 Actor**
2. 启动其它节点加入集群，通过名称 **resolve/find**
3. 使用 `ask` 远程更新状态并获取返回值

## 示例草图

```python
import asyncio
from pulsing.actor import Actor, Message, SystemConfig, create_actor_system


class Counter(Actor):
    def __init__(self):
        self.v = 0

    def receive(self, msg: Message) -> Message:
        if msg.msg_type == "Inc":
            self.v += int(msg.to_json().get("n", 1))
            return Message.from_json("Value", {"v": self.v})
        if msg.msg_type == "Get":
            return Message.from_json("Value", {"v": self.v})
        return Message.empty()


async def seed():
    system = await create_actor_system(SystemConfig.with_addr("0.0.0.0:8000"))
    await system.spawn("global_counter", Counter(), public=True)
    await asyncio.Event().wait()


async def worker():
    system = await create_actor_system(
        SystemConfig.with_addr("0.0.0.0:8001").with_seeds(["127.0.0.1:8000"])
    )
    await asyncio.sleep(1.0)
    ref = await system.find("global_counter")
    resp = await ref.ask(Message.from_json("Inc", {"n": 1}))
    print(resp.to_json())


asyncio.run(asyncio.gather(seed(), worker()))
```
