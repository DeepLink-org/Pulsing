# Ping-Pong

最简单的 Actor 通信示例。

## 代码

```python
import asyncio
from pulsing.actor import Actor, SystemConfig, create_actor_system


class PingPong(Actor):
    async def receive(self, msg):
        if msg == "ping":
            return "pong"
        return f"echo: {msg}"


async def main():
    system = await create_actor_system(SystemConfig.standalone())
    actor = await system.spawn("pingpong", PingPong())

    print(await actor.ask("ping"))   # -> pong
    print(await actor.ask("hello"))  # -> echo: hello

    await system.shutdown()


asyncio.run(main())
```

## 运行

```bash
python examples/python/ping_pong.py
```

## 要点

- `Actor` 是基类，实现 `receive()` 处理消息
- **任意 Python 对象**都可以作为消息（字符串、字典、列表等）
- `actor.ask(msg)` 发送消息并等待响应
- `system.shutdown()` 干净地停止所有 Actor
