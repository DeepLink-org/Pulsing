# Actor 元信息辅助工具

`pulsing.agent` 保留少量 actor 辅助功能：

- `agent`：在 `@remote` 基础上附加 role/goal 等元信息。
- `runtime`：初始化和关闭 actor system 的异步上下文管理器。
- `get_agent_meta`、`list_agents`、`cleanup`：管理元信息。
- `parse_json` 和 `extract_field`：解析工具。

```python
import asyncio
from pulsing.agent import agent, cleanup, get_agent_meta, runtime


@agent(role="Worker", goal="Count messages")
class Counter:
    def __init__(self):
        self.value = 0

    def inc(self):
        self.value += 1
        return self.value


async def main():
    try:
        async with runtime():
            counter = await Counter.spawn(name="counter")
            print(await counter.inc())
            print(get_agent_meta("counter").role)
    finally:
        cleanup()


asyncio.run(main())
```

这些工具不提供 LLM 客户端、prompt、工具执行或 coding-agent 循环。
这些能力由上层应用自行选择和实现。

集群配置见 [远程 Actor](../guide/remote_actors.zh.md)，生命周期示例见
`examples/agent/pulsing/runtime_lifecycle_example.py`。
