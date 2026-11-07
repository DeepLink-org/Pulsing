# 快速入门

```bash
pip install pulsing
python examples/quickstart/hello_agent.py
python examples/quickstart/ai_chat_room.py --rounds 2
python examples/quickstart/function_to_fleet.py
```

这些示例无需 LLM 或 API key。聊天室使用本地文本模板，演示 actor 间通信。

核心用法：

```python
import pulsing as pul
from pulsing.agent import runtime


@pul.remote
class Counter:
    def __init__(self):
        self.value = 0

    def inc(self):
        self.value += 1
        return self.value


async def main():
    async with runtime():
        counter = await Counter.spawn(name="counter")
        print(await counter.inc())
```

分布式部署时为 `runtime` 配置 `addr` 和 `seeds`，
通过 `await Counter.resolve("counter")` 查找已有 actor。

下一步：

- `examples/python/message_patterns.py`：请求响应、单向与流式消息。
- `examples/python/cluster.py`：多节点通信。
- `examples/python/tensor_message_fast_path.py`：buffer 传输。
