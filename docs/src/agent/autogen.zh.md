# AutoGen 集成

Pulsing 实现了 `PulsingRuntime`，为 Microsoft AutoGen 提供分布式运行时。

## 特性

- **分布式执行** - 在集群不同节点上运行 Agent
- **服务发现** - 通过 Gossip 协议自动发现 Agent
- **负载均衡** - 在多个 Agent 实例间分配负载
- **API 兼容** - 可直接替换 `SingleThreadedAgentRuntime`

## 快速开始

```python
from pulsing.integrations.autogen import PulsingRuntime
from autogen_core import AgentId, RoutedAgent, message_handler

class MyAgent(RoutedAgent):
    @message_handler
    async def handle_msg(self, message: str, ctx) -> str:
        return f"已处理: {message}"

async def main():
    # 单机模式
    runtime = PulsingRuntime()
    await runtime.start()

    # 注册 Agent (与 AutoGen API 一致)
    await runtime.register_factory("my_agent", lambda: MyAgent())

    # 发送消息
    response = await runtime.send_message("Hello", AgentId("my_agent", "default"))
    print(response)

    await runtime.stop()
```

## 分布式模式

### 节点 1 (种子节点)

```python
runtime = PulsingRuntime(addr="0.0.0.0:8000")
await runtime.start()
await runtime.register_factory("writer", lambda: WriterAgent())
```

### 节点 2 (Worker)

```python
runtime = PulsingRuntime(
    addr="0.0.0.0:8001",
    seeds=["192.168.1.100:8000"]
)
await runtime.start()
await runtime.register_factory("editor", lambda: EditorAgent())
```

### 任意节点 (客户端)

```python
# 消息自动路由到集群中的目标 Agent
await runtime.send_message(draft, AgentId("editor", "default"))
```

## 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `addr` | 节点地址 | `None` (单机模式) |
| `seeds` | 种子节点地址 | `[]` |

## 示例

```bash
cd examples/agent/autogen
./run_distributed.sh
```

## API 参考

### PulsingRuntime

```python
class PulsingRuntime:
    def __init__(self, addr: str | None = None, seeds: list[str] = [])
    async def start() -> None
    async def stop() -> None
    async def register_factory(type: str, factory: Callable) -> None
    async def send_message(message: Any, recipient: AgentId, sender: AgentId | None = None) -> Any
```
