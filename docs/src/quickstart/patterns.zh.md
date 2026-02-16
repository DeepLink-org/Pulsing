# Actor 模式

在写完第一个 Actor 之后常用的几种写法：命名 Actor、resolve，以及何时用 ask / tell。

---

## 命名 Actor 与 resolve

给 Actor 起一个 **name**，其他代码（本进程或集群内）可以用 **resolve** 按名查找：

```python
import pulsing as pul

@pul.remote
class Worker:
    def process(self, data: str) -> str:
        return f"processed: {data}"

async def main():
    await pul.init()
    # 带名字 spawn，可通过 resolve 发现
    await Worker.spawn(name="worker")
    # 之后（或另一节点）：按名拿到 proxy
    worker = await Worker.resolve("worker")
    result = await worker.process("hello")
    await pul.shutdown()
```

匿名 Actor（不传 `name=`）只能通过 `spawn()` 返回的 `ActorRef` 访问。

---

## Ask 与 tell

| 模式 | 方法 | 适用场景 |
|------|------|----------|
| **请求–响应** | `await ref.ask(msg)` 或 `await proxy.method()` | 需要返回值。 |
| **发送即忘** | `await ref.tell(msg)` | 不需要回复；尽力而为投递。 |

有类型 proxy 时，方法调用相当于 **ask**（会返回结果）。只有在手头是 `ActorRef` 且不想等待回复时再用 **tell**。

---

## 下一步

- [集群组网](cluster_networking.zh.md) — 组建集群（Gossip / Head / Ray）
- [Actor 基础](../guide/actors.zh.md) — 模型与 API 深入
- [通信范式](../guide/communication_patterns.zh.md) — 流式、超时等
