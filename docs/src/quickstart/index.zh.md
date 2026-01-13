# 快速开始

**5 分钟**让 Pulsing 跑起来。

## 安装

```bash
pip install pulsing
```

---

## 第一个 Actor

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

`@remote` 装饰器将任意 Python 类变成分布式 Actor。

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

-   :material-swap-horizontal:{ .lg .middle } **从 Ray 迁移**

    ---

    一行导入替换 Ray。零外部依赖。

    [:octicons-arrow-right-24: ~5 分钟](migrate_from_ray.zh.md)

</div>

---

## 深入了解

| 目标 | 链接 |
|------|------|
| 理解 Actor 模型 | [指南：Actor](../guide/actors.zh.md) |
| 构建集群 | [指南：远程 Actor](../guide/remote_actors.zh.md) |
| 运维系统 | [指南：运维操作](../guide/operations.zh.md) |
