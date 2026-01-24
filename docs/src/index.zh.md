---
template: home.html
title: Pulsing - 轻量级分布式 Actor 框架
description: 基于 Rust 和 Tokio 构建的轻量级分布式 Actor 框架，专为 AI 系统设计。零外部依赖，内置服务发现，Python 优先。
hide: toc
---

<!-- This content is hidden by the home.html template but indexed for search -->

# Pulsing

基于 Rust 和 Tokio 构建的**轻量级分布式 Actor 框架**。

## 为什么选择 Pulsing？

<div class="grid cards" markdown>

-   :material-package-variant-closed:{ .lg .middle } **零外部依赖**

    ---

    纯 Rust + Tokio 实现。无需 etcd、NATS、Redis 或 Consul。

-   :material-radar:{ .lg .middle } **内置集群发现**

    ---

    SWIM/Gossip 协议实现自动节点发现和故障检测。

-   :material-lightning-bolt:{ .lg .middle } **高性能**

    ---

    异步运行时 + HTTP/2 传输 + 原生流式支持。

-   :material-language-python:{ .lg .middle } **Python 优先**

    ---

    通过 PyO3 提供完整 Python API。`@remote` 装饰器将任意类变成 Actor。

</div>

---

## 你可以构建什么？

<div class="grid cards" markdown>

-   :material-robot:{ .lg .middle } **LLM 推理服务**

    ---

    可扩展的推理后端，支持流式 token 生成。开箱即用的 OpenAI 兼容 API。

    [:octicons-arrow-right-24: LLM 推理](quickstart/llm_inference.zh.md)

-   :material-account-group:{ .lg .middle } **分布式 Agent**

    ---

    原生集成 AutoGen 和 LangGraph。跨机器分布你的 Agent。

    [:octicons-arrow-right-24: 分布式 Agent](quickstart/agent.zh.md)

-   :material-swap-horizontal:{ .lg .middle } **替代 Ray**

    ---

    兼容 API，一行导入即可从 Ray 迁移。

    [:octicons-arrow-right-24: 从 Ray 迁移](quickstart/migrate_from_ray.zh.md)

</div>

---

## 快速开始

```bash
pip install pulsing
```

```python
import asyncio
import pulsing as pul

@pul.remote
class Counter:
    def __init__(self, value=0):
        self.value = value

    def inc(self):
        self.value += 1
        return self.value

async def main():
    await pul.init()
    counter = await Counter.spawn(value=0)
    print(await counter.inc())  # 1
    print(await counter.inc())  # 2
    await pul.shutdown()

asyncio.run(main())
```

[:octicons-arrow-right-24: 快速开始](quickstart/index.zh.md){ .md-button }

---

## 深入了解

| 目标 | 链接 |
|------|------|
| 理解 Actor 模型 | [指南：Actor](guide/actors.zh.md) |
| 构建集群 | [指南：远程 Actor](guide/remote_actors.zh.md) |
| 运维系统 | [指南：CLI 操作](guide/operations.zh.md) |
| 深入设计 | [设计文档](design/architecture.md) |
| API 详情 | [API 参考](api_reference.md) |

---

## 社区

- [GitHub 仓库](https://github.com/reiase/pulsing)
- [Issue 追踪](https://github.com/reiase/pulsing/issues)
- [讨论区](https://github.com/reiase/pulsing/discussions)
