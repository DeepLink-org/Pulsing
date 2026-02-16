---
template: home.html
title: Pulsing - 分布式 AI 系统的通信骨干
description: "Pulsing：分布式 AI 系统的通信骨干。Actor 运行时。流式优先。零依赖。内置发现。"
hide: toc
---

<!-- This content is hidden by the home.html template but indexed for search -->

# Pulsing

**分布式 AI 系统的通信骨干。**

Actor 运行时。流式优先。零依赖。内置发现。

用 Rust 构建、为 Python 设计的分布式 Actor 运行时。跨机器连接 AI Agent 和服务——不需要 Redis，不需要 etcd，不需要 YAML。

## 为什么选择 Pulsing？

<div class="grid cards" markdown>

-   :material-package-variant-closed:{ .lg .middle } **零依赖**

    ---

    纯 Rust + Tokio 实现。无需 etcd、NATS、Redis 或 Consul。只需 `pip install pulsing`。

-   :material-lightning-bolt:{ .lg .middle } **流式优先**

    ---

    原生流式支持，为 LLM token 生成和实时通信而设计。

-   :material-radar:{ .lg .middle } **内置发现**

    ---

    SWIM/Gossip 协议实现自动节点发现和故障检测。无需配置。

-   :material-language-python:{ .lg .middle } **Rust 构建，Python 设计**

    ---

    通过 PyO3 提供完整异步 Python API。`@remote` 装饰器将任意类变成分布式 Actor。

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

-   :material-swap-horizontal:{ .lg .middle } **与 Ray 协同**

    ---

    用 Pulsing 作为 Ray Actor 的通信层。流式、发现、跨集群调用——开箱即用。

    [:octicons-arrow-right-24: Ray + Pulsing](quickstart/migrate_from_ray.zh.md)

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
| Pulsing 是什么 / 适合谁？ | [概述](overview.zh.md) |
| 理解 Actor 模型 | [Actor 基础](guide/actors.zh.md) |
| 构建集群 | [远程 Actor](guide/remote_actors.zh.md) |
| 运维系统 | [CLI 运维](guide/operations.zh.md) |
| 架构与设计 | [架构与设计](design/architecture.zh.md) |
| API 详情 | [API 概述](api/overview.zh.md) |
| 完整 API 契约 | [完整参考](api_reference.zh.md) |

---

## 社区

- [GitHub 仓库](https://github.com/DeepLink-org/pulsing)
- [Issue 追踪](https://github.com/DeepLink-org/pulsing/issues)
- [讨论区](https://github.com/DeepLink-org/pulsing/discussions)
