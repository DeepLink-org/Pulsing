---
template: home.html
title: Pulsing - 轻量级分布式 Actor 框架
description: 一个为构建可扩展 AI 系统设计的轻量级分布式 Actor 框架。零外部依赖，SWIM 协议发现，Python 优先设计。
hide: toc
---

<!-- 此内容被 home.html 模板隐藏，但可被搜索引擎索引 -->

# Pulsing

**Pulsing** 是一个为构建可扩展 AI 系统设计的轻量级分布式 Actor 框架。

## 核心特性

- **零外部依赖** - 纯 Rust + Tokio 实现，无需 etcd、NATS 或 Consul。
- **SWIM 协议发现** - 内置基于 Gossip 的节点发现和故障检测。
- **位置透明** - ActorRef 支持统一访问本地和远程 Actor。
- **流式消息** - 原生支持流式请求和响应。
- **Python 优先** - 通过 PyO3 提供完整的 Python API，支持 `@remote` 装饰器。
- **高性能** - 基于 Tokio 异步运行时，使用 HTTP/2 传输。
- **Ray 兼容** - 提供兼容 API，轻松从 Ray 迁移。

## 快速开始

```bash
# 安装
pip install maturin
maturin develop
```

```python
from pulsing.actor import init, shutdown, remote

@remote
class Calculator:
    def __init__(self, initial: int = 0):
        self.value = initial

    def add(self, n: int) -> int:
        self.value += n
        return self.value

async def main():
    await init()
    calc = await Calculator.spawn(initial=100)
    result = await calc.add(50)  # 150
    await shutdown()
```

## 使用场景

- **LLM 推理服务** - 构建可扩展的 LLM 推理后端，支持流式 Token 生成。
- **分布式计算** - 替代 Ray 用于轻量级分布式工作负载。
- **Kubernetes 原生** - 服务发现与 K8s Service IP 无缝配合。

## 社区

- [GitHub 仓库](https://github.com/reiase/pulsing)
- [问题追踪](https://github.com/reiase/pulsing/issues)
- [讨论区](https://github.com/reiase/pulsing/discussions)
