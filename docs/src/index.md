---
template: home.html
title: Pulsing - Lightweight Distributed Actor Framework
description: A lightweight distributed Actor framework for building scalable AI systems. Zero external dependencies, SWIM protocol discovery, Python-first design.
hide: toc
---

<!-- This content is hidden by the home.html template but indexed for search -->

# Pulsing

**Pulsing** is a lightweight distributed Actor framework for building scalable AI systems.

## Key Features

- **Zero External Dependencies** - Pure Rust + Tokio implementation. No etcd, NATS, or Consul required.
- **SWIM Protocol Discovery** - Built-in gossip-based node discovery and failure detection.
- **Location Transparent** - ActorRef supports unified access to local and remote actors.
- **Streaming Messages** - Native support for streaming requests and responses.
- **Python First** - Full Python API via PyO3 with `@remote` decorator.
- **High Performance** - Built on Tokio async runtime with HTTP/2 transport.
- **Ray Compatible** - Drop-in replacement API for easy migration from Ray.

## Quick Start

```bash
# Install
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

## Use Cases

- **LLM Inference Services** - Build scalable LLM inference backends with streaming token generation.
- **Distributed Computing** - Replace Ray for lightweight distributed workloads.
- **Kubernetes Native** - Service discovery works seamlessly with K8s Service IPs.

## Community

- [GitHub Repository](https://github.com/reiase/pulsing)
- [Issue Tracker](https://github.com/reiase/pulsing/issues)
- [Discussions](https://github.com/reiase/pulsing/discussions)
