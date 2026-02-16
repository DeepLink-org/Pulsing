# Pulsing Overview

## What is Pulsing?

**Pulsing: Backbone for distributed AI systems.**

Pulsing is a distributed actor runtime built in Rust, designed for Python. Actor runtime. Streaming-first. Zero dependencies. Built-in discovery.

In one sentence: turn any Python class into a distributed Actor with `@remote` — no etcd, NATS, or Redis required. Same API for local and remote, with native streaming support.

---

## What Can You Do with Pulsing?

| Use case | What you get |
|----------|----------------|
| **LLM inference services** | Scalable backends with streaming, OpenAI-compatible API, and optional vLLM/Transformers workers. |
| **Distributed agents** | Multi-agent systems with native integration for AutoGen and LangGraph; same code runs locally or across machines. |
| **Enhance Ray communication** | Add streaming, actor discovery, and cross-cluster calls to Ray actors via `pul.mount()`. Use Ray for scheduling, Pulsing for communication. |
| **Custom distributed apps** | Build services and workers that discover each other via built-in gossip or a head node, over a single HTTP/2 port. |

---

## Who Is It For?

| Role | Benefit |
|------|---------|
| **AI / ML application developers** | One-line scaling: add `addr` and `seeds` (or use init-in-Ray) to run agents and inference across nodes without learning a new paradigm. |
| **Distributed systems engineers** | Zero external coordination stores; built-in SWIM/gossip and optional head-node topology; single-port networking. |
| **Ray users** | Use Pulsing as a communication layer alongside Ray: `pul.mount()` bridges Ray actors onto the Pulsing network for streaming, discovery, and cross-cluster calls. |

You don't need to be a distributed systems expert to get value — the API is designed to stay simple from single process to multi-node.

---

## Design Principles

- **Zero external dependencies** — Pure Rust core + Tokio; no etcd, NATS, or Redis. Cluster discovery uses built-in gossip or an optional head node.
- **Location transparency** — Same API for local and remote actors: `await actor.method()` whether the actor is on this process or another machine.
- **Python first** — `@pul.remote` turns a class into an Actor; `spawn()` and `resolve()` for creation and discovery; native async/await and streaming.
- **Single port** — Actor RPC and cluster protocol share one HTTP/2 port per node, simplifying deployment and firewalls.

---

## Next Steps

- **[Quick Start](quickstart/index.md)** — Run your first Actor in minutes, then go stateful and distributed.
- **[Ray + Pulsing](quickstart/migrate_from_ray.md)** — Use Pulsing as Ray's communication layer, or use the standalone API.
