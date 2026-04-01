# Pulsing 概述

## 什么是 Pulsing？

**Pulsing：分布式 AI 系统的通信骨干。**

Pulsing 是一个用 Rust 构建、为 Python 设计的分布式 Actor 运行时。流式优先。零依赖。内置发现。

一句话：用 `@remote` 把任意 Python 类变成分布式 Actor，无需 etcd、NATS 或 Redis。本地和远程使用同一套 API，原生支持流式通信。

---

## 你能用 Pulsing 做什么？

| 场景 | 你能得到什么 |
|------|----------------|
| **LLM 推理服务** | 可扩展的推理后端、流式输出、OpenAI 兼容 API，以及可选的 vLLM/Transformers Worker。 |
| **分布式 Agent** | 多智能体系统，原生集成 AutoGen 与 LangGraph；同一套代码可在本机或跨机运行。 |
| **增强 Ray 通信** | 通过 `pul.mount()` 为 Ray Actor 增加流式、发现和跨集群调用能力。Ray 负责调度，Pulsing 负责通信。 |
| **带资源约束的子进程执行** | 保持 `subprocess` 兼容调用方式，并在传入 `resources` 时按需切到 Pulsing 后端。 |
| **自定义分布式应用** | 通过内置 Gossip 或 Head 节点组网，单端口 HTTP/2，构建服务与 Worker。 |

---

## 适合谁用？

| 角色 | 收益 |
|------|------|
| **AI / ML 应用开发者** | 一行级扩展：加上 `addr` 和 `seeds`（或用 init-in-Ray），即可在多节点跑 Agent 与推理，无需学习新范式。 |
| **分布式系统工程师** | 零外部协调存储；内置 SWIM/Gossip 与可选 Head 拓扑；单端口组网。 |
| **Ray 用户** | 用 Pulsing 作为 Ray 的通信层：`pul.mount()` 将 Ray Actor 接入 Pulsing 网络，获得流式、发现和跨集群调用能力。 |

你不需要成为分布式系统专家也能用好 —— 从单进程到多节点，API 保持简洁。

---

## 设计理念

- **零外部依赖** — 核心纯 Rust + Tokio；不依赖 etcd、NATS、Redis。集群发现采用内置 Gossip 或可选 Head 节点。
- **位置透明** — 本地与远程 Actor 同一套 API：`await actor.method()` 无论 Actor 在本进程还是远程。
- **Python 优先** — `@pul.remote` 将类变成 Actor；`spawn()` / `resolve()` 用于创建与发现；原生 async/await 与流式。
- **渐进式接入** — `pulsing.subprocess` 保持标准库调用风格，只在需要时显式切换到 Pulsing 后端。
- **单端口** — 每节点一个 HTTP/2 端口同时承载 Actor RPC 与集群协议，便于部署与防火墙配置。

---

## 下一步

- **[快速开始](quickstart/index.zh.md)** — 几分钟内跑起第一个 Actor，再进阶到有状态与分布式。
- **[Ray + Pulsing](quickstart/migrate_from_ray.zh.md)** — 用 Pulsing 作为 Ray 的通信层，或使用 Pulsing 独立 API。
- **[子进程示例](examples/subprocess.zh.md)** — 通过兼容 `subprocess` 的 API 执行命令，并可选启用资源调度。
