# 集群组网（设计）

本文说明 Pulsing 中**集群组网是如何实现的**。配置、用法与选型请参阅 [集群组网（快速开始）](../quickstart/cluster_networking.zh.md)。

---

## 概述

Pulsing 的**集群**由若干节点组成，节点之间共享：

- **成员关系**：哪些节点在集群中、是否存活
- **Actor 注册表**：哪些命名 Actor 存在、分别运行在哪些节点上

所有集群通信（成员、注册表与 Actor 消息）共用每个节点上的**单一 HTTP/2 端口**，且不依赖 etcd、NATS、Redis 等外部服务。

系统支持三种**命名后端**，由配置选择：

| 后端 | 角色 | 发现 / 注册表 |
|------|------|----------------|
| **GossipBackend** | 节点对等；seed 仅用于首次加入 | Gossip 循环 + SWIM，同一传输 |
| **HeadNodeBackend** | 一个 Head，其余 Worker | Head 持状态；Worker 注册并同步 |
| **Init in Ray** | 与 Gossip 相同 | Ray KV 提供首个 seed；之后 Gossip |

运行时根据 `SystemConfig` 选择后端：若设置 `head_addr` 或 `is_head_node` 则使用 `HeadNodeBackend`，否则使用 `GossipBackend`。Init-in-Ray 是一种启动方式，得到 seed 后仍走 Gossip。

---

## Gossip + seed：实现原理

- 每个节点有**绑定地址**。非首节点需配置一个或多个 **seed** 地址。
- 节点**加入**时向 seed 发送 Join 请求。若 seed 是负载均衡入口（如 Kubernetes Service），会多次探测，每次可能打到不同对端；收到 **Welcome** 后得到当前成员列表。
- 加入后节点运行 **Gossip 循环**：按固定间隔（如 200 ms）选择若干对端（如 `fanout` = 3），发送 **Gossip** 消息（部分成员视图、故障信息、可选命名 Actor 注册表），对端合并到本地状态。
- **SWIM** 故障检测在同一 HTTP/2 传输上运行：节点互相 Ping；疑似节点被标记并最终从视图中剔除。详见 [节点发现](node-discovery.zh.md) 与 SWIM 实现。
- 可选：节点周期性地**重新探测** seed 地址（如每 15 s），便于网络分区恢复，以及在 seed 为负载均衡端点时发现新节点。

因此：**Seed 仅用于获得初始成员列表**；之后由 Gossip 维持集群，无常驻主节点，任意节点都可作为新节点的 seed。

### 消息流（简化）

```
新节点 --Join--> Seed(s)
Seed(s)  --Welcome(members)--> 新节点
新节点合并成员，然后：
  loop: 选择对端 -> 发送 Gossip(partial_view, failures, actors) -> 合并响应
```

### 配置（实现）

Gossip 的节奏与行为由 `GossipConfig` 控制：`gossip_interval`、`fanout`、`seed_probe_count`、`seed_probe_interval`、`seed_rejoin_interval` 以及 SWIM 参数。完整列表与默认值见 [节点发现](node-discovery.zh.md)。

---

## Head 节点：实现原理

- 一个节点配置为 **Head**（`is_head_node`），其余为 **Worker**（设置 `head_addr`）。
- **Head** 在内存中维护权威的成员列表与 Actor 注册表，**不**跑 Gossip，仅提供 HTTP 接口供 Worker **注册**、**心跳**和**同步**（拉取成员/注册表）。
- **Worker** 启动时向 Head 注册，然后运行两个循环：
  - **心跳**：周期性地向 Head 发送心跳，Head 据此判断 Worker 存活。
  - **同步**：周期性地从 Head 拉取完整成员与注册表。
- Worker 上命名 Actor 的创建/退出会通知 Head，Head 更新注册表；Worker 在下次同步时得到新视图。

因此 Head 是**中心协调者**。Head 宕机期间，Worker 无法完成新发现或解析 Actor，直到 Head 恢复或改配新 Head。

### 后端选择

在 Rust 中，`ActorSystem::new(config)` 根据配置构建 `NamingBackend`：若 `config.head_addr.is_some() || config.is_head_node` 则创建 `HeadNodeBackend`，否则创建 `GossipBackend`。后端实现 `NamingBackend::join()`、Actor 注册/注销以及按名解析；系统在所有集群相关操作中使用该后端。

---

## Init in Ray / Bootstrap：实现原理

推荐在 Ray 或 torchrun 下使用统一入口 **`pulsing.bootstrap(ray=..., torchrun=..., on_ready=..., wait_timeout=...)`**；其后台会执行 `init_in_ray` 和/或 `init_in_torchrun`。

- Pulsing 运行在 **Ray** 集群内。每个使用 Pulsing 的进程调用 `init_in_ray()`（或 `async_init_in_ray()`）。
- **Seed 发现**使用 Ray 的 **internal KV**：
  - 第一个调用 `init_in_ray()` 的进程以**无 seed** 方式启动 Pulsing，得到本机地址后将该地址**写入** Ray KV 的固定 key（如 `pulsing:seed_addr`），成为初始 “seed” 节点。
  - 之后任意进程读取该 key，得到 seed 地址，并以该 seed 启动 Pulsing，从而加入同一集群。底层仍是 **Gossip + seed**，首个写入者的地址即 seed。
- 若两进程竞争写 key，实现上会对其中一个实例做 shutdown 并用胜出者地址重新 join（见 `pulsing.integrations.ray`）。

因此：**Ray KV 仅提供首个 seed**；之后集群行为与普通 Gossip 集群一致，没有单独的 “Ray 后端”，只是 Gossip 的另一种启动来源。

---

## 单端口与传输

所有集群协议（Gossip、Head 注册/同步、Actor RPC）共用同一 **HTTP/2 服务**与 **Http2Transport**。例如 `POST /cluster/gossip` 与 `POST /actor/{name}` 在同一端口多路复用。便于部署与防火墙配置。见 [HTTP2 传输](http2-transport.zh.md) 与 [节点发现](node-discovery.zh.md)。

---

## 相关文档

- [集群组网（快速开始）](../quickstart/cluster_networking.zh.md) — 如何使用与配置
- [节点发现](node-discovery.zh.md) — Gossip 协议与 seed 探测细节
- [架构](architecture.zh.md) — 系统组件与消息流
