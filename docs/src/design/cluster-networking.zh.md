# 集群组网

本文说明如何组建和运维 Pulsing 集群。Pulsing 支持三种不同的分布式组网方式，各有适用场景与取舍。

## 概述

Pulsing 的**集群**由若干节点组成，节点之间共享：

- **成员关系**：哪些节点在集群中、是否存活
- **Actor 注册表**：哪些命名 Actor 存在、分别运行在哪些节点上

所有集群通信（成员、注册表与 Actor 消息）共用每个节点上的**单一 HTTP/2 端口**，且不依赖 etcd、NATS、Redis 等外部服务。

三种**组网方式**如下：

| 方式 | 说明 | 适用场景 |
|------|------|----------|
| **1. Gossip + Seed 节点** | 节点通过 Gossip 协议互相发现；你只需提供若干 seed 地址即可加入集群。 | Kubernetes、裸机、云 VM；弹性扩缩；无单点故障。 |
| **2. Head 节点** | 指定一个节点为 Head，其余为 Worker；Worker 向 Head 注册并拉取成员/注册表。 | 部署简单、可接受单一协调节点的环境。 |
| **3. 借助 Ray 组网（init in Ray）** | 在 Ray 集群内运行 Pulsing，用 Ray 的 internal KV 发现首个 seed，再走 Gossip。 | 已有 Ray 用户；与 Ray 任务同机部署。 |

下文分别说明三种方式，最后对比并给出选型建议。

---

## 方式一：Gossip + Seed 节点组网

### 原理

- 每个节点配置**本机监听地址**，非首节点还需配置一个或多个 **seed** 地址。
- 带 seed 的节点**加入**时向 seed 发送 Join 请求（若 seed 是负载均衡入口，会多次探测以发现多个对端），收到 **Welcome** 后得到当前成员列表。
- 加入后节点运行 **Gossip 循环**：周期性与部分对端交换成员、故障信息和 Actor 注册表（详见 [节点发现](node-discovery.zh.md)）。
- **SWIM** 故障检测在同一传输上运行，疑似/死亡节点会从视图中剔除。
- 可选：节点周期性地**重新探测** seed 地址（如每 15s），便于网络分区恢复，以及在 seed 为负载均衡端点（如 K8s Service）时发现新节点。

因此：**Seed 仅用于首次加入**；之后由 Gossip 维持集群，没有常驻“主节点”，任意节点都可作为新节点的 seed。

### 配置

**Rust**

```rust
use pulsing_actor::prelude::*;
use std::net::SocketAddr;

// 首节点（不配 seeds，仅绑定地址）
let config = SystemConfig::with_addr("0.0.0.0:8000".parse()?);
let system = ActorSystem::new(config).await?;

// 后续节点：通过 seeds 加入
let config = SystemConfig::with_addr("0.0.0.0:8001".parse()?)
    .with_seeds(vec!["192.168.1.10:8000".parse()?]);
let system = ActorSystem::new(config).await?;
```

**Python**

```python
import pulsing as pul

# 首节点
await pul.init(addr="0.0.0.0:8000")

# 后续节点：通过 seeds 加入
await pul.init(addr="0.0.0.0:8001", seeds=["192.168.1.10:8000"])
```

若有多个 seed（如高可用或 K8s Service），传入列表即可；节点会探测直到获得成员列表。

### 与 Kubernetes 配合

当 seed 是 **Kubernetes Service**（ClusterIP 或 headless）时，新 Pod 将 Service 名作为 seed。平台负载均衡可能把每次探测打到不同 Pod，新节点几次探测即可发现多个成员。推荐参数见 [节点发现](node-discovery.zh.md) 中的 `seed_probe_count` 与 `seed_rejoin_interval`。

```yaml
# 示例：Pod 使用 Service 作为 seed
# seed_nodes: ["actor-cluster.default.svc.cluster.local:8080"]
```

### 何时选用

- 希望**发现逻辑无单点故障**。
- 运行在 **Kubernetes、裸机或云 VM**，能提供至少一个稳定地址（或 Service）作为 seed。
- 能接受成员与 Actor 注册表的**最终一致性**（通常几百毫秒内收敛）。

### 小结

- **Seed**：仅用于首次加入，之后由 Gossip 维持集群。
- **单端口**：Actor RPC 与 Gossip 共用同一 HTTP/2 服务。
- **无外部存储**：不依赖 etcd/NATS/Redis。

---

## 方式二：Head 节点组网

### 原理

- 指定一个节点为 **Head**，其余为 **Worker**。
- **Head** 在内存中维护权威的成员列表与 Actor 注册表，不跑 Gossip，只接受 Worker 的注册与心跳。
- **Worker** 启动时连接 Head 地址完成注册，并运行**心跳**与**同步**循环（按间隔从 Head 拉取成员/注册表）。
- Worker 上的 Actor 注册/注销会上报给 Head；Head 更新状态后，Worker 在下次同步时得到一致视图。

因此：Head 是**中心协调者**。Head 宕机期间，Worker 无法完成新发现或解析 Actor，直到 Head 恢复（或你改为指向新 Head）。

### 配置

**Rust**

```rust
use pulsing_actor::prelude::*;
use std::net::SocketAddr;

// Head 节点
let config = SystemConfig::with_addr("0.0.0.0:8000".parse()?)
    .with_head_node();
let system = ActorSystem::new(config).await?;

// Worker 节点
let head_addr: SocketAddr = "192.168.1.10:8000".parse()?;
let config = SystemConfig::with_addr("0.0.0.0:8001".parse()?)
    .with_head_addr(head_addr);
let system = ActorSystem::new(config).await?;
```

**Python**

Head 模式在 Rust 侧通过 `SystemConfig` 的 `with_head_node()` / `with_head_addr()` 支持。当前 Python 高层 API `init(addr=..., seeds=...)` 仅支持 **Gossip + seed**。若要在 Python 中使用 Head 模式，需通过 Python 绑定的 `SystemConfig`（若已暴露 head 相关接口）构建配置并传给 `ActorSystem.create(config, loop)`，具体以当前版本 API 为准。

### Head 相关参数

Head 后端使用少量定时参数（Rust 中通过 `HeadNodeConfig` 配置）：

- **同步间隔**：Worker 从 Head 拉取成员/注册表的周期（默认 5s）。
- **心跳间隔**：Worker 向 Head 发送心跳的周期（默认 10s）。
- **心跳超时**：超过多久未收到心跳则视为 Worker 死亡（默认 30s）。

调整这些参数可影响故障 Worker 从视图中剔除的速度。

### 何时选用

- 希望**运维简单**：只需固定一个 Head 地址做防火墙与监控。
- 可接受**协调层面的单点**（Head 宕机期间无法新加入或更新视图，直到恢复）。
- 希望成员/注册表以 Head 为**强一致**来源，Worker 每次同步后与其一致。

### 与 Gossip 对比

| 维度 | Gossip + seed | Head 节点 |
|------|----------------|-----------|
| 发现方式 | 去中心化；seed 仅用于加入，之后 Gossip | 中心化；Worker 只与 Head 通信 |
| “特殊”节点故障 | 无单点；任意节点都可作 seed | Head 宕机则无法新加入/更新，直到恢复 |
| 一致性 | 最终一致（有传播延迟） | Head 为唯一真相源；Worker 与 Head 最终一致 |
| 配置复杂度 | 至少一个可达 seed 地址 | 每个 Worker 需配置 Head 地址 |

---

## 方式三：借助 Ray 组网（init in Ray）

### 原理

- 你已有 **Ray 集群**，并通过 `ray.init(...)` 等方式拉起任务。
- 使用 **`pulsing.ray.init_in_ray()`**，让**每个进程**（driver 与需要使用 Pulsing 的 worker）都启动一套 Pulsing 并**加入同一个 Pulsing 集群**。
- 首个 seed 的发现依赖 **Ray 的 internal KV**：
  - 第一个调用 `init_in_ray()` 的进程以“无 seed”方式启动 Pulsing，得到本机地址后，将该地址**写入** Ray KV 的固定 key（如 `pulsing:seed_addr`）。
  - 之后任意进程调用 `init_in_ray()` 时**读取**该 key，得到 seed 地址，并以该 seed 启动 Pulsing，从而加入已有集群。
- 底层仍是 **Gossip + seed**：只是 seed 由 Ray KV 提供，而不是你在配置里写死。因此每个 Ray 集群（或每个 KV 命名空间）对应一个 Pulsing 集群，且无需额外 etcd/NATS。

这就是“**init in Ray**”：Pulsing 的**组网**仍用自己的 Gossip，但**部署与发现**借助 Ray 运行时完成。

### 配置与用法

**前置条件**

- 需安装 Ray，且必须先 **`ray.init()`**，再调用 `init_in_ray()`。
- 每个要使用 Pulsing 的进程（driver 与 worker）都必须在该进程中调用 `init_in_ray()`（或异步版本）。

**基本用法**

```python
import ray
from pulsing.ray import init_in_ray

# 方式 A：将 init_in_ray 设为 worker_process_setup_hook（推荐）
# 这样每个 worker 进程启动时都会执行 init_in_ray。
ray.init(runtime_env={"worker_process_setup_hook": init_in_ray})

# driver 进程也需要初始化 Pulsing
init_in_ray()

# 之后按常规使用 Pulsing
import pulsing as pul
@pul.remote
class MyActor:
    def run(self): return "ok"

actor = await MyActor.spawn(name="my_actor")  # 可落在任意节点
```

**异步版本（如在 async Ray actor 中）**

```python
from pulsing.ray import async_init_in_ray

# 在 async Ray actor 或 async 上下文中
await async_init_in_ray()
```

**清理（可选）**

若在 teardown（如测试）时希望清除 Ray KV 中的 seed key：

```python
from pulsing.ray import cleanup
cleanup()
```

### Seed 如何确定

- **先写先得**：第一个成功写入 KV key 的 `init_in_ray()` 调用者成为“seed 节点”，其地址被写入。
- 若 key 已存在，进程读取该地址并作为 `seeds=[...]` 启动，从而加入已有集群。
- 极少数并发下可能发生竞争（两个进程都以为没有 seed 并写入），实现上会对其中一个实例做 shutdown 并用胜出者的地址重新 join，详见 `pulsing.ray` 源码。

因此**无需手动配置 seed**：Ray KV 提供首个 seed，之后集群按 **Gossip + seed** 运行，该节点为初始联络点。

### 何时选用

- 已在用 **Ray**（做调度或其他任务），希望 Pulsing Actor 跑在同一批节点并组成一个集群。
- 希望**每个进程一行代码**完成组网（`init_in_ray()`），而不自己维护 seed 列表或 Head 地址。
- 可以接受在**启动阶段**依赖 Ray 运行时（及其 KV）；启动后仅使用 Pulsing 自己的 HTTP/2 + Gossip。

### 限制

- **依赖 Ray**：`init_in_ray()` 依赖 Ray 及其 internal KV，未使用 Ray 时不要选此方式。
- **进程模型**：每个使用 Pulsing 的进程都必须调用 `init_in_ray()`（或 `async_init_in_ray()`）。通过 hook 可保证 worker 调用；driver 需显式调用。
- **一个 Ray 集群对应一个 Pulsing 集群**：KV key 在 Ray 集群内全局唯一，因此该集群内所有 `init_in_ray()` 调用者都会加入同一个 Pulsing 集群。

---

## 三种方式对比与选型

| 维度 | Gossip + seed | Head 节点 | Init in Ray |
|------|----------------|-----------|-------------|
| **外部依赖** | 无 | 无 | Ray（及其 KV） |
| **单点故障** | 无 | 有（Head） | 无（启动后与 Gossip 一致） |
| **配置** | 绑定地址 + 可选 seeds | 绑定地址 + Head 地址或 Head 角色 | 无（由 Ray KV 提供 seed） |
| **适用环境** | K8s、VM、裸机 | 可接受单一协调节点 | 已有 Ray 集群 |
| **一致性** | 最终一致 | Head 权威，再最终一致 | 与 Gossip + seed 相同 |
| **Python init()** | 支持（`addr`、`seeds`） | 需通过 SystemConfig（若暴露） | 使用 `init_in_ray()` |

**选型建议：**

- **已有 Ray** → 用 **init in Ray**，配置最少、自动发现 seed。
- **不要单点且不用 Ray** → 用 **Gossip + seed**（K8s 下用 Service 作 seed）。
- **希望一个固定协调节点、运维简单** → 用 **Head 节点**。

---

## 最佳实践

1. **Gossip + seed**
   - 在 Kubernetes 中用 **Service**（或多个 seed）作 seed；合理设置 `seed_probe_count`（如 3）和 `seed_rejoin_interval`（如 15s），便于新节点加入与分区恢复。
   - 保证各节点**同一端口**开放（Actor + Gossip），避免多余防火墙规则。

2. **Head 节点**
   - Head 部署在**稳定主机/端口**，可按需前面挂负载均衡做 HA（进程可换，地址不变）。
   - 根据负载调整**心跳超时**，避免 Worker 在压力下被误判为死亡。

3. **Init in Ray**
   - **Driver 中调用 `init_in_ray()`**，并设置 **`worker_process_setup_hook`**，确保每个 worker 进程都加入同一 Pulsing 集群。
   - 测试场景下若希望 KV 干净，可在 Ray 集群 teardown 时调用 **`cleanup()`**。

4. **安全**
   - 任意方式下均可开启 **TLS**（如基于 passphrase 的证书），对 Actor 与集群流量加密和认证，见 [安全](../guide/security.zh.md)。

---

## 相关文档

- [节点发现](node-discovery.zh.md) — Gossip 协议与 seed 探测细节。
- [架构](architecture.zh.md) — 系统组件与消息流。
- [从 Ray 迁移](../quickstart/migrate_from_ray.zh.md) — Ray 到 Pulsing 的 API 映射。
