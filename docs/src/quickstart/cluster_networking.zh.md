# 集群组网（如何使用）

本页说明**如何组建和使用** Pulsing 集群。协议与实现细节见 [集群组网（设计）](../design/cluster-networking.zh.md)。

---

## 三种方式

| 方式 | 你需要配置什么 | 适用场景 |
|------|----------------|----------|
| **Gossip + seed** | 绑定地址 + 可选 seed 地址以加入 | Kubernetes、VM、裸机；无单点故障 |
| **Head 节点** | 一个节点作 Head，其余填 Head 地址 | 运维简单；一个固定协调地址 |
| **Init in Ray** | 每个进程调用 `init_in_ray()`，无需 seeds | 已在用 Ray；自动发现 seed |

所有方式每节点**单一 HTTP/2 端口**，不依赖 etcd、NATS、Redis。

---

## 方式一：Gossip + seed

### 配置

**Python**

```python
import pulsing as pul

# 首节点
await pul.init(addr="0.0.0.0:8000")

# 后续节点：通过 seeds 加入
await pul.init(addr="0.0.0.0:8001", seeds=["192.168.1.10:8000"])
```

**Rust**

```rust
use pulsing_actor::prelude::*;
use std::net::SocketAddr;

// 首节点
let config = SystemConfig::with_addr("0.0.0.0:8000".parse()?);
let system = ActorSystem::new(config).await?;

// 后续节点
let config = SystemConfig::with_addr("0.0.0.0:8001".parse()?)
    .with_seeds(vec!["192.168.1.10:8000".parse()?]);
let system = ActorSystem::new(config).await?;
```

多个 seed（如 Kubernetes Service）时传入列表即可，节点会探测直到获得成员列表。

### Kubernetes

用 Service 名作为 seed，新 Pod 即可加入：

```python
await pul.init(addr="0.0.0.0:8080", seeds=["actor-cluster.default.svc.cluster.local:8080"])
```

### 何时选用

- 发现逻辑无单点故障
- 运行在 K8s、VM 或裸机，能提供至少一个地址（或 Service）作 seed
- 能接受成员关系的最终一致性（通常几百毫秒内收敛）

---

## 方式二：Head 节点

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

```python
import pulsing as pul

# Head 节点
await pul.init(addr="0.0.0.0:8000", is_head_node=True)

# Worker 节点
await pul.init(addr="0.0.0.0:8001", head_addr="192.168.1.10:8000")
```

也可使用 `SystemConfig.with_head_node()` / `.with_head_addr(addr)` 后传给 `ActorSystem.create(config, loop)` 做高级用法。

### Head 参数（Rust）

- **同步间隔**：Worker 从 Head 拉取的周期（默认 5s）
- **心跳间隔**：Worker 向 Head 发送心跳的周期（默认 10s）
- **心跳超时**：Head 将 Worker 判为死亡的时间（默认 30s）

### 何时选用

- 希望一个固定地址（Head）做防火墙与监控
- 可接受协调单点（Head 宕机期间无法新加入直到恢复）
- 希望以 Head 为成员/注册表的唯一真相源

---

## 方式三：Init in Ray

### 前置条件

- 已安装 Ray，且先执行 `ray.init()` 再调用 `init_in_ray()`
- 每个使用 Pulsing 的进程（driver 与 worker）都必须在该进程中调用 `init_in_ray()`

### 用法

```python
import ray
from pulsing.integrations.ray import init_in_ray

# 推荐：用 hook 让每个 worker 启动时执行 init_in_ray
ray.init(runtime_env={"worker_process_setup_hook": init_in_ray})

# driver 也必须初始化
init_in_ray()

# 按常规使用 Pulsing
import pulsing as pul
@pul.remote
class MyActor:
    def run(self): return "ok"

actor = await MyActor.spawn(name="my_actor")
```

**异步**（如 async Ray actor）：

```python
from pulsing.integrations.ray import async_init_in_ray
await async_init_in_ray()
```

**清理**（如测试）：

```python
from pulsing.integrations.ray import cleanup
cleanup()
```

### 何时选用

- 已在用 Ray，希望 Pulsing 在同一批节点上组成一个集群
- 希望每个进程一行代码完成组网，无需自己维护 seed 或 Head 地址
- 能接受仅在启动阶段依赖 Ray 的 KV；之后仅用 Pulsing 自己的 gossip

### 限制

- 依赖 Ray 及其 internal KV
- 每个进程都必须调用 `init_in_ray()`（driver 显式；worker 通过 hook）
- 一个 Ray 集群对应一个 Pulsing 集群（一个 KV key）

---

## 对比与选型

| 维度 | Gossip + seed | Head 节点 | Init in Ray |
|------|----------------|-----------|-------------|
| 外部依赖 | 无 | 无 | Ray |
| 单点故障 | 无 | 有（Head） | 无 |
| 配置 | addr + 可选 seeds | addr + Head 地址或 Head 角色 | 无（Ray KV） |
| 适用环境 | K8s、VM、裸机 | 可接受单一协调节点 | 已有 Ray 集群 |
| Python init() | `addr`、`seeds` | 通过 SystemConfig（若暴露） | `init_in_ray()` |

**选型建议：**

- **已有 Ray** → **Init in Ray**
- **不要单点且不用 Ray** → **Gossip + seed**（K8s 下用 Service 作 seed）
- **一个固定协调节点、运维简单** → **Head 节点**

---

## 最佳实践

1. **Gossip + seed**：K8s 下用 Service 作 seed；各节点开放同一端口（Actor + Gossip）。
2. **Head 节点**：Head 部署在稳定主机/端口；根据负载调整心跳超时。
3. **Init in Ray**：Driver 中调用 `init_in_ray()` 并设置 `worker_process_setup_hook`；测试中如需可调用 `cleanup()`。
4. **安全**：任意方式均可为集群流量开启 TLS（如 passphrase），见 [安全](../guide/security.zh.md)。

---

## 相关文档

- [集群组网（设计）](../design/cluster-networking.zh.md) — 协议与后端如何实现
- [远程 Actor](../guide/remote_actors.zh.md) — resolve、命名 Actor、多节点
- [Ray + Pulsing](migrate_from_ray.zh.md) — 用 Pulsing 作为 Ray 的通信层
