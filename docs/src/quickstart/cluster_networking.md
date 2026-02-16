# Cluster Networking (How to Use)

This page describes **how to form and use** a Pulsing cluster. For protocol and implementation details, see [Cluster Networking (Design)](../design/cluster-networking.md).

---

## Three modes

| Mode | What you configure | Best for |
|------|--------------------|----------|
| **Gossip + seed** | Bind address + optional seed addresses to join | Kubernetes, VMs, bare metal; no single point of failure |
| **Head node** | One node as head, others with head address | Simple ops; one fixed coordinator address |
| **Init in Ray** | `init_in_ray()` in each process; no seeds | Already using Ray; automatic seed discovery |

All modes use a **single HTTP/2 port** per node. No etcd, NATS, or Redis.

---

## Mode 1: Gossip + seed

### Configuration

**Python**

```python
import pulsing as pul

# First node
await pul.init(addr="0.0.0.0:8000")

# Later nodes — join via seeds
await pul.init(addr="0.0.0.0:8001", seeds=["192.168.1.10:8000"])
```

**Rust**

```rust
use pulsing_actor::prelude::*;
use std::net::SocketAddr;

// First node
let config = SystemConfig::with_addr("0.0.0.0:8000".parse()?);
let system = ActorSystem::new(config).await?;

// Later nodes
let config = SystemConfig::with_addr("0.0.0.0:8001".parse()?)
    .with_seeds(vec!["192.168.1.10:8000".parse()?]);
let system = ActorSystem::new(config).await?;
```

With multiple seeds (e.g. a Kubernetes Service), pass a list; the node probes until it gets a member list.

### Kubernetes

Use the Service name as the seed so new pods can join:

```python
await pul.init(addr="0.0.0.0:8080", seeds=["actor-cluster.default.svc.cluster.local:8080"])
```

### When to use

- No single point of failure for discovery
- You run on K8s, VMs, or bare metal and can expose at least one address (or Service) as seed
- Eventual consistency of membership is acceptable (typically hundreds of ms)

---

## Mode 2: Head node

### Configuration

**Rust**

```rust
use pulsing_actor::prelude::*;
use std::net::SocketAddr;

// Head node
let config = SystemConfig::with_addr("0.0.0.0:8000".parse()?)
    .with_head_node();
let system = ActorSystem::new(config).await?;

// Worker nodes
let head_addr: SocketAddr = "192.168.1.10:8000".parse()?;
let config = SystemConfig::with_addr("0.0.0.0:8001".parse()?)
    .with_head_addr(head_addr);
let system = ActorSystem::new(config).await?;
```

**Python**

```python
import pulsing as pul

# Head node
await pul.init(addr="0.0.0.0:8000", is_head_node=True)

# Worker nodes
await pul.init(addr="0.0.0.0:8001", head_addr="192.168.1.10:8000")
```

You can also use `SystemConfig.with_head_node()` / `.with_head_addr(addr)` and pass the config to `ActorSystem.create(config, loop)` for advanced use.

### Head parameters (Rust)

- **Sync interval**: how often workers pull from head (default 5s)
- **Heartbeat interval**: worker → head (default 10s)
- **Heartbeat timeout**: head marks worker dead after (default 30s)

### When to use

- One fixed address (the head) for firewalls and monitoring
- You accept a single point of failure for coordination until head recovers
- You want the head as the single source of truth for membership/registry

---

## Mode 3: Init in Ray

### Requirements

- Ray installed and `ray.init()` called before `init_in_ray()`
- Every process that uses Pulsing (driver and workers) must call `init_in_ray()` in that process

### Usage

```python
import ray
from pulsing.integrations.ray import init_in_ray

# Recommended: hook so every worker runs init_in_ray at startup
ray.init(runtime_env={"worker_process_setup_hook": init_in_ray})

# Driver must also init
init_in_ray()

# Use Pulsing as usual
import pulsing as pul
@pul.remote
class MyActor:
    def run(self): return "ok"

actor = await MyActor.spawn(name="my_actor")
```

**Async** (e.g. async Ray actors):

```python
from pulsing.integrations.ray import async_init_in_ray
await async_init_in_ray()
```

**Cleanup** (e.g. tests):

```python
from pulsing.integrations.ray import cleanup
cleanup()
```

### When to use

- You already run Ray and want Pulsing on the same nodes as one cluster
- You want one-line cluster formation per process without managing seeds or head address
- You are okay depending on Ray’s KV only for bootstrap; after that Pulsing uses its own gossip

### Limitations

- Requires Ray and its internal KV
- Every process must call `init_in_ray()` (driver explicitly; workers via hook)
- One Pulsing cluster per Ray cluster (one KV key)

---

## Comparison and choice

| Criterion | Gossip + seed | Head node | Init in Ray |
|-----------|----------------|-----------|-------------|
| External deps | None | None | Ray |
| Single point of failure | No | Yes (head) | No |
| Config | addr + optional seeds | addr + head addr or head role | None (Ray KV) |
| Best environment | K8s, VMs, bare metal | One coordinator OK | Existing Ray cluster |
| Python `init()` | `addr`, `seeds` | Via SystemConfig if exposed | `init_in_ray()` |

**Suggested choice:**

- **Already on Ray** → **Init in Ray**
- **No SPOF, no Ray** → **Gossip + seed** (use a K8s Service as seed when on K8s)
- **One fixed coordinator, simple ops** → **Head node**

---

## Best practices

1. **Gossip + seed**: In K8s use a Service as seed; keep one port open for all nodes (actor + gossip).
2. **Head node**: Run head on a stable host/port; tune heartbeat timeout under load.
3. **Init in Ray**: Call `init_in_ray()` in the driver and set `worker_process_setup_hook`; use `cleanup()` in tests if needed.
4. **Security**: For any mode, enable TLS (e.g. passphrase) for cluster traffic — see [Security](../guide/security.md).

---

## See also

- [Cluster Networking (Design)](../design/cluster-networking.md) — how the protocols and backends work
- [Remote Actors](../guide/remote_actors.md) — resolve, named actors, multi-node
- [Ray + Pulsing](migrate_from_ray.md) — use Pulsing as Ray's communication layer
