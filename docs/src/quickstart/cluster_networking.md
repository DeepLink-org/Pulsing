# Cluster Networking (How to Use)

This page describes **how to form and use** a Pulsing cluster. For protocol and implementation details, see [Cluster Networking (Design)](../design/cluster-networking.md).

---

## Three modes

| Mode | What you configure | Best for |
|------|--------------------|----------|
| **Gossip + seed** | Bind address + optional seed addresses to join | Kubernetes, VMs, bare metal; no single point of failure |
| **Head node** | One node as head, others with head address | Simple ops; one fixed coordinator address |
| **Bootstrap (Ray/torchrun)** | `bootstrap(ray=..., torchrun=..., ...)`; no seeds | Already using Ray or torchrun; auto-detect or specify backend |

All modes use a **single HTTP/2 port** per node. No etcd, NATS, or Redis.

### When to use `init()` vs `bootstrap()`

- **`await pul.init(addr=..., seeds=...)`** or **`await pul.init()`** (standalone): use when **you** have the config (bind address, seed list, or no cluster). You control how the process joins.
- **`pul.bootstrap(ray=..., torchrun=..., ...)`**: use when the process is **already inside** a Ray or torchrun job and you want Pulsing to **auto-join** that environment (no seeds to pass; bootstrap discovers or uses Ray/torch.distributed).
- **Do not mix for the same “first init”**: pick one. If you already called `init()`, you do **not** need `bootstrap()`. Calling `bootstrap(on_ready=...)` after `init()` is safe (on_ready runs immediately because the system is already initialized).
- **Using only init (no bootstrap)** is fine: use `await pul.init(...)` for explicit config, `init_in_ray()` in a Ray process, and `init_in_torchrun()` in torchrun. All set the global system the same way. You give up a single "auto-detect" entry point (bootstrap tries both backends) and the optional "init in background then wait" pattern; otherwise behavior is the same.

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

## Mode 3: Bootstrap (Ray / torchrun)

Use the **bootstrap** API to form a Pulsing cluster when you are already in a Ray or torchrun environment. Omit `ray`/`torchrun` or pass both `True` to auto-detect (try Ray first, then torchrun).

### Usage

**Recommended: unified bootstrap**

```python
import pulsing as pul

# Auto-detect: try Ray then torchrun (default)
pul.bootstrap()
# Or block until ready
if pul.bootstrap(wait_timeout=30):
    system = pul.get_system()

# Only Ray
pul.bootstrap(ray=True, torchrun=False, wait_timeout=10)

# Only torchrun (e.g. launched with torchrun)
pul.bootstrap(ray=False, torchrun=True, on_ready=lambda s: print("ready:", s))
```

**Ray cluster: driver + workers**

In a Ray cluster, the driver can use `bootstrap(ray=True, ...)`. Every worker process must also initialize Pulsing; use `worker_process_setup_hook` so each worker runs `init_in_ray` on startup:

```python
import ray
from pulsing.integrations.ray import init_in_ray

ray.init(runtime_env={"worker_process_setup_hook": init_in_ray})

# Driver: bootstrap (calls init_in_ray under the hood when Ray is available)
import pulsing as pul
if pul.bootstrap(ray=True, torchrun=False, wait_timeout=30):
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

- You already run Ray or torchrun and want Pulsing on the same nodes as one cluster
- You want one API (`bootstrap`) to auto-detect or choose backend without managing seeds or head address
- You are okay depending on Ray KV or torch.distributed only for bootstrap; after that Pulsing uses its own gossip

### Limitations

- Ray path requires Ray and its internal KV; torchrun path requires `torch.distributed.init_process_group()` (e.g. torchrun)
- In a Ray cluster, every process must run Pulsing init (driver via `bootstrap(ray=True)` or `init_in_ray()`; workers via `worker_process_setup_hook`)
- One Pulsing cluster per Ray cluster (one KV key)

---

## Comparison and choice

| Criterion | Gossip + seed | Head node | Bootstrap (Ray/torchrun) |
|-----------|----------------|-----------|--------------------------|
| External deps | None | None | Ray and/or PyTorch |
| Single point of failure | No | Yes (head) | No |
| Config | addr + optional seeds | addr + head addr or head role | `bootstrap(ray=..., torchrun=...)` |
| Best environment | K8s, VMs, bare metal | One coordinator OK | Existing Ray or torchrun |
| Python init | `addr`, `seeds` | Via SystemConfig if exposed | `bootstrap()` or `init_in_ray` / `init_in_torchrun` |

**Suggested choice:**

- **Already on Ray or torchrun** → **Bootstrap** (`pul.bootstrap(ray=..., torchrun=...)`)
- **No SPOF, no Ray** → **Gossip + seed** (use a K8s Service as seed when on K8s)
- **One fixed coordinator, simple ops** → **Head node**

---

## Best practices

1. **Gossip + seed**: In K8s use a Service as seed; keep one port open for all nodes (actor + gossip).
2. **Head node**: Run head on a stable host/port; tune heartbeat timeout under load.
3. **Bootstrap**: Prefer `pul.bootstrap(ray=..., torchrun=..., on_ready=..., wait_timeout=...)`; in a Ray cluster set `worker_process_setup_hook=init_in_ray` for workers. Use `cleanup()` in tests if needed.
4. **Security**: For any mode, enable TLS (e.g. passphrase) for cluster traffic — see [Security](../guide/security.md).

---

## See also

- [Cluster Networking (Design)](../design/cluster-networking.md) — how the protocols and backends work
- [Remote Actors](../guide/remote_actors.md) — resolve, named actors, multi-node
- [Ray + Pulsing](migrate_from_ray.md) — use Pulsing as Ray's communication layer
