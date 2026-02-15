# Cluster Networking

This document describes how to form and operate a Pulsing cluster. Pulsing supports three distinct ways to build a distributed network of nodes, each with different trade-offs and use cases.

## Overview

A Pulsing **cluster** is a set of nodes that share:

- **Membership**: who is in the cluster and whether they are alive
- **Actor registry**: which named actors exist and on which node(s) they run

All cluster traffic (membership, registry, and actor messages) uses a single HTTP/2 port per node. No external services like etcd, NATS, or Redis are required.

The three supported **networking modes** are:

| Mode | Description | Best for |
|------|-------------|----------|
| **1. Gossip + seed** | Nodes discover each other via a gossip protocol; you provide one or more seed addresses to join. | Kubernetes, bare metal, cloud VMs; flexible scaling; no single point of failure. |
| **2. Head node** | One node is the head; workers register with the head and get membership/registry from it. | Simple deployments; environments where a single coordinator is acceptable. |
| **3. Init in Ray** | Pulsing runs inside a Ray cluster; Ray’s internal KV store is used to discover the first seed, then gossip is used. | Existing Ray users; running Pulsing alongside Ray jobs. |

The rest of this document explains each mode in detail, then compares them and gives practical guidance.

---

## Mode 1: Gossip + Seed Nodes

### How it works

- You configure each node with a **bind address** and (for non-first nodes) one or more **seed** addresses.
- A node with seeds **joins** by sending a join request to each seed (and, if the seed is behind a load balancer, possibly multiple times to discover different peers). It receives a **Welcome** with the current member list.
- Once in the cluster, nodes run a **gossip loop**: they periodically exchange membership, failure information, and actor registry with a subset of peers (see [Node Discovery](node-discovery.md)).
- **SWIM**-style failure detection runs over the same transport; suspected/dead nodes are removed from the view.
- Optionally, nodes periodically **re-probe** the seed address(es) (e.g. every 15s). This helps with network partition recovery and discovering new nodes when the seed is a load-balanced endpoint (e.g. a Kubernetes Service).

So: **seed nodes are only used to get an initial member list**; after that, the cluster is maintained by gossip. There is no permanent “master”; any node can serve as a seed for newcomers.

### Configuration

**Rust**

```rust
use pulsing_actor::prelude::*;
use std::net::SocketAddr;

// First node (no seeds) – can still bind for incoming connections
let config = SystemConfig::with_addr("0.0.0.0:8000".parse()?);
let system = ActorSystem::new(config).await?;

// Later nodes – join via seeds
let config = SystemConfig::with_addr("0.0.0.0:8001".parse()?)
    .with_seeds(vec!["192.168.1.10:8000".parse()?]);
let system = ActorSystem::new(config).await?;
```

**Python**

```python
import pulsing as pul

# First node
await pul.init(addr="0.0.0.0:8000")

# Later nodes – join via seeds
await pul.init(addr="0.0.0.0:8001", seeds=["192.168.1.10:8000"])
```

With multiple seeds (e.g. for HA or when the seed is a K8s Service), pass a list; the node will probe until it gets a member list.

### Kubernetes-friendly usage

When the seed is a **Kubernetes Service** (ClusterIP or headless), new pods use the Service name as the seed. The platform’s load balancer may send each probe to a different pod, so the new node can discover several members in a few probes. See [Node Discovery](node-discovery.md) for the recommended `seed_probe_count` and `seed_rejoin_interval` behavior.

```yaml
# Example: pods use the service as seed
# seed_nodes: ["actor-cluster.default.svc.cluster.local:8080"]
```

### When to use

- You want **no single point of failure** for discovery.
- You run on **Kubernetes**, **bare metal**, or **cloud VMs** and can expose one or more stable addresses (or a Service) as seeds.
- You are fine with **eventual consistency** of membership and actor registry (gossip propagates in a few hundred ms typically).

### Summary

- **Seeds**: only for initial join; then gossip maintains the cluster.
- **Single port**: actor RPC and gossip share the same HTTP/2 server.
- **No external store**: no etcd/NATS/Redis.

---

## Mode 2: Head Node

### How it works

- One node is designated the **head**; all other nodes are **workers**.
- The **head** holds the authoritative membership and actor registry in memory. It does not run gossip; it only accepts worker registration and heartbeat.
- **Workers** at startup connect to the head’s address, register themselves, and then run **heartbeat** and **sync** loops (pull membership/registry from the head at intervals).
- Actor registration/deregistration from workers is sent to the head; the head updates its state and workers get it on the next sync.

So: the head is a **central coordinator**. If the head is down, workers cannot discover each other or resolve actors until the head is back (or you reconfigure them to a new head).

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

Head node mode is supported via the Rust `SystemConfig` (e.g. `with_head_node()` / `with_head_addr()`). The Python high-level `init(addr=..., seeds=...)` API currently only supports **Gossip + seed** mode. To use head node from Python you need to build a `SystemConfig` with head options (if exposed on the Python `SystemConfig` in your version) and pass it to `ActorSystem.create(config, loop)`. Check the API for `SystemConfig` in the Python bindings for availability.

### Head node parameters

The head backend uses a small set of timers (configurable in Rust via `HeadNodeConfig`):

- **Sync interval**: how often workers pull membership/registry from the head (default 5s).
- **Heartbeat interval**: how often workers send a heartbeat to the head (default 10s).
- **Heartbeat timeout**: after how long the head considers a worker dead (default 30s).

Tuning these affects how quickly failed workers are removed from the view.

### When to use

- You want **simple operations**: one fixed address (the head) to open in firewalls and to monitor.
- You accept a **single point of failure** for coordination (head down ⇒ no new discovery until head is back).
- You prefer **strong consistency** of membership/registry from the head’s perspective (workers eventually see the same view after each sync).

### Comparison with Gossip

| Aspect | Gossip + seed | Head node |
|--------|----------------|-----------|
| Discovery | Decentralized; seeds only for join, then gossip | Centralized; workers talk only to head |
| Failure of “special” node | No single point of failure; any node can be seed | Head down ⇒ no new joins/updates until head recovers |
| Consistency | Eventually consistent (propagation delay) | Head is source of truth; workers eventually consistent with head |
| Config complexity | Need at least one reachable seed address | Need head address for every worker |

---

## Mode 3: Init in Ray (Pulsing on top of Ray)

### How it works

- You already have a **Ray cluster** and run your code with `ray.init(...)` (or equivalent).
- You use **`pulsing.ray.init_in_ray()`** so that **each process** (driver and each worker that uses Pulsing) starts a Pulsing actor system and **joins a single Pulsing cluster**.
- Seed discovery is done via **Ray’s internal KV store**:
  - The first process that calls `init_in_ray()` starts Pulsing with no seeds, gets its own bind address, and **writes that address** into Ray KV under a fixed key (e.g. `pulsing:seed_addr`).
  - Any later process that calls `init_in_ray()` **reads** that key, gets the seed address, and starts Pulsing **with that seed**. So all Pulsing nodes join the same gossip cluster, with the first node’s address as the initial seed.
- Under the hood, Pulsing still uses **Gossip + seed**: the “seed” is simply supplied by Ray KV instead of by your config. So you get one Pulsing cluster per Ray cluster (or per KV namespace, if you use it that way), with no extra etcd/NATS.

This is “**init in Ray**”: Pulsing is **networked** using its own gossip protocol, but **deployed and discovered** using Ray’s runtime.

### Configuration and usage

**Requirements**

- Ray must be installed and **initialized** before calling `init_in_ray()`.
- Every process that uses Pulsing (driver and workers) must call `init_in_ray()` (or the async variant) in that process.

**Basic usage**

```python
import ray
from pulsing.ray import init_in_ray

# Option A: init_in_ray as worker_process_setup_hook (recommended)
# Then every worker process will run init_in_ray at startup.
ray.init(runtime_env={"worker_process_setup_hook": init_in_ray})

# Driver process must also initialize Pulsing
init_in_ray()

# Now use Pulsing as usual
import pulsing as pul
@pul.remote
class MyActor:
    def run(self): return "ok"

actor = await MyActor.spawn(name="my_actor")  # can be on any node
```

**Async variant (e.g. for async Ray actors)**

```python
from pulsing.ray import async_init_in_ray

# Inside an async Ray actor or async context
await async_init_in_ray()
```

**Cleanup (optional)**

If you want to clear the seed key from Ray KV when tearing down (e.g. in tests):

```python
from pulsing.ray import cleanup
cleanup()
```

### How the seed is chosen

- **First writer wins**: the first process that calls `init_in_ray()` and successfully writes the KV key becomes the “seed” node (its address is stored).
- If the key already exists, the process reads it and uses that address as `seeds=[...]`, so it joins the existing cluster.
- In the rare case of a race (two processes start with no seed, both write), the implementation may shut down one Pulsing instance and re-join using the winning seed. See `pulsing.ray` source for the exact logic.

So you do **not** configure seeds manually; Ray KV provides the first seed, and from then on the cluster runs in **Gossip + seed** mode with that node as the initial contact.

### When to use

- You already run **Ray** (for other workloads or for scheduling) and want Pulsing actors to run on the same nodes and form one cluster.
- You want **one-line** cluster formation per process (`init_in_ray()`) without managing seed lists or head addresses yourself.
- You are okay depending on **Ray’s runtime** (and its KV) for bootstrap; after that, Pulsing uses only its own HTTP/2 + gossip.

### Limitations

- **Ray is required**: `init_in_ray()` depends on Ray and its internal KV. Do not use this mode if you are not using Ray.
- **Process model**: each process that uses Pulsing must call `init_in_ray()` (or `async_init_in_ray()`). The hook ensures workers get it; the driver must call it explicitly.
- **Single cluster per Ray cluster**: the KV key is global to the Ray cluster, so all callers of `init_in_ray()` in that cluster join the same Pulsing cluster.

---

## Comparison and choice

| Criterion | Gossip + seed | Head node | Init in Ray |
|-----------|----------------|-----------|-------------|
| **External deps** | None | None | Ray (and its KV) |
| **Single point of failure** | No | Yes (head) | No (after bootstrap, same as gossip) |
| **Config** | Bind addr + optional seeds | Bind addr + head addr or head role | None (Ray KV provides seed) |
| **Best environment** | K8s, VMs, bare metal | Single coordinator acceptable | Existing Ray cluster |
| **Consistency** | Eventually consistent | Head-authoritative, then eventual | Same as Gossip + seed |
| **Python `init()`** | Yes (`addr`, `seeds`) | Via `SystemConfig` if exposed | `init_in_ray()` |

**Suggested choice:**

- **Already on Ray** → use **init in Ray** for minimal config and automatic seed discovery.
- **Need no SPOF and no Ray** → use **Gossip + seed** (and, in K8s, a Service as seed).
- **Want one fixed coordinator and simple ops** → use **Head node**.

---

## Best practices

1. **Gossip + seed**
   - In Kubernetes, use a **Service** (or multiple seeds) as seed; set `seed_probe_count` (e.g. 3) and `seed_rejoin_interval` (e.g. 15s) so new nodes and partition recovery work well.
   - Ensure the **same port** is open for all nodes (actor + gossip); avoid extra firewall rules.

2. **Head node**
   - Run the head on a **stable host/port** and optionally put it behind a load balancer for HA (replace head process but keep the same address).
   - Tune **heartbeat timeout** so workers are not marked dead too early under load.

3. **Init in Ray**
   - Call **`init_in_ray()` in the driver** and use **`worker_process_setup_hook`** so every worker process joins the same Pulsing cluster.
   - For tests, call **`cleanup()`** when tearing down the Ray cluster if you want a clean KV state.

4. **Security**
   - For any mode, you can enable **TLS** (e.g. passphrase-derived certs) so that actor and cluster traffic are encrypted and authenticated. See [Security](../guide/security.md).

---

## See also

- [Node Discovery](node-discovery.md) – Gossip protocol and seed probing in detail.
- [Architecture](architecture.md) – System components and message flow.
- [Migrate from Ray](../quickstart/migrate_from_ray.md) – API mapping from Ray to Pulsing.
