# Cluster Networking (Design)

This document describes **how cluster networking is implemented** in Pulsing. For configuration, usage, and which mode to choose, see [Cluster Networking (Quick Start)](../quickstart/cluster_networking.md).

---

## Overview

A Pulsing **cluster** is a set of nodes that share:

- **Membership**: which nodes are in the cluster and whether they are alive
- **Actor registry**: which named actors exist and on which node(s) they run

All cluster traffic (membership, registry, and actor messages) uses a **single HTTP/2 port** per node. No external services (etcd, NATS, Redis) are required.

The system supports three **naming backends**, selected by configuration:

| Backend | Role | Discovery / registry |
|---------|------|----------------------|
| **GossipBackend** | All nodes equal; seeds only for initial join | Gossip loop + SWIM over same transport |
| **HeadNodeBackend** | One head, rest workers | Head holds state; workers register and sync |
| **Init in Ray** | Same as Gossip | Ray KV provides first seed; then Gossip |

The runtime chooses the backend from `SystemConfig`: if `head_addr` or `is_head_node` is set, it uses `HeadNodeBackend`; otherwise it uses `GossipBackend`. Init-in-Ray is a bootstrap pattern that still uses Gossip once the seed is known.

---

## Gossip + seed: how it works

- Each node has a **bind address**. Non-first nodes are given one or more **seed** addresses.
- A node **joins** by sending a join request to each seed. If the seed is behind a load balancer (e.g. a Kubernetes Service), the node may probe multiple times; each probe can hit a different peer. It receives a **Welcome** message with the current member list.
- After joining, nodes run a **gossip loop**: on a fixed interval (e.g. 200 ms), each node picks a subset of peers (e.g. `fanout` = 3) and sends a **Gossip** message containing a partial view of membership, failure information, and (optionally) named-actor registry. Peers merge this into their local state.
- **SWIM**-style failure detection runs over the same HTTP/2 transport: nodes ping each other; suspected nodes are marked and eventually removed from the view. See [Node Discovery](node-discovery.md) and [SWIM](cluster/swim) for details.
- Optionally, nodes periodically **re-probe** the seed address(es) (e.g. every 15 s). This helps recover from network partitions and, when the seed is a load-balanced endpoint, discover new nodes.

So: **seeds are only used to obtain an initial member list**. After that, the cluster is maintained by gossip; there is no permanent master. Any node can act as a seed for new joiners.

### Message flow (simplified)

```
New node --Join--> Seed(s)
Seed(s)  --Welcome(members)--> New node
New node merges members, then:
  loop: pick peers -> send Gossip(partial_view, failures, actors) -> merge responses
```

### Configuration (implementation)

Gossip timing and behavior are controlled by `GossipConfig`: `gossip_interval`, `fanout`, `seed_probe_count`, `seed_probe_interval`, `seed_rejoin_interval`, and SWIM parameters. See [Node Discovery](node-discovery.md) for the full list and defaults.

---

## Head node: how it works

- One node is configured as the **head** (`is_head_node`); all others are **workers** (`head_addr` set).
- The **head** holds in memory the authoritative membership and actor registry. It does **not** run gossip. It exposes HTTP endpoints for worker **registration**, **heartbeat**, and **sync** (pull membership/registry).
- **Workers** at startup POST to the head to register, then run two loops:
  - **Heartbeat**: periodically send a heartbeat to the head so the head knows the worker is alive.
  - **Sync**: periodically pull the full membership and actor registry from the head.
- When a worker spawns or stops a named actor, it notifies the head; the head updates its registry. Workers get the updated view on the next sync.

So the head is a **central coordinator**. If the head is down, workers cannot discover new members or resolve actors until the head is back (or reconfigured to a new head).

### Backend selection

In Rust, `ActorSystem::new(config)` builds a `NamingBackend`: if `config.head_addr.is_some() || config.is_head_node`, it creates a `HeadNodeBackend`; otherwise it creates a `GossipBackend`. The backend implements `NamingBackend::join()`, register/unregister of actors, and resolve of named actors; the system uses it for all cluster-related operations.

---

## Init in Ray: how it works

- Pulsing runs inside a **Ray** cluster. Each process that uses Pulsing calls `init_in_ray()` (or `async_init_in_ray()`).
- **Seed discovery** uses Ray’s **internal KV store**:
  - The first process to call `init_in_ray()` starts Pulsing with **no seeds**, gets its bind address, and **writes** that address into Ray KV under a fixed key (e.g. `pulsing:seed_addr`). It is the initial “seed” node.
  - Any later process reads that key, gets the seed address, and starts Pulsing **with that seed**. So all processes join the same Pulsing cluster; under the hood it is still **Gossip + seed**, with the first writer’s address as the seed.
- If two processes race to write the key, the implementation may shut down one Pulsing instance and re-join using the winner’s address (see `pulsing.integrations.ray`).

So: **Ray KV only provides the first seed**. After that, the cluster behaves like a normal Gossip cluster. There is no separate “Ray backend”; it is Gossip with a different bootstrap source.

---

## Single port and transport

All cluster protocols (Gossip, Head registration/sync, and actor RPC) use the same **HTTP/2 server** and the same **Http2Transport**. Paths such as `POST /cluster/gossip` and `POST /actor/{name}` are multiplexed on one port. This simplifies deployment and firewalls. See [HTTP2 Transport](http2-transport.md) and [Node Discovery](node-discovery.md).

---

## See also

- [Cluster Networking (Quick Start)](../quickstart/cluster_networking.md) — how to use and configure
- [Node Discovery](node-discovery.md) — Gossip protocol and seed probing in detail
- [Architecture](architecture.md) — system components and message flow
