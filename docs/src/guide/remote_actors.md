# Remote Actors Guide

Guide to using actors across a cluster with location transparency.

## Cluster Setup

### Starting a Seed Node

```python
from pulsing.actor import SystemConfig, create_actor_system

# Node 1: Start seed node
config = SystemConfig.with_addr("0.0.0.0:8000")
system = await create_actor_system(config)

# Spawn a public actor
await system.spawn(WorkerActor(), "worker", public=True)
```

### Joining a Cluster

```python
# Node 2: Join cluster
config = SystemConfig.with_addr("0.0.0.0:8001").with_seeds(["192.168.1.1:8000"])
system = await create_actor_system(config)

# Wait for cluster sync
await asyncio.sleep(1.0)
```

## Finding Remote Actors

### Using system.find()

```python
# Find actor by name (searches entire cluster)
remote_ref = await system.find("worker")

if remote_ref:
    response = await remote_ref.ask(Message.single("request", b"data"))
```

### Checking Actor Existence

```python
# Check if actor exists in cluster
if await system.has_actor("worker"):
    ref = await system.find("worker")
    # Use ref...
```

## Public vs Private Actors

### Public Actors

Public actors are visible to all nodes in the cluster:

```python
# Public actor - can be found by other nodes
await system.spawn(WorkerActor(), "worker", public=True)
```

### Private Actors

Private actors are only accessible locally:

```python
# Private actor - local only
await system.spawn(WorkerActor(), "local-worker", public=False)
```

## Location Transparency

The same API works for both local and remote actors:

```python
# Local actor
local_ref = await system.spawn(MyActor(), "local")

# Remote actor (found via cluster)
remote_ref = await system.find("remote-worker")

# Same API for both
response1 = await local_ref.ask(msg)
response2 = await remote_ref.ask(msg)
```

## Error Handling

Remote actor calls can fail due to network issues:

```python
try:
    remote_ref = await system.find("worker")
    if remote_ref:
        response = await remote_ref.ask(msg)
    else:
        print("Actor not found")
except Exception as e:
    print(f"Remote call failed: {e}")
```

## Best Practices

1. **Wait for cluster sync**: Add a small delay after joining a cluster
2. **Handle missing actors**: Always check if `find()` returns None
3. **Use public actors for cluster communication**: Set `public=True` for actors that need remote access
4. **Handle network errors**: Wrap remote calls in try-except blocks
5. **Use timeouts**: Consider adding timeouts for remote calls

## Example: Distributed Counter

```python
@remote
class DistributedCounter:
    def __init__(self, init_value: int = 0):
        self.value = init_value

    def get(self) -> int:
        return self.value

    def increment(self, n: int = 1) -> int:
        self.value += n
        return self.value

# Node 1
system1 = await create_actor_system(SystemConfig.with_addr("0.0.0.0:8000"))
counter1 = await DistributedCounter.local(system1, init_value=0)
await system1.spawn(counter1, "counter", public=True)

# Node 2
system2 = await create_actor_system(
    SystemConfig.with_addr("0.0.0.0:8001").with_seeds(["127.0.0.1:8000"])
)
await asyncio.sleep(1.0)

# Access remote counter from Node 2
remote_counter = await system2.find("counter")
if remote_counter:
    value = await remote_counter.get()  # 0
    value = await remote_counter.increment(5)  # 5
```

## Next Steps

- Learn about [Actor System](actor_system.md) basics
- Check [Node Discovery](../design/node-discovery.md) for cluster details
