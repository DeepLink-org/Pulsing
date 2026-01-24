# Remote Actors Guide

Guide to using actors across a cluster with location transparency.

## Cluster Setup

### Starting a Seed Node

```python
import pulsing as pul

# Node 1: Start seed node
system = await pul.actor_system(addr="0.0.0.0:8000")

# Spawn a public actor
await system.spawn(WorkerActor(), name="worker", public=True)
```

### Joining a Cluster

```python
# Node 2: Join cluster
system = await pul.actor_system(
    addr="0.0.0.0:8001",
    seeds=["192.168.1.1:8000"]
)

# Wait for cluster sync
await asyncio.sleep(1.0)
```

## Finding Remote Actors

### Using system.resolve()

```python
# Find actor by name (searches entire cluster)
remote_ref = await system.resolve("worker")
response = await remote_ref.ask({"action": "process", "data": "hello"})
```

### Using @remote Class.resolve()

```python
@pul.remote
class Worker:
    def process(self, data): return f"processed: {data}"

# Resolve with type info - returns ActorProxy with methods
worker = await Worker.resolve("worker")
result = await worker.process("hello")  # Direct method call
```

## Public vs Private Actors

### Public Actors

Public actors are visible to all nodes in the cluster:

```python
# Public actor - can be found by other nodes
await system.spawn(WorkerActor(), name="worker", public=True)
```

### Private Actors

Private actors are only accessible locally:

```python
# Private actor - local only
await system.spawn(WorkerActor(), name="local-worker", public=False)
```

## Location Transparency

The same API works for both local and remote actors:

```python
# Local actor
local_ref = await system.spawn(MyActor(), name="local")

# Remote actor (found via cluster)
remote_ref = await system.resolve("remote-worker")

# Same API for both
response1 = await local_ref.ask(msg)
response2 = await remote_ref.ask(msg)
```

## Error Handling

Remote actor calls can fail due to network issues:

```python
try:
    remote_ref = await system.resolve("worker")
    response = await remote_ref.ask(msg)
except Exception as e:
    print(f"Remote call failed: {e}")
```

## Best Practices

1. **Wait for cluster sync**: Add a small delay after joining a cluster
2. **Handle errors gracefully**: Wrap remote calls in try-except blocks
3. **Use public actors for cluster communication**: Set `public=True` for actors that need remote access
4. **Use @remote with resolve()**: Get typed proxies for better API experience
5. **Use timeouts**: Consider adding timeouts for remote calls

## Example: Distributed Counter

```python
import pulsing as pul

@pul.remote
class DistributedCounter:
    def __init__(self, init_value: int = 0):
        self.value = init_value

    def get(self) -> int:
        return self.value

    def increment(self, n: int = 1) -> int:
        self.value += n
        return self.value

# Node 1: Create counter
system1 = await pul.actor_system(addr="0.0.0.0:8000")
counter = await DistributedCounter.local(system1, init_value=0)

# Node 2: Access remote counter
system2 = await pul.actor_system(addr="0.0.0.0:8001", seeds=["127.0.0.1:8000"])
await asyncio.sleep(1.0)

# Resolve and use the remote counter
remote_counter = await DistributedCounter.resolve("counter")
value = await remote_counter.get()  # 0
value = await remote_counter.increment(5)  # 5
```

## Next Steps

- Learn about [Actor System](actor_system.md) basics
- Check [Node Discovery](../design/node-discovery.md) for cluster details
