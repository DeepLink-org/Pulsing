# Distributed Memory Queue

Pulsing includes a **distributed memory queue** built on the same actor + cluster primitives as the rest of the system.

It is designed for:

- **High-throughput ingestion** with sharding (buckets)
- **Location-transparent access** (writers/readers don’t need to know where data lives)
- **Optional persistence** backed by [Lance](https://github.com/lancedb/lance) (`lance` + `pyarrow`)

## Architecture

- **Topic**: logical queue name, e.g. `my_queue`
- **Buckets**: a topic is sharded into \(N\) buckets (`num_buckets`)
- **BucketStorage (Actor)**: each bucket is a `BucketStorage` actor holding:
  - an in-memory buffer
  - a persisted dataset (`data.lance`) when Lance is available
- **StorageManager (Actor)**: one per node (`queue_storage_manager`)
  - uses **consistent hashing** to decide which node owns a given bucket
  - creates / returns the local `BucketStorage` if the bucket is owned locally
  - otherwise returns a redirect to the owner node

### Consistent hashing & redirect flow

`StorageManager` decides the owner node of each `(topic, bucket_id)` and returns either:

- `BucketReady` (owned locally) → use the returned `BucketStorage` actor
- `Redirect` (owned remotely) → resolve the remote `StorageManager` and retry

```mermaid
flowchart TB
    C[Client: get_bucket_ref(topic, bucket_id)] --> SM[Local StorageManager]
    SM --> H[Compute owner via consistent hashing]
    H --> D{owner == local?}
    D -->|Yes| BR[BucketReady(actor_id, node_id_hex)]
    D -->|No| RD[Redirect(owner_node_id_hex, owner_addr)]

    RD --> RSM[Resolve remote StorageManager]
    RSM --> SM2[Remote StorageManager]
    SM2 --> BR2[BucketReady(actor_id, node_id_hex)]

    BR --> REF[ActorSystem.actor_ref(ActorId)]
    BR2 --> REF

    style SM fill:#e3f2fd,stroke:#1976d2
    style SM2 fill:#e3f2fd,stroke:#1976d2
    style RD fill:#fff3e0,stroke:#f57c00
    style BR fill:#e8f5e9,stroke:#388e3c
    style BR2 fill:#e8f5e9,stroke:#388e3c
```

## Quick start (async)

```python
import asyncio
from pulsing.actor import SystemConfig, create_actor_system
from pulsing.queue import write_queue, read_queue


async def main():
    system = await create_actor_system(SystemConfig.standalone())
    try:
        writer = await write_queue(
            system,
            topic="my_queue",
            bucket_column="user_id",
            num_buckets=4,
            batch_size=10,
        )
        reader = await read_queue(system, topic="my_queue")

        # write
        await writer.put({"user_id": "u1", "payload": "hello"})

        # read (memory + persisted are both visible)
        records = await reader.get(limit=10)
        print(records)

        # persist buffered records
        await writer.flush()
    finally:
        await system.shutdown()


asyncio.run(main())
```

## Sync wrapper

If you need a blocking API (e.g. called from a thread), use `.sync()`:

```python
writer = (await write_queue(system, "my_queue")).sync()
reader = (await read_queue(system, "my_queue")).sync()

writer.put({"id": "1", "value": 100})
records = reader.get(limit=10)
writer.flush()
```

Note: don’t call the sync wrapper **inside** an async function (it blocks).

## Partitioning & bucketing

- A record **must** include the `bucket_column` (default `id`)
- The bucket is chosen by `md5(str(value)) % num_buckets`
- This gives stable sharding: same key always maps to the same bucket

## Reading modes

`read_queue` supports:

- **All buckets** (default): one reader iterates all buckets
- **Specific buckets**: `bucket_id=` or `bucket_ids=`
- **Distributed consumption**: `rank=` / `world_size=` assigns buckets via round-robin

Example:

```python
reader0 = await read_queue(system, "q", rank=0, world_size=2, num_buckets=4)  # [0, 2]
reader1 = await read_queue(system, "q", rank=1, world_size=2, num_buckets=4)  # [1, 3]
```

## Streaming & blocking reads

Bucket reads default to a streaming path (`GetStream`).

- **wait=false**: return immediately if no new data
- **wait=true**: block until new data arrives (optional `timeout`)

## Visibility semantics (buffer vs persisted)

Each `BucketStorage` has two segments:

- **Persisted segment**: `data.lance` (when Lance is available)
- **In-memory buffer**: newly written records not flushed yet

Readers see a unified logical view:

```mermaid
flowchart LR
    P[Persisted: 0..persisted_count) --> V[Unified view by offset]
    B[Buffer: persisted_count..total_count) --> V

    style P fill:#e8f5e9,stroke:#388e3c
    style B fill:#fff3e0,stroke:#f57c00
```

**Guarantees**

- After a successful `put`, data is **immediately visible** to readers (at least from the buffer).

**Non-guarantees**

- Durability is not guaranteed unless Lance is installed and `flush()` succeeds.

## Persistence (Lance)

`BucketStorage` persists buffered records on `flush()` (and also auto-flushes when the buffer reaches `batch_size`).

- If `lance` / `pyarrow` are missing, it falls back to **in-memory only**

## Multi-consumer offsets: strategy & limitations

### How offsets work

- Reads are **offset-based** (`offset`, `limit`) per bucket.
- `QueueReader` keeps a **client-side offset per bucket** and advances it by the number of records it returned.

This means the queue behaves more like a **log** than a destructive queue.

### Distributed consumption (`rank` / `world_size`)

When you pass `rank` and `world_size`, Pulsing assigns buckets via round-robin:

- `num_buckets=4, world_size=2`: rank0 → `[0,2]`, rank1 → `[1,3]`

This avoids multiple consumers reading the same buckets **by construction**.

### Limitations (important)

- There is no built-in **consumer group / ack / commit log**.
- If two consumers read the same bucket with independent offsets, they can read the same records (duplicates).
- Offsets are in-memory (client-side) unless you persist them yourself.

Recommended patterns:

- Include an **idempotency key** in records and deduplicate on the consumer side.
- Model acks/commits as actor state (or a separate commit log).

## Where to look in code

- `python/pulsing/queue/queue.py`: high-level `Queue`, `write_queue`, `read_queue`
- `python/pulsing/queue/manager.py`: `StorageManager` and bucket routing / redirects
- `python/pulsing/queue/storage.py`: `BucketStorage` (buffer + Lance persistence + streaming reads)
- `examples/python/distributed_queue.py`: end-to-end example
- `tests/python/test_queue.py`: behavior + stress tests
