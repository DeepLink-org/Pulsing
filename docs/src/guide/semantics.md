# Semantics & Guarantees

This page defines what Pulsing **guarantees** (and does **not** guarantee) for:

- Actor execution
- Remote messaging (`ask` / `tell`)
- Streaming responses (`StreamMessage`)
- Distributed memory queue (`pulsing.queue`)

The goal is to make it safe to build production systems without assuming stronger semantics than Pulsing actually provides.

## Quick summary

- **Actors are single-threaded** from the perspective of message handling: a given actor processes messages sequentially.
- **No exactly-once**: Pulsing does not provide end-to-end exactly-once delivery or processing guarantees.
- **No consumer-group semantics** in the queue: reads are offset-based and **non-destructive** unless your application implements acknowledgements.
- **Streaming is bounded**: `StreamMessage.create(..., buffer_size=32)` uses a bounded channel; writers naturally apply backpressure by awaiting writes.

## Terminology

- **Delivery**: a message reaches the target actor.
- **Processing**: the target actor executes its `receive`.
- **At-most-once**: a message is delivered/processed 0 or 1 time.
- **At-least-once**: a message is delivered/processed 1+ times (duplicates possible).
- **Exactly-once**: delivered/processed exactly one time (requires dedup + commit protocols).

## Actor execution semantics

### Ordering (within a single actor)

- **Guaranteed**: a single actor processes messages **one at a time** (no concurrent `receive` for the same actor).
- **Not guaranteed**: Pulsing does not promise a global ordering across different actors.

Practical implication:

- If your actor state depends on ordering, keep the state inside the actor and serialize changes through its mailbox.

### Failure & exceptions

- If `receive` raises an exception (or returns an error message), the caller will observe a failure (typically surfaced as an exception in the client wrapper).
- Pulsing supports **actor-level restarts** (configurable restart policy + backoff) but does **not** provide a supervision tree.
  - See: [Reliability](reliability.md)

## Remote messaging semantics (`ask` / `tell`)

### `ask`

`ask` is request/response:

- **What you get**:
  - a single `Message`, or
  - a streaming `Message` (where `msg.is_stream == True`) that you consume via `msg.stream_reader()`
- **Timeouts/retries**:
  - Pulsing does not guarantee automatic retries at the API level.
  - Timeouts are configuration- and transport-dependent; you should assume requests can fail due to network partitions, node failures, or overload.

**Guarantees you can safely rely on**:

- **At-most-once per successful response**: if you received a response from the target actor, that response corresponds to one execution of `receive` on that actor instance.

**What you must NOT assume**:

- Exactly-once across failures (if you add retries in your app, duplicates are possible).
- Stable ordering for multiple in-flight requests across the network.

### `tell`

`tell` is fire-and-forget:

- **No response** and therefore **no delivery confirmation**.
- Use it only for messages that are safe to drop or for which your app can tolerate eventual reconciliation.

## Streaming semantics (`StreamMessage`)

`StreamMessage.create(msg_type, buffer_size=32)` returns `(stream_msg, writer)`.

### Stream composition

- A stream is composed of **Message objects**, not raw bytes.
- Each chunk in the stream is a complete `Message` with its own `msg_type` and payload.
- This enables **heterogeneous streams** where different chunks can have different types.

### Transparent Python object streaming

For Python-to-Python communication, streaming is **fully transparent**:

```python
# Writer side - just write Python objects directly
async def generate_stream():
    stream_msg, writer = StreamMessage.create("tokens")
    for token in tokens:
        await writer.write({"token": token, "index": i})  # dict is auto-pickled
    await writer.close()
    return stream_msg

# Reader side - receive Python objects directly
async for chunk in response.stream_reader():
    token = chunk["token"]  # chunk is already a Python dict
    print(token)
```

### Backpressure & buffering

- The stream is backed by a **bounded channel** of size `buffer_size`.
- `writer.write(...)` **awaits** when the buffer is full → natural backpressure.

### Lifecycle

- A stream can be terminated by:
  - `writer.close()` (normal completion)
  - `writer.error("...")` (error completion)
  - consumer cancellation via `reader.cancel()` or dropping the reader

### Delivery semantics

- Streaming chunks are best-effort.
- Consumers should be robust to partial streams and handle early termination.

Recommendation:

- Make each chunk independently meaningful (include `seq` / offsets / ids) so consumers can resume or deduplicate if needed.

## Queue semantics (`pulsing.queue`)

The distributed queue is **sharded** into buckets:

- records are assigned to buckets by `md5(str(bucket_value)) % num_buckets`
- each bucket is served by one `BucketStorage` actor in the cluster (chosen by consistent hashing)

### Visibility model (buffer + persisted)

`BucketStorage` maintains:

- **persisted** records (Lance dataset, when available)
- **in-memory buffer** records (not yet flushed)

**Guaranteed**:

- After a successful `put`, the record becomes **immediately visible** to readers (from the in-memory buffer).

**Not guaranteed**:

- Durable persistence unless Lance is installed and flush succeeds.

### Read semantics: offset-based, non-destructive

- Reads are **offset-based** (`offset`, `limit`) and do not delete data.
- `QueueReader` maintains offsets **per bucket** on the client side.

Implications:

- Multiple readers can read the same records unless you design a coordination/ack scheme.
- There is no built-in consumer group / exactly-once processing.

### Blocking reads (`wait` / `timeout`)

- `GetStream` supports waiting for new data when `wait=True`.
- On timeout, the stream ends and the caller gets fewer (or zero) records.

### Ordering

- Within a single bucket: records are appended to the bucket buffer and then flushed; readers observe a bucket’s records in offset order.
- Across buckets: there is **no total order**. If you read “all buckets”, results are a merge of per-bucket streams.

### Failure modes you must handle

- Node failure may temporarily make a bucket unavailable until routing converges.
- Retries at the application layer can produce duplicates; design idempotent consumers.

## Recommended application-level patterns

- **Idempotency key**: include a stable `id` and deduplicate at the consumer/actor.
- **Explicit acknowledgement**: model ack as an actor state update (or a separate “commit log”).
- **Timeout + retry policy**: keep it explicit in your app; don’t rely on implicit retries.
