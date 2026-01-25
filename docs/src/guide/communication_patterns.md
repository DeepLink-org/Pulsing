# Communication Patterns Guide

This guide explains the **design rationale** and **use cases** for different communication patterns in Pulsing, helping you understand **why** these patterns exist and **when** to use them.

## Why Different Communication Patterns?

### Core Actor Property

In the Actor model, each Actor **processes one message at a time**. This is a fundamental guarantee of the Actor model, ensuring safe state updates.

```
Actor Mailbox (FIFO Queue)
    ↓
[Message1] → Actor processes → Response1
[Message2] → Actor processes → Response2  ← Must wait for Message1
[Message3] → Actor processes → Response3  ← Must wait for Message2
```

### The Problem: Blocking vs Non-Blocking

If an Actor is blocked while processing a message (e.g., waiting for a network response):

```
❌ Synchronous blocking mode:
Message1: [Waiting for HTTP...████████] 500ms  ← Blocked
Message2: [Waiting...]                          ← Cannot process!
Message3: [Waiting...]                          ← Cannot process!
```

**Result**: Actor cannot process other messages, extremely low throughput.

**Solution**: Use asynchronous non-blocking mode:

```
✅ Asynchronous non-blocking mode:
Message1: [Waiting for HTTP...] 500ms  ← Waiting in background
Message2: [Processing...] 10ms         ← Can process concurrently!
Message3: [Processing...] 10ms         ← Can process concurrently!
```

**Result**: Actor can process multiple requests concurrently, dramatically improved throughput.

### Why Streaming?

For operations that take a long time to complete (e.g., LLM generating 1000 tokens), if we wait for everything:

```
❌ Wait for everything:
User: [Waiting...████████████████] 10 seconds → See result
```

**Problem**: Poor user experience, long wait time.

**Solution**: Stream results incrementally:

```
✅ Streaming:
User: [token1][token2][token3]...  ← See results immediately
```

**Result**: Users see progress immediately, much better experience.

---

## Four Communication Patterns

Based on the above principles, Pulsing provides four communication patterns:

| Pattern | Method Type | Why Needed | Use Case |
|---------|-------------|------------|----------|
| **Sync** | `def method()` | Fast operations don't need concurrency, simpler code | Fast CPU work, state mutation |
| **Async** | `async def method()` | Avoid blocking, allow concurrent processing | I/O operations, external API calls |
| **Streaming** | `async def method()` with `yield` | Incremental return, better UX | LLM token generation, large data transfer |
| **Fire-and-Forget** | `tell()` | No response needed, maximize throughput | Logging, notifications |

## 1. Sync Methods (`def method`)

### Why Sync Methods?

**Principle**: For fast operations (< 10ms), the overhead of concurrency outweighs the benefits.

- ✅ **Simple and direct**: No need for `async/await`, cleaner code
- ✅ **No concurrency overhead**: Fast operations don't need concurrency, sequential is fine
- ✅ **Predictable**: Strict sequential execution, easy to understand and debug

**Use case**: Operations are fast enough that blocking time is negligible.

### Behavior

- **Sequential execution**: Actor processes one request at a time
- **Blocks the actor**: While processing, the actor cannot handle other messages
- **Simple and predictable**: No concurrency concerns

### When to Use

✅ **Best for:**
- Fast CPU-bound operations (calculations, state updates)
- Simple state mutations (incrementing counters, updating dictionaries)
- Operations that complete in microseconds to milliseconds (< 10ms)

❌ **Avoid for:**
- Network requests (HTTP, database queries)
- File I/O operations
- Any operation that might take > 10ms

### Example

```python
@pul.remote
class Counter:
    def __init__(self):
        self.value = 0
        self.history = []

    # ✅ Good: Fast state mutation
    def increment(self, n: int = 1) -> int:
        self.value += n
        self.history.append(self.value)
        return self.value

    # ✅ Good: Simple calculation
    def get_average(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    # ❌ Bad: Network I/O blocks the actor
    def fetch_data(self, url: str) -> dict:
        # This blocks the actor for the entire HTTP request!
        response = requests.get(url)  # Don't do this!
        return response.json()
```

### Performance Characteristics

```
Request 1: [████████████] 2ms
Request 2:              [████████████] 2ms
Request 3:                            [████████████] 2ms
Total: 6ms (sequential)
```

---

## 2. Async Methods (`async def method`)

### Why Async Methods?

**Core Problem**: If you use sync methods for I/O operations, the Actor will be blocked and cannot process other messages.

**Principle**:
- Async methods **yield control** when `await`ing
- Actor can **process other messages** while waiting
- Multiple async operations can **execute concurrently**

**Comparison**:

```python
# ❌ Sync: Blocks Actor
def fetch_data(self, url: str) -> dict:
    response = requests.get(url)  # Blocks for 500ms
    return response.json()
# Result: Actor cannot process any other messages during these 500ms

# ✅ Async: Non-blocking
async def fetch_data(self, url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)  # Can process other messages while waiting
        return response.json()
# Result: Actor can process other requests while waiting for HTTP response
```

### Behavior

- **Non-blocking execution**: Actor can process other messages while awaiting
- **Concurrent processing**: Multiple async methods can run simultaneously
- **Background task**: Method runs as a background task on the actor

### When to Use

✅ **Best for:**
- I/O operations (HTTP requests, database queries, file I/O)
- External API calls
- Operations that might take > 10ms
- When you need concurrent processing of multiple requests

❌ **Avoid for:**
- Fast CPU-bound operations (use sync methods instead)
- Simple state mutations (sync methods are simpler)

### Example

```python
@pul.remote
class DataService:
    def __init__(self):
        self.cache = {}

    # ✅ Good: Network I/O - doesn't block actor
    async def fetch_user(self, user_id: str) -> dict:
        # While waiting for HTTP response, actor can handle other requests
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://api.example.com/users/{user_id}")
            return response.json()

    # ✅ Good: Database query
    async def get_orders(self, user_id: str) -> list[dict]:
        # While waiting for DB, actor can process other requests
        async with database.transaction() as tx:
            return await tx.fetch("SELECT * FROM orders WHERE user_id = $1", user_id)

    # ✅ Good: Multiple concurrent operations
    async def fetch_user_profile(self, user_id: str) -> dict:
        # These run concurrently, not sequentially
        user, orders, preferences = await asyncio.gather(
            self.fetch_user(user_id),
            self.get_orders(user_id),
            self.get_preferences(user_id),
        )
        return {"user": user, "orders": orders, "preferences": preferences}

    # ❌ Bad: Fast operation - sync is simpler
    async def get_cache(self, key: str) -> dict:
        # This is fast enough for sync method
        return self.cache.get(key, {})
```

### Performance Characteristics

```
Request 1: [████████████████████] 50ms (awaiting HTTP)
Request 2: [████████████████████] 50ms (awaiting HTTP)  ← Concurrent!
Request 3: [████████████████████] 50ms (awaiting HTTP)  ← Concurrent!
Total: ~50ms (concurrent, not 150ms!)
```

### Usage Patterns

#### Pattern 1: Await Final Result

```python
service = await DataService.spawn()

# Wait for final result
result = await service.fetch_user("user123")
print(result)
```

#### Pattern 2: Fire-and-Forget (Background Task)

```python
# Start async operation, don't wait
task = asyncio.create_task(service.fetch_user("user123"))

# Do other work...
await other_operations()

# Get result later
result = await task
```

---

## 3. Streaming (`async def method` with `yield`)

### Why Streaming?

**Core Problem**: Some operations take a long time to complete (e.g., LLM generating 1000 tokens). If we wait for everything:

```
❌ Wait for everything:
User request → [Generating...████████] 10 seconds → Return all results
Problem: User must wait 10 seconds to see anything
```

**Principle**:
- Use `yield` to **incrementally return** results
- Client can **start processing** the first result immediately
- Better user experience, reduced perceived latency

```
✅ Streaming:
User request → [token1] → [token2] → [token3]... → Complete
Result: User sees first token immediately, no waiting
```

**Additional Benefits**:
- Can **cancel early** (if user doesn't need it)
- Can show **progress updates**
- Can handle **large datasets** (don't need to load everything into memory)

### Behavior

- **Incremental delivery**: Results are sent as they become available
- **Non-blocking**: Actor can handle other messages while generating stream
- **Backpressure**: Natural flow control via bounded channels
- **Cancellation**: Client can cancel stream consumption

### When to Use

✅ **Best for:**
- LLM token generation (users want to see output immediately)
- Large data transfer (process in chunks, avoid memory overflow)
- Real-time data feeds (sensor data, logs)
- Progress updates (long-running tasks need feedback)

❌ **Avoid for:**
- Small, complete responses (use regular async methods)
- When you need atomic results (all-or-nothing)

### Example

```python
@pul.remote
class LLMService:
    # ✅ Good: Streaming LLM tokens
    async def generate(self, prompt: str):
        # Stream tokens as they're generated
        async for token in self.llm_client.stream(prompt):
            yield {"token": token, "type": "token"}

        # Final result
        yield {"type": "done", "total_tokens": count}

    # ✅ Good: Large file processing
    async def process_large_file(self, file_path: str):
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                processed = process_line(line)
                yield {"line": i, "data": processed}

                # Allow other messages to be processed
                await asyncio.sleep(0)  # Yield control

    # ✅ Good: Progress updates
    async def long_running_task(self, task_id: str):
        for step in range(100):
            result = await do_work(step)
            yield {"progress": step, "result": result}
```

### Usage Patterns

#### Pattern 1: Consume Stream Incrementally

```python
service = await LLMService.spawn()

# Process tokens as they arrive
async for chunk in service.generate("Hello, world!"):
    if chunk["type"] == "token":
        print(chunk["token"], end="", flush=True)
    elif chunk["type"] == "done":
        print(f"\nTotal tokens: {chunk['total_tokens']}")
```

#### Pattern 2: Await Final Result (Skip Intermediate)

```python
# If you only care about final result
result = await service.generate("Hello, world!")
# Pulsing automatically collects all chunks and returns final value
```

#### Pattern 3: Cancel Stream Early

```python
async def consume_with_timeout():
    async with asyncio.timeout(5.0):
        async for chunk in service.generate("Very long prompt..."):
            process(chunk)
    # Stream automatically cancelled on timeout
```

### Performance Characteristics

```
Client:     [chunk1][chunk2][chunk3]...
            ↓       ↓       ↓
Network:    [████][████][████]...
            ↓       ↓       ↓
Actor:      [gen][gen][gen]...  ← Non-blocking generation
            ↓       ↓       ↓
LLM API:    [████████████████]...  ← Continuous generation

Total latency: First chunk arrives quickly, not waiting for all chunks
```

---

## 4. Ask vs Tell

### Why Two Modes?

**Core Difference**: Whether you need to wait for a response.

- **`ask()`**: Needs response, waits for result
- **`tell()`**: No response needed, continues immediately after sending

**Why It Matters**:

```
❌ Using ask() for everything:
await logger.ask({"level": "info", "msg": "..."})  # Wait for response
await metrics.ask({"event": "..."})                # Wait for response
await notifier.ask({"user": "..."})                 # Wait for response
Problem: Even when you don't need results, you wait, reducing throughput

✅ Distinguish usage:
await logger.tell({"level": "info", "msg": "..."})  # Don't wait
await metrics.tell({"event": "..."})                # Don't wait
result = await service.get_user("123")               # Need result, use ask
Benefit: Operations that don't need responses don't block, higher throughput
```

### `ask()` - Request/Response

**Why use**: Need to know the operation result or success status.

**When to use:**
- Need response for further processing
- Need to know if operation succeeded
- Need error handling

```python
# ✅ Good: Need the result
result = await counter.increment(10)
print(f"New value: {result}")

# ✅ Good: Need to check success
try:
    user = await service.get_user("user123")
except PulsingActorError:
    print("User not found")
```

### `tell()` - Fire-and-Forget

**Why use**: Maximize throughput, no need to wait for response.

**When to use:**
- Don't need response (logging, metrics)
- Operation is safe to drop
- Want maximum throughput

```python
# ✅ Good: Logging - don't need response
await logger.tell({"level": "info", "message": "User logged in"})

# ✅ Good: Metrics - fire and forget
await metrics.tell({"event": "page_view", "page": "/home"})

# ✅ Good: Notifications - eventual delivery OK
await notifier.tell({"user_id": "123", "message": "New email"})
```

### Comparison

| Aspect | `ask()` | `tell()` |
|--------|---------|----------|
| **Response** | ✅ Returns value | ❌ No response |
| **Error handling** | ✅ Exceptions raised | ❌ Silent failures |
| **Throughput** | Lower (waits for response) | Higher (no waiting) |
| **Use case** | Operations that need results | Operations that can be dropped |

---

## 5. Quick Decision Guide

### Decision Flow

```
Start: What does your operation need?

1. Need a response?
   ├─ No → Use `tell()` (fire-and-forget)
   │      Reason: No need to wait, maximize throughput
   │
   └─ Yes → Continue to next step

2. How long does the operation take?
   ├─ < 10ms → Use `def method()` (sync)
   │           Reason: Fast enough, no concurrency needed, simpler code
   │
   └─ > 10ms → Continue to next step

3. Need incremental results?
   ├─ No → Use `async def method()` (async)
   │       Reason: Avoid blocking, allow concurrent processing
   │
   └─ Yes → Use `async def method()` with `yield` (streaming)
            Reason: Return partial results immediately, better UX
```

### Why Choose This Way?

| Choice | Reason |
|--------|--------|
| `tell()` | No response needed, not waiting maximizes throughput |
| `def method()` | Fast operations don't need concurrency, sync code is simpler |
| `async def method()` | Avoid blocking Actor, allow concurrent processing of multiple requests |
| `async def method()` + `yield` | Return partial results immediately, better user experience |

---

## 6. Real-World Examples

### Example 1: Counter Service

```python
@pul.remote
class Counter:
    def __init__(self):
        self.value = 0

    # ✅ Sync: Fast state mutation
    def increment(self, n: int = 1) -> int:
        self.value += n
        return self.value

    # ✅ Sync: Simple read
    def get(self) -> int:
        return self.value

    # ✅ Sync: Fast calculation
    def reset(self) -> None:
        self.value = 0
```

**Why use sync?**
- All operations are fast (< 1ms)
- No I/O operations, pure in-memory operations
- No concurrency needed, sequential execution is fine
- Sync code is simpler and easier to understand

**What if we use async instead?**
- ❌ Adds unnecessary `async/await` overhead
- ❌ More complex code with no performance benefit
- ❌ Operation is too fast, concurrency provides zero benefit

---

### Example 2: HTTP API Client

```python
@pul.remote
class APIClient:
    # ✅ Async: Network I/O
    async def fetch_data(self, url: str) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)  # While waiting, Actor can process other requests
            return response.json()

    # ✅ Async: Multiple concurrent requests
    async def fetch_multiple(self, urls: list[str]) -> list[dict]:
        tasks = [self.fetch_data(url) for url in urls]
        return await asyncio.gather(*tasks)  # Concurrent execution, not sequential
```

**Why use async?**
- Network requests take time (typically 50-500ms)
- If using sync, Actor would be blocked and cannot process other requests
- Using async, Actor can process other messages while waiting for HTTP response
- Multiple requests can execute concurrently, dramatically improving throughput

**What if we use sync instead?**
- ❌ Actor cannot process any other messages while waiting for HTTP response
- ❌ Extremely low throughput (can only process one request at a time)
- ❌ Poor user experience (all requests queue up)

---

### Example 3: LLM Service

```python
@pul.remote
class LLMService:
    # ✅ Streaming: Tokens arrive incrementally
    async def generate(self, prompt: str):
        async for token in self.llm_client.stream(prompt):
            yield {"token": token}  # Return each token immediately
        yield {"done": True}

    # ✅ Async: Single completion (no streaming needed)
    async def embed(self, text: str) -> list[float]:
        return await self.llm_client.embed(text)  # Fast completion, no streaming needed
```

**Why `generate` uses streaming?**
- LLM generation takes time (possibly 5-30 seconds)
- If waiting for everything, users must wait a long time to see any content
- Using streaming, users see the first token immediately, much better experience
- Users can cancel early (if they don't need it)

**Why `embed` uses async instead of streaming?**
- Embedding operations are usually fast (< 1 second)
- Result is a single vector, no need for incremental return
- Using async avoids blocking, no need for streaming

**What if `generate` doesn't use streaming?**
- ❌ Users must wait 10-30 seconds to see any output
- ❌ Cannot cancel early (must wait even if not needed)
- ❌ Extremely poor user experience

---

### Example 4: Mixed Patterns

```python
@pul.remote
class DataProcessor:
    def __init__(self):
        self.processed_count = 0  # Fast state update

    # ✅ Sync: Fast counter update
    def get_stats(self) -> dict:
        return {"processed": self.processed_count}

    # ✅ Async: I/O operation
    async def fetch_from_db(self, query: str) -> list[dict]:
        return await database.query(query)

    # ✅ Streaming: Process large dataset incrementally
    async def process_large_dataset(self, dataset_id: str):
        async for record in self.fetch_records(dataset_id):
            processed = await self.process_record(record)
            self.processed_count += 1  # Fast update
            yield {"record": processed, "count": self.processed_count}
```

**Why mixed?** Different operations have different characteristics - use the right tool for each.

---

## 7. Performance Comparison: Understanding the Difference

### Scenario: Process 1000 requests

#### Sync Method (Sequential Execution)

```python
def process(self, data: str) -> str:
    return process_data(data)  # 2ms each
```

**Execution Timeline**:
```
Request1: [████] 2ms
Request2:      [████] 2ms
Request3:          [████] 2ms
...
Request1000:                    [████] 2ms
Total: 2000ms (2 seconds)
```

**Why slow?** Must wait for previous request to complete before processing next.

#### Async Method (Concurrent Execution)

```python
async def process(self, data: str) -> str:
    result = await external_api(data)  # 50ms each (waiting for network)
    return result
```

**Execution Timeline**:
```
Request1-1000: [████████████████████████████████] 50ms (all concurrent)
Total: ~50ms (not 50 seconds!)
```

**Why fast?** All requests execute concurrently, Actor can process other requests while waiting for network.

#### Streaming (Incremental Return)

```python
async def process(self, data: str):
    for chunk in split_data(data):
        result = await process_chunk(chunk)
        yield result  # Return immediately
```

**Execution Timeline**:
```
Client receives first result: [██] 10ms  ← See immediately!
Client receives all results:   [████████████████████] 50ms
```

**Why better?** Users don't need to wait for everything, can start processing first result immediately.

### Key Understanding

- **Sync**: Sequential execution, simple but slow (good for fast operations)
- **Async**: Concurrent execution, fast but requires `async/await` (good for I/O operations)
- **Streaming**: Incremental return, better UX (good for long-running operations)

---

## 8. Common Pitfalls: Understanding Why They're Wrong

### ❌ Pitfall 1: Using Sync for I/O

**Problem**: Blocks Actor, cannot process other messages.

```python
# ❌ Bad: Blocks Actor during HTTP request
def fetch_data(self, url: str) -> dict:
    response = requests.get(url)  # Blocks for seconds!
    return response.json()
# Result: Actor cannot process any other messages during these seconds
```

**Why wrong?**
- Actor is blocked, cannot process other requests
- Extremely low throughput (can only process one request at a time)
- Poor user experience (all requests queue up)

```python
# ✅ Good: Non-blocking async
async def fetch_data(self, url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)  # Can process other requests while waiting
        return response.json()
# Result: Actor can process multiple requests concurrently
```

### ❌ Pitfall 2: Using Async for Fast Operations

**Problem**: Adds unnecessary complexity with no performance benefit.

```python
# ❌ Bad: Unnecessary async overhead
async def increment(self, n: int) -> int:
    self.value += n  # This operation only takes < 1ms
    return self.value
# Problem: Operation is too fast, concurrency provides zero benefit, but code is more complex
```

**Why wrong?**
- Operation is too fast (< 1ms), doesn't need concurrency
- Adds `async/await` syntax complexity
- No performance improvement

```python
# ✅ Good: Simple sync method
def increment(self, n: int) -> int:
    self.value += n
    return self.value
# Result: Simpler code, same performance
```

### ❌ Pitfall 3: Not Using Streaming for LLM

**Problem**: Poor user experience, long wait time.

```python
# ❌ Bad: Wait for all tokens
async def generate(self, prompt: str) -> str:
    tokens = []
    async for token in self.llm_client.stream(prompt):
        tokens.append(token)
    return "".join(tokens)  # User waits 10-30 seconds to see anything
# Problem: User must wait for everything, cannot cancel early
```

**Why wrong?**
- Users must wait 10-30 seconds to see any output
- Cannot cancel early (must wait even if not needed)
- Extremely poor user experience

```python
# ✅ Good: Stream tokens as they arrive
async def generate(self, prompt: str):
    async for token in self.llm_client.stream(prompt):
        yield token  # User sees tokens immediately
# Result: Users see output immediately, can cancel early
```

### ❌ Pitfall 4: Using Ask for Fire-and-Forget

**Problem**: Unnecessary waiting, reduces throughput.

```python
# ❌ Bad: Unnecessary waiting
await logger.ask({"level": "info", "msg": "..."})  # Waits for response
# Problem: Even though you don't need result, you wait, reducing throughput
```

**Why wrong?**
- Don't need response, but still wait
- Reduces throughput (all logging operations must wait)
- Increases latency

```python
# ✅ Good: Fire and forget
await logger.tell({"level": "info", "msg": "..."})  # No waiting
# Result: Maximize throughput, no blocking
```

---

## 9. Best Practices Summary

### Core Principles

1. **Fast operations (< 10ms)**: Use `def method()` (sync)
   - **Reason**: Fast enough, no concurrency needed, simpler code

2. **I/O operations (> 10ms)**: Use `async def method()` (async)
   - **Reason**: Avoid blocking Actor, allow concurrent processing

3. **Incremental results**: Use `async def method()` with `yield` (streaming)
   - **Reason**: Return partial results immediately, better UX

4. **No response needed**: Use `tell()` (fire-and-forget)
   - **Reason**: Maximize throughput, no blocking

5. **Need response**: Use `ask()` or method call
   - **Reason**: Need to know operation result or success status

6. **LLM token generation**: Always use streaming
   - **Reason**: Generation takes time, users want to see output immediately

7. **Multiple concurrent operations**: Use `async def` with `asyncio.gather()`
   - **Reason**: Concurrent execution, not sequential

---

## 10. Quick Reference

| Operation Type | Pattern | Why |
|----------------|---------|-----|
| Counter increment | `def increment()` | Fast (< 1ms), no concurrency needed |
| HTTP request | `async def fetch()` | Network I/O (> 50ms), needs concurrency |
| Database query | `async def query()` | I/O operation, needs concurrency |
| LLM generation | `async def generate()` with `yield` | Long time, users want immediate output |
| File processing | `async def process()` with `yield` | Large data, incremental processing avoids memory overflow |
| Logging | `tell()` | No response needed, maximize throughput |
| Metrics | `tell()` | No response needed, maximize throughput |
| Get result | `ask()` or `await method()` | Need to know operation result |

---

## Summary: Understanding Design Principles

### Core Ideas

1. **Actor processes one message at a time**: This is a fundamental guarantee of the Actor model
2. **Blocking is a performance killer**: If Actor is blocked, cannot process other messages
3. **Async yields control**: `await` yields control, allowing processing of other messages
4. **Streaming improves UX**: Return partial results immediately, don't wait for everything

### Selection Principles

- **Simplicity first**: If sync is enough, use sync
- **Avoid blocking**: I/O operations must use async
- **User experience**: Long-running operations use streaming
- **Throughput first**: No response needed, use `tell()`

---

## Next Steps

- Learn about [Error Handling](../guide/reliability.md#error-handling) for robust communication
- Check [Reliability Guide](reliability.md) for timeout and retry patterns
- See [Examples](../examples/index.md) for more real-world patterns
