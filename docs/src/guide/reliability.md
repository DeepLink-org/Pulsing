# Reliability (timeouts, retries, restarts)

This page collects **practical reliability rules** for building production systems with Pulsing.

## TL;DR

- Treat network + nodes as unreliable: **use timeouts** and design for failures.
- Pulsing does **not** provide end-to-end exactly-once semantics: **design idempotent handlers**.
- Pulsing supports **actor-level restart** (no supervision tree): use it for crash recovery, not for correctness.

## Timeouts

Prefer explicit timeouts on `ask`:

```python
result = await asyncio.wait_for(ref.ask({"op": "compute"}), timeout=10.0)
```

For proxy method calls: `await asyncio.wait_for(proxy.compute(), timeout=10.0)`.

## Retries (application-level)

Pulsing does not hide retries for you. If you retry, assume duplicates are possible.

Recommended pattern:

- **idempotency key** in every request
- **dedup** in the actor state (or an external store)

## Actor-level restart (supervision)

You can configure restart policy on Python actors created via `@pul.remote`:

```python
import pulsing as pul

@pul.remote(restart_policy="on_failure", max_restarts=5, min_backoff=0.2, max_backoff=10.0)
class Worker:
    def work(self, x: int) -> int:
        return 100 // x
```

### What it is (and isn’t)

- **Is**: a crash-recovery mechanism for actor instances (with backoff and restart limits)
- **Is not**: a supervision tree, and **not** an exactly-once guarantee

## Error Handling

Pulsing distinguishes between framework errors and actor execution errors, enabling appropriate recovery strategies.

### Error Categories

- **Framework errors** (`PulsingRuntimeError`): Network failures, cluster issues, configuration errors, actor system errors
- **Actor errors** (`PulsingActorError`): Errors from user code
  - **Business errors** (`PulsingBusinessError`): User input validation failures (recoverable, return to caller)
  - **System errors** (`PulsingSystemError`): Internal processing failures (may trigger actor restart)
  - **Timeout errors** (`PulsingTimeoutError`): Operation timeouts (retryable)

### Error Recovery Strategies

1. **Business errors**: Return to caller, don't retry
   ```python
   except PulsingBusinessError as e:
       # User input issue - return error to caller
       return {"error": e.message, "code": e.code}
   ```

2. **System errors**: Check `recoverable` flag, may trigger actor restart
   ```python
   except PulsingSystemError as e:
       if e.recoverable:
           # May retry or wait for actor restart
           # Actor will restart if restart_policy is configured
           pass
       else:
           # Non-recoverable - log and fail
           logger.error(f"Non-recoverable error: {e.error}")
   ```

3. **Timeout errors**: Retry with backoff
   ```python
   except PulsingTimeoutError as e:
       # Retry with exponential backoff
       await asyncio.sleep(backoff_seconds)
       return await retry_operation()
   ```

4. **Framework errors**: Log and handle at application level
   ```python
   except PulsingRuntimeError as e:
       # Network/cluster issue - log and handle at app level
       logger.error(f"Framework error: {e}")
       # May need to retry or failover
   ```

### Example: Comprehensive Error Handling

```python
from pulsing.exceptions import (
    PulsingBusinessError,
    PulsingSystemError,
    PulsingTimeoutError,
    PulsingRuntimeError,
)

async def process_with_retry(actor, data, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await actor.process(data)
        except PulsingBusinessError as e:
            # Don't retry business errors
            raise
        except PulsingSystemError as e:
            if not e.recoverable:
                raise
            # Wait for actor restart, then retry
            await asyncio.sleep(2 ** attempt)
        except PulsingTimeoutError:
            # Retry timeout errors
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise
        except PulsingRuntimeError as e:
            # Framework error - may need failover
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise
```

## Streaming resilience

For streaming responses, assume partial streams are possible. Make chunks independently meaningful:

- include `seq` / offsets / ids per chunk
- allow resume or dedup on the client side
