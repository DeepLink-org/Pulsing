# Reliability (timeouts, retries, restarts)

This page collects **practical reliability rules** for building production systems with Pulsing.

## TL;DR

- Treat network + nodes as unreliable: **use timeouts** and design for failures.
- Pulsing does **not** provide end-to-end exactly-once semantics: **design idempotent handlers**.
- Pulsing supports **actor-level restart** (no supervision tree): use it for crash recovery, not for correctness.

## Timeouts

Prefer explicit timeouts on `ask`:

```python
from pulsing.actor import ask_with_timeout

result = await ask_with_timeout(ref, {"op": "compute"}, timeout=10.0)
```

## Retries (application-level)

Pulsing does not hide retries for you. If you retry, assume duplicates are possible.

Recommended pattern:

- **idempotency key** in every request
- **dedup** in the actor state (or an external store)

## Actor-level restart (supervision)

You can configure restart policy on Python actors created via `@remote`:

```python
from pulsing.actor import remote

@remote(restart_policy="on-failure", max_restarts=5, min_backoff=0.2, max_backoff=10.0)
class Worker:
    def work(self, x: int) -> int:
        return 100 // x
```

### What it is (and isn’t)

- **Is**: a crash-recovery mechanism for actor instances (with backoff and restart limits)
- **Is not**: a supervision tree, and **not** an exactly-once guarantee

## Streaming resilience

For streaming responses, assume partial streams are possible. Make chunks independently meaningful:

- include `seq` / offsets / ids per chunk
- allow resume or dedup on the client side
