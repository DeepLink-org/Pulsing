# Python API Reference

This page contains the complete auto-generated API documentation for Pulsing's Python interface.

## Installation

Pulsing requires Python 3.10+ and can be installed via pip:

```bash
pip install pulsing
```

For development, clone the repository and install in development mode:

```bash
git clone https://github.com/DeepLink-org/pulsing
cd pulsing
pip install -e .
```

## Core Module

::: pulsing

## Actor Module

::: pulsing.core

## Agent Module

::: pulsing.agent

## Queue Module

::: pulsing.streaming

## Sync API Event Loop Requirements

Pulsing's synchronous wrappers, including `transfer_queue.get_client()`,
`queue.sync()` / `reader.sync()`, and the Pulsing-backed path in
`pulsing.subprocess`, submit work onto a dedicated Pulsing event loop.

- Call them from synchronous code, or from another thread.
- Do not call them from the active async / Pulsing event loop thread.
- In async code, prefer the native async APIs, or move the sync entrypoint and
  subsequent sync calls into `asyncio.to_thread(...)`.

## Subprocess Module

`pulsing.subprocess` provides a subprocess-compatible synchronous API.
Without `resources`, calls fall back to Python's native `subprocess`.
When `resources` is provided and `USE_POLSING_SUBPROCESS=1` is set, the
command is executed through Pulsing's backend.

::: pulsing.subprocess
