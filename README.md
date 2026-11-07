# Pulsing

[中文文档](README.zh.md)

A distributed actor runtime built in Rust, designed for Python.
Pulsing provides location-transparent actor calls, streaming messages, and
built-in cluster discovery without Redis or etcd.

## Install

```bash
pip install pulsing
```

## First actor

```python
import asyncio
import pulsing as pul


@pul.remote
class Counter:
    def __init__(self, value=0):
        self.value = value

    def inc(self):
        self.value += 1
        return self.value


async def main():
    await pul.init()
    try:
        counter = await Counter.spawn(value=0, name="counter")
        print(await counter.inc())  # 1
    finally:
        await pul.shutdown()


asyncio.run(main())
```

For a cluster, initialize each node with `addr` and join using `seeds`.
Resolve named actors with `await Counter.resolve("counter")`.

## Scope

- Actor mailboxes, lifecycle, supervision, and named actor discovery.
- Local and remote request/reply, one-way calls, and streaming responses.
- HTTP/2 transport, TLS, and opaque tensor-buffer transport.
- Python `@remote`, subprocess integration, queues, and pub/sub.
- CLI actor startup and cluster inspection.
- Workspace scaffolding, checkpoints, rollback, and Python workflows.

Inference servers, LLM clients, Forge tools and coding agents, the desktop GUI,
and benchmark modules are no longer bundled. Applications can implement their
own services on top of the actor API.

## Agent mailbox prototype

An optional coordination layer gives each router, worker, reviewer, or subagent
its own durable actor mailbox. External agents use MCP tools and an installable
skill to send mail and track unread, read, processing, completed, and failed
states. No LLM runtime or coding tools are bundled.

```bash
pip install 'pulsing[mailbox]'
python -m pulsing.mailbox serve --state-dir .pulsing/agent-mail
# In the agent's stdio MCP configuration:
python -m pulsing.mailbox mcp --identity router --seed 127.0.0.1:8100
```

This is a trusted-local-cluster prototype, not a public email service.
See the [design and setup guide](docs/src/design/agent-mailboxes.md).

## CLI

```bash
pulsing actor my_app.Counter --addr 0.0.0.0:8000 --name counter
pulsing inspect actors --endpoint 127.0.0.1:8000
pulsing init --template minimal
pulsing checkpoint -m "working state"
pulsing history
pulsing rollback
```

The Rust CLI also supports `pulsing run [script.py]` (alias `workflow`).
Workflows execute once and return to the shell, without an LLM session.

## Development

Requires Python 3.10+ and a current stable Rust toolchain. The RustPython CLI
requires Rust 1.93 or newer.

```bash
uv sync --extra dev
uv run maturin develop
just test
just fmt
just lint
```

`just build-wheel` builds the Python wheel; `just build-binary` builds the
RustPython CLI. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## Project layout

| Path | Purpose |
|------|---------|
| `crates/pulsing-actor/` | Actor runtime, discovery, transport |
| `crates/pulsing-py/` | CPython/PyO3 bindings |
| `crates/pulsing-rpymod/` | RustPython bindings |
| `crates/pulsing-bindings-core/` | Shared binding data types |
| `crates/pulsing-cli/` | Rust CLI |
| `crates/pulsing-workspace/` | Workspace scaffolding and journal |
| `python/pulsing/` | Python SDK and integrations |
| `examples/` | Actor and transport examples |
| `docs/` | Documentation |

## License

Apache-2.0
