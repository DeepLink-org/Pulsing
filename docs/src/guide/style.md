# Terminology and Style

| Symbol | Meaning |
|--------|---------|
| `ActorSystem` | Runtime managing actors and cluster connectivity |
| `Actor` | Message-handling interface |
| `ActorRef` | Low-level reference for messaging |
| `ActorProxy` | Python method-call proxy returned by `@remote` |

Use user-defined class paths in CLI examples:

```bash
pulsing actor my_app.Counter --addr 0.0.0.0:8000 --name counter -- --initial 10
pulsing inspect actors --endpoint 127.0.0.1:8000
```

Arguments after `--` belong to the actor constructor.
Inspection is observer-only and does not join the cluster.

Use backticks for symbols and paths, PEP 8 for Python, `cargo fmt` for Rust,
and `.zh.md` for Chinese documentation.
