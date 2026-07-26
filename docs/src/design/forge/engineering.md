# Forge Engineering Notes

> **User docs**: [Forge chapter](../../forge/index.md) · [Abstractions](../../forge/abstractions.md)
> **Package API**: [python/pulsing/forge/README.md](https://github.com/DeepLink-org/pulsing/blob/main/python/pulsing/forge/README.md)
> **Target architecture**: [Forge Core Architecture](core-architecture.md). This page describes the current implementation.

Implementation-focused notes for `pulsing-forge` (Rust) and `pulsing.forge` (Python).

---

## Crates and modules

| Layer | Path |
|-------|------|
| Rust handlers | `crates/pulsing-forge/` |
| PyO3 binding | `crates/pulsing-py/src/forge.rs` → `pulsing._core.ForgeRuntime` |
| Python API | `python/pulsing/forge/` |
| Craft Host | `python/pulsing/craft/agent/forge_*.py` |

---

## Architecture

```text
Host (LLM + ToolSession)
  → HybridForgeRuntime / ToolWorkerActor
  → pulsing-forge ToolRuntime
  → handlers + sandbox + MCP
  → tell_forge_event → ForgeEventInbox → Host
```

---

## Related design docs

| Doc | Topic |
|-----|-------|
| [Core architecture](core-architecture.md) | Target boundaries and versioned protocols |
| [Craft architecture](craft-architecture.md) | Forge × Craft integration |
| [Naming](naming.md) | Package and gossip names |
| [Session REPL](session-repl.md) | `pulsing forge repl` trace/replay |

Full Chinese engineering detail: see `engineering.zh.md` in this directory.
