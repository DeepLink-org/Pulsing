# Pulsing Forge

**Agent tool and environment runtime** for coding agents — sandboxed shell, filesystem, session tools, MCP, and plugins in one embeddable library.

> **Snapshot**: 2026-05 · **32 tools callable out of the box** (Hybrid + MCP runtime)
> **Import**: `pip install pulsing` → `from pulsing.forge import ForgeEnvironment`

---

## One sentence

Forge answers: **given a tool name and JSON arguments, how do we execute safely in a workspace?**

It is **not** a chat product or CLI clone. **Host** owns LLM and UI; **Forge** owns environment and tool execution.

---

## Why Forge

| Pain | Forge provides |
|------|----------------|
| Reimplementing Read / shell / patch per project | One tool surface + `ToolResult` |
| Scattered sandbox policy | `sandbox_policy`: `off` / `restricted` / `bwrap` |
| Plan / prompts coupled to tools | `ToolSession` callbacks — Forge has no built-in UI |
| Dev vs isolated execution drift | Same 32 tool names: in-process or `ToolWorkerActor` |

---

## Ecosystem

```text
Pulsing        Distributed Actor runtime (optional)
Pulsing Forge  Tool + environment library (this chapter)
Craft          Multi-Agent reference app (Forge reference Host)
```

---

## Three deployment modes

| Mode | API | Use when |
|------|-----|----------|
| **In-process** | `ForgeEnvironment.runtime()` | Tests, prototypes |
| **Hybrid (default)** | `HybridForgeRuntime` after `maturin develop` | Production host tools |
| **Isolated actor** | `ForgeBackend` + `ToolWorkerActor` | Sandbox in child process / cluster |

Same LLM tool schema across modes.

---

## Quick start

```python
from pulsing.forge import ForgeEnvironment, LocalToolSession

env = ForgeEnvironment(cwd=".", session=LocalToolSession())
rt = env.runtime()
rt.call_tool("shell_command", {"cmd": "pytest -q", "workdir": "."})
```

See [Getting Started](getting-started.md) · [Abstractions](abstractions.md) · [Deployment on Pulsing](deployment.md).

---

## Documentation map

| Doc | Audience |
|-----|----------|
| [Getting Started](getting-started.md) | Install, first `call_tool`, tests |
| [Concepts](concepts.md) | Host vs Forge, core types |
| [Abstractions](abstractions.md) | Environment, Session, tool domains |
| [Tools (32)](tools.md) | Tool inventory by domain |
| [Deployment on Pulsing](deployment.md) | `ForgeBackend`, actors, isolation |
| [Pulsing Integration](pulsing-integration.md) | ask/tell, inbox, MCP hub, code cells |

**Design deep dives** (Architecture & Design → Forge):

- [Engineering](../design/forge/engineering.md) · [Craft architecture](../design/forge/craft-architecture.md)

**Package README**: [python/pulsing/forge/README.md](https://github.com/DeepLink-org/pulsing/blob/main/python/pulsing/forge/README.md)

---

## License

Apache-2.0 · Third-party notices in `crates/pulsing-forge/NOTICE`
