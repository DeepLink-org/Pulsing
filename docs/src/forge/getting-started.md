# Getting Started

## Install

```bash
pip install pulsing
# Optional extras if defined in your distribution:
# pip install pulsing[forge]
```

For Rust-accelerated handlers and MCP runtime:

```bash
uv run maturin develop   # from repo root
```

---

## Minimal example

```python
from pulsing.forge import ForgeEnvironment, LocalToolSession

env = ForgeEnvironment(
    cwd=".",
    sandbox_policy="off",
    session=LocalToolSession(token_budget=128_000),
)
rt = env.runtime()

out = rt.call_tool("Glob", {"pattern": "*.md", "path": "."})
print(out.content, out.is_error)
```

After `maturin develop`, `env.runtime()` defaults to **`HybridForgeRuntime`**: Rust handlers first, Python fallback for Host-only tools — all **32 tools callable**.

---

## Host session hooks

Tools like `request_user_input` and `update_plan` call back into **`ToolSession`**:

```python
from pulsing.forge import ForgeEnvironment, LocalToolSession

session = LocalToolSession()
session.user_input = lambda args: {"answers": {"confirm": "yes"}}

env = ForgeEnvironment(session=session)
env.runtime().call_tool("request_user_input", {"questions": [...]})
```

Forge emits structured requests; your Host decides how to show UI or auto-approve.

---

## Isolated worker (Pulsing Actor)

```python
import pulsing as pul
from pulsing.forge import ToolWorkerActor, ToolWorkerConfig

await pul.init()
try:
    worker = await ToolWorkerActor.spawn(ToolWorkerConfig(cwd="."), public=False)
    pong = await worker.ping()
    out = await worker.Read(file_path="README.md")
finally:
    await pul.shutdown()
```

Unified deployment: [`ForgeBackend`](deployment.md) (LOCAL / DEDICATED / SHARED).

---

## REPL (no LLM)

```bash
python -m pulsing.forge.repl
pulsing forge repl
```

Drive tools directly, save JSONL traces, replay steps. See [Session REPL design](../design/forge/session-repl.md).

---

## Verify install

```bash
pytest tests/python/test_pulsing_forge.py tests/python/test_hybrid_forge_callable.py -q
```

Example script: `examples/python/forge_minimal.py`

---

## Next steps

- [Concepts](concepts.md) — mental model
- [Tools (32)](tools.md) — inventory
- [Deployment on Pulsing](deployment.md) — cluster path
- [Craft integration](../design/forge/craft-architecture.md) — reference Host
