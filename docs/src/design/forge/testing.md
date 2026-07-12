# Forge Testing

> **Audience**: Forge contributors, CI maintainers
> **See also**: [engineering.md](./engineering.md) · [session-repl.md](./session-repl.md)

Forge tests are organized in **L0–L4** layers. We reuse portable fixture layouts from the agent-tool ecosystem (notably `apply_patch` scenario directories) while asserting **Pulsing Forge** behavior.

Full detail: **`testing.zh.md`** (Chinese, primary).

---

## Quick commands

```bash
pytest tests/python/forge/ -m forge -q
cargo test -p pulsing-forge apply_patch_scenarios
maturin develop   # required for Hybrid L1/L2
```

Shared harness: `python/pulsing/testing/forge_harness.py`
