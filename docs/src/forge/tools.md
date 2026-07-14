# Tools (32)

Forge registers **32 standard tools** covering the main coding-agent path. Definitions: `python/pulsing/forge/integrated.py`.

---

## By deployment

| Partition | Count | Runs on |
|-----------|------:|---------|
| **Isolated** | 11 | `ToolWorkerActor` (child process) |
| **Host** | 21 | `HybridForgeRuntime` / in-process |

Host and isolated sets are **disjoint** — routing is automatic via `ForgeBackend`.

---

## Isolated (11)

Filesystem, shell, patch (sandbox boundary):

`Read` · `Glob` · `Grep` · `Edit` · `Write` · `Bash` · `shell_command` · `exec_command` · `write_stdin` · `apply_patch` · `view_image`

---

## Host (21)

| Domain | Tools |
|--------|-------|
| **Session** | `update_plan`, `new_context`, `get_context_remaining`, `request_user_input`, `request_permissions` |
| **Discovery** | `tool_search`, `request_plugin_install`, `list_resources` |
| **MCP** | `list_mcp_resources`, `list_mcp_resource_templates`, `mcp__*` (dynamic) |
| **Code Mode** | `exec`, `wait` |
| **Extension** | `memories.*`, `skills.*`, `web.run`, `web_search` (8) |

**Python-only Host (10)** when Rust lacks handler: `exec`, `wait`, Extension×8 — covered by Hybrid fallback.

---

## Claude interoperability

Forge also exposes Claude-style names (`Read`, `Bash`, …) on the isolated path — complementary to the standard tool surface.

---

## Dynamic tools

- **Plugin / tool_search** — deferred stubs activated after search
- **MCP** — live tools after `refresh_mcp()`; registered into Host LLM schema via MCP hub sync

---

## Verification

```bash
pytest tests/python/test_hybrid_forge_callable.py -q
pytest tests/python/craft/test_tools.py -q   # Craft schema coverage
pytest tests/python/test_pulsing_forge.py -q
```

---

## Deep reference

- [Engineering](../design/forge/engineering.md) — crate layout and implementation
- Package API table: [python/pulsing/forge/README.md](https://github.com/DeepLink-org/pulsing/blob/main/python/pulsing/forge/README.md)
