# Forge Naming

| Symbol | Package / path |
|--------|----------------|
| **Pulsing Forge** | `pulsing.forge` · crate `pulsing-forge` |
| **Pulsing** | Actor runtime |
| **Craft** | Reference Multi-Agent app |

## Gossip names

| Resource | Pattern |
|----------|---------|
| Craft agent | `craft/ws/{workspace_id}/{short_name}` |
| Shared tool worker | `craft/ws/{workspace_id}/_tools` |
| Forge event inbox | `{host}/events` |
| MCP hub | `craft/ws/{workspace_id}/_mcp_hub` |
| Code cell registry | `{host}/code_cells` |

Implementation: `python/pulsing/forge/naming.py`

Details: **`naming.zh.md`** (Chinese, primary).
