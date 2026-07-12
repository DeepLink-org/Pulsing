# Pulsing Agent — Workspace Demo

基于 Pulsing 的 workspace CLI：在项目目录内管理 `.pulsing/`、启动节点、与 Agent 协作。

```bash
cd ~/myproject
pulsing agent init
pulsing agent wake --agents guide    # terminal 1
pulsing agent look                   # default
pulsing agent list
pulsing agent task list
pulsing agent say guide "fix unit-tests"
pulsing agent spawn coder
pulsing agent demo                   # offline demo + optional dashboard
```

安装 LLM 依赖：`pip install pulsing[agent]`

任务配置在 `.pulsing/cluster.json` 的 `puzzles` 字段。

隔离工具由 [Pulsing Forge](../../python/pulsing/forge/README.md) 提供（`ToolWorkerActor`）。

## 已弃用

- `pcraft` / `pulsing craft` → 使用 `pulsing agent`
- `pulsing.craft` 包 → 使用 `pulsing.agent`

见 [agent-craft-migration.md](../../docs/design/agent-craft-migration.md)。
