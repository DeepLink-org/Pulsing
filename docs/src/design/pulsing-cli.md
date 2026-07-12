# Pulsing CLI 设计

## 概述

Pulsing 提供一组命令行入口，覆盖 Actor 运行时运维、推理压测、示例浏览，以及基于 workspace 的 **Agent** 交互。各入口共享同一 Python 包 `pulsing`，通过 `pyproject.toml` 注册多个 script。

> **迁移说明**：Workspace Agent CLI 已统一为 `pulsing agent`（实现于 `python/pulsing/cli/agent/`）。`pcraft`、`pulsing craft`、`pulsing.craft` 仍可用但已弃用，会打印警告并转发至 `pulsing agent`。详见 [agent-craft-migration.md](../../design/agent-craft-migration.md)。

本文档描述**全部** CLI 入口、命令树、行为约定与代码布局。Workspace 持久化见 [Workspace 持久化](./npc-workspace-world-persistence.md)。

**实现位置**：

| 区域 | 路径 |
|------|------|
| 主 CLI | `python/pulsing/cli/` |
| Agent CLI | `python/pulsing/cli/agent/` |
| Agent SDK | `python/pulsing/agent/`（actors、loop、workspace、cluster） |
| Craft shim（弃用） | `python/pulsing/craft/` |

---

## 设计目标

1. **职责分离** — `pulsing` 承载运行时与集群运维；Workspace Agent 子命令集中在 `pulsing agent`。
2. **稳定主路径** — 生产与文档以 `pulsing agent` 为准；`pcraft` / `pulsing craft` 为弃用 shim。
3. **参数约定** — `pulsing actor` 使用 `--` 分隔进程级参数与 Actor 构造参数；Agent CLI 使用 `argparse` 子命令树。
4. **Observer 与 Member 分离** — `pulsing inspect` 仅 HTTP 观察，不加入 gossip；`pulsing agent wake` 与 `pulsing actor` 启动完整节点。
5. **向后兼容** — 旧 `npc` / `puzzle` 子命令通过 shim 映射至 `spawn` / `say` / `task`。

---

## 入口注册

`pyproject.toml` → `[project.scripts]`：

| 命令 | 入口 | 说明 |
|------|------|------|
| `pulsing` | `pulsing.cli.__main__:main` | 主 CLI |
| `pulsing-agent` | `pulsing.cli.agent.main:main` | Agent workspace CLI（推荐） |
| `pcraft` | `pulsing.craft.cli:main` | **弃用**；转发至 `pulsing agent` |

`pulsing` 在 `main()` 中对 `argv[1] == "craft"`、`argv[1] == "forge"` 做 early dispatch，其余子命令由 `hyperparameter` 解析。

---

## 4. 总命令树

```text
pulsing
├─ actor TYPE [--addr …] [--seeds …] [--name …] [-- …]
├─ inspect {cluster|actors|metrics|watch} …
├─ bench MODEL …
├─ examples [NAME]
├─ craft …                         → 见 §9；与 pcraft 相同
└─ forge repl …                    → Forge session REPL（Rust 优先）

pcraft …                           → §9

python -m pulsing.craft …             → §10（调试）
```

---

## 5. `pulsing actor`

启动一个 Actor 服务进程并加入（或组建）集群。

### 命令形式

```text
pulsing actor <module.path.ClassName> [--addr ADDR] [--seeds SEEDS] [--name NAME] [-- …]
```

- **`actor_type`**（ positional）：完整类路径，必须含 `.`。
- **`--addr`**：本节点绑定地址，如 `0.0.0.0:8000`。
- **`--seeds`**：逗号分隔的 seed 节点，用于加入已有集群。
- **`--name`**：Actor 注册名（默认 `worker`）。
- **`--` 之后**：Actor 构造函数 kwargs，经 `_actor_argv_rewrite` 注入 `extra_kwargs`。

### 行为

- 调用 `pulsing.cli.actors.start_generic_actor`。
- `pulsing actor list` 已移除；提示改用 `pulsing inspect actors`。

### 示例

```bash
pulsing actor pulsing.serving.Router --addr 0.0.0.0:8000 --name my-llm -- \
  --http_port 8080 --model_name my-llm

pulsing actor pulsing.serving.TransformersWorker --seeds 127.0.0.1:8000 -- \
  --model_name gpt2 --device cpu
```

---

## 6. `pulsing inspect`

通过 HTTP **观察者模式**查看集群状态，**不**加入 gossip。

### 子命令

```text
pulsing inspect cluster --seeds SEEDS [--timeout …] [--best_effort]
pulsing inspect actors (--seeds SEEDS | --endpoint ADDR) [--top N] [--filter …]
                       [--json] [--detailed] [--all_actors]
pulsing inspect metrics --seeds SEEDS [--raw …]
pulsing inspect watch --seeds SEEDS [--interval …] [--kind …] [--max_rounds …]
```

| 子命令 | 作用 |
|--------|------|
| `cluster` | 成员列表 |
| `actors` | Actor 分布；`--endpoint` 与 `--seeds` 互斥 |
| `metrics` | 节点 metrics |
| `watch` | 周期性刷新；`kind`: cluster / actors / metrics / all |

公共选项：`--timeout`（默认 10s）、`--best_effort`。

---

## 7. `pulsing bench`

对 LLM 推理端点做压测，内部使用 Actor 架构采集指标。

### 命令形式

```text
pulsing bench MODEL [--url URL] [--max_vus N] [--duration …] [--warmup …]
                    [--benchmark_kind …] [--num_workers …] …
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `MODEL` | — | 模型名（ positional） |
| `--url` | `http://localhost:8000` | 后端 URL |
| `--max_vus` | 128 | 最大并发 |
| `--duration` | `120s` | 压测时长 |
| `--warmup` | `30s` | 预热 |
| `--benchmark_kind` | `throughput` | throughput / sweep / csweep / rate |
| `--num_workers` | 4 | worker Actor 数 |
| `--tokenizer` | 同 MODEL | HF tokenizer |
| `--rates` | — | rate 模式下的速率列表 |

---

## 8. `pulsing examples`

列出或查看内置示例。

```text
pulsing examples
pulsing examples NAME
```

无 `NAME` 时打印列表；有 `NAME` 时打印 docstring 与 `python -m pulsing.examples.{name}` 运行方式。
`pulsing examples foo` 在 `main()` 中 rewrite 为 `--name foo`。

---

## 9. Craft（`pulsing craft` / `pcraft`）

Craft 在项目目录内管理 workspace（`.pulsing/`）、启动本地节点、spawn 命名 Agent（NPC），并通过 shell 与 Puzzle 配置交互。

### 9.1 术语

| 术语 | 含义 |
|------|------|
| **Workspace** | 项目根 + `.pulsing/`；`cluster_id` 由根路径导出 |
| **Operator** | CLI 操作者；显示名 `PLAYER` / `HERO` / `USER` |
| **NPC** | gossip 命名 Agent（如 `guide`）；实现为 `HubActor` |
| **Puzzle** | `cluster.json` 内任务项 |
| **Node record** | `.pulsing/node.json`：当前 `wake` 的 addr、pid |

### 9.2 等价入口

```text
pcraft ARGS…  ≡  pulsing craft ARGS…
```

### 9.3 命令树

```text
pcraft [ -h | --help ]

├─ (default)                         → look

├─ init                              写入 .pulsing/cluster.json
│
├─ wake                              启动节点并 spawn NPC（阻塞至信号）
│   ├─ --agents AGENTS
│   ├─ --addr ADDR                   默认 127.0.0.1:0
│   ├─ --auto-approve
│   ├─ --provider {anthropic,openai}
│   └─ --model MODEL
│
├─ look                              打印 workspace 摘要
│
├─ sleep                             [未实现] 快照后停止 node
│
├─ npc
│   ├─ who
│   ├─ summon NAME [--role …] [--provider …] [--model …]
│   └─ say NAME MESSAGE… [--puzzle ID 未实现]
│
└─ puzzle
    ├─ list
    ├─ show ID
    └─ mark ID --status …            [未实现]
```

### 9.4 命令分层

| 层级 | 命令 | 作用对象 |
|------|------|----------|
| Workspace | `init` `wake` `look` `sleep` | 整个 workspace / 本地 node |
| NPC | `npc who` `say` `summon` | 已注册 Agent |
| Puzzle | `puzzle list` `show` `mark` | 配置与进度 |

无参数 → `look`。未 init → 提示 `pcraft init`；需在线 node → 提示 `pcraft wake`。

### 9.5 Legacy 映射（`npc` 入口）

| 输入 | 解析结果 |
|------|----------|
| _(empty)_ | `look` |
| `seed` | `init` |
| `puzzles` | `puzzle list` |
| `who` / `say` / `summon` | `npc …` |

### 9.6 Workspace 布局

```text
.pulsing/
  cluster.json       静态配置
  node.json          当前 wake（临时）
  puzzle_state.json  [未实现]
  snapshot/          [未实现] 见持久化文档
```

### 9.7 命令与运行时

| 命令 | 连接集群 | 启动 node | 副作用 |
|------|----------|-----------|--------|
| `init` | 否 | 否 | 写 `cluster.json` |
| `look` | 可选 | 否 | 只读 |
| `wake` | 是 | 是 | spawn NPC；写 `node.json` |
| `npc *` | 是 | 否 | RPC |
| `puzzle list/show` | 否 | 否 | 读配置 |
| `sleep` | 是 | 否 | [未实现] 快照 |

NPC gossip 全名：`craft/ws/<cluster_id>/<short_name>`；CLI 使用短名。

### 9.8 子命令行为（摘要）

- **`init`**：创建 workspace；已存在则 exit 0 并提示。
- **`wake`**：`workspace_session` → spawn → 阻塞 → `clear_node_record`。
- **`look`**：`render_look`；含 operator、路径、node、puzzle、NPC 列表。
- **`npc say`**：`resolve_cluster_agent` → `receive_agent_message`。
- **`puzzle show`**：未知 id 时 exit 2。

`look` 输出格式为稳定契约，见 `pulsing.craft.workspace.world_view.render_look`。

### 9.9 Craft 示例

```bash
pcraft init
pcraft wake --agents guide    # terminal 1
pcraft look
pcraft npc say guide "run tests"
pcraft puzzle list
```

---

## 10. 调试入口（`python -m pulsing.craft`）

不在主 CLI 稳定面内；供开发与回归。

```text
python -m pulsing.craft {session|minimal|hub|cluster|agent|ctl|legacy} …
```

| 子命令 | 说明 |
|--------|------|
| `session` / `minimal` | 最小 REPL，无 LLM |
| `hub` | 完整 craft hub + REPL/TUI |
| `cluster` | 本地 agent 集群 + 控制 REPL |
| `agent` | spawn 单个命名 cluster agent |
| `ctl` | 已有集群的控制 REPL |
| `legacy` | 进程内 sync Engine |

首参数以 `-` 开头时默认 `hub`。

---

## 11. 模块结构

```text
python/pulsing/
  cli/
    __main__.py           pulsing 入口；craft dispatch；top-level help
    actor_argv.py         pulsing actor ``--`` rewrite
    help_text.py          grouped ``pulsing`` help
    actors.py             pulsing actor 启动
    inspect.py            pulsing inspect
    bench.py              pulsing bench
  craft/
    cli.py                entry: pcraft / pulsing craft / deprecated npc
    parser.py             argparse
    normalize.py          legacy argv mapping
    dispatch.py           route parsed args to handlers
    helpers.py            shared session/spawn helpers
    commands/
      world.py            init, wake, look, sleep
      npc.py              who, say, summon
      puzzle.py           list, show, mark
    runtime/              hub, engine, tools
    workspace/            config, session, world_view
    cluster/              discovery, messaging
    payload/              isolated workers
    __main__.py           debug: hub, cluster, session
  examples/               pulsing examples 数据源
```

---

## 12. 环境变量

| 变量 | 适用 | 说明 |
|------|------|------|
| `PLAYER` / `HERO` / `USER` | Craft | Operator 显示名 |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Craft | LLM |
| `HF_TOKEN` | bench | HuggingFace 私有模型 |

---

## 13. 测试

| 目录 / 文件 | 覆盖 |
|-------------|------|
| `tests/python/craft/test_cli.py` | parser、normalize、init/look |
| `tests/python/craft/test_world.py` | look、puzzle 视图 |
| `tests/python/craft/test_workspace.py` | workspace config、scoped discovery |
| `tests/python/craft/test_agent.py` | CraftAgent spawn、RPC、mailbox handoff |
| `tests/python/craft/test_npc.py` | `spawn_npc`、say/whisper、class 注册 |
| `tests/python/craft/test_cluster.py` | 集群命名、gossip discovery |
| `tests/python/craft/test_tools.py` | 隔离 worker、split_tools |
| `tests/python/craft/test_coordinator.py` | delegate 通知 XML |
| `test_pulsing_cli_helpers.py` | actor argv rewrite |

---

## 14. 实现阶段

| 阶段 | 范围 | 状态 |
|------|------|------|
| P0 | `pulsing` 子命令 + `pcraft` / `pulsing craft` 现有 Craft 树；`npc` deprecated | 已完成 |
| P0.5 | Craft 拆包；`pulsing --help` 分组展示 | 已完成 |
| P1 | Craft `sleep`、快照；`wake --detach` | 未开始 |
| P2 | Craft puzzle 闭环（`say --puzzle`、`puzzle mark`） | 未开始 |
| P3 | 移除 `npc` script；收敛 `python -m pulsing.craft` 调试面 | 未开始 |

---

## 15. 非目标

- Craft 不以 TUI/REPL 为主路径（`pulsing.craft hub` 除外）。
- `pulsing actor` 不自动管理 workspace / `.pulsing/`。
- `inspect` 不作为 cluster 成员运行。
- Craft 不支持跨 workspace 全局命令；workspace 由 cwd 向上解析。

---

## 16. 变更记录

| 日期 | 说明 |
|------|------|
| 2026-05 | 初稿：全 CLI 结构；Craft 为 §9；`seed`→`init` |
