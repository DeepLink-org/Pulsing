# Pulsing Forge

**Agent 工具与环境运行时** — 在可配置沙箱中执行 shell、文件、Session 工具、MCP 与插件。

```python
from pulsing.forge import ForgeEnvironment, LocalToolSession

env = ForgeEnvironment(cwd=".", session=LocalToolSession())
rt = env.runtime()
rt.call_tool("Read", {"file_path": "README.md"})
```

---

## 文档

| 文档 | 说明 |
|------|------|
| **[Pulsing 文档 · Forge 章节](https://github.com/DeepLink-org/pulsing/tree/main/docs/src/forge)** | 用户指南（概述、快速开始、部署、集成） |
| **本地预览** | `cd docs && uv run mkdocs serve` → 导航 **Pulsing Forge** |
| **设计与实现** | [`docs/src/design/forge/`](https://github.com/DeepLink-org/pulsing/tree/main/docs/src/design/forge) |

---

## 安装

```bash
pip install pulsing
uv run maturin develop   # Rust Hybrid 路径（仓库开发）
```

---

## 核心 API

| 类型 | 作用 |
|------|------|
| `ForgeEnvironment` | 推荐入口：工作区 + 沙箱 + `ToolSession` |
| `ForgeBackend` | Pulsing 部署统一入口（LOCAL / DEDICATED / SHARED） |
| `ToolWorkerActor` | 隔离 worker（`@remote` + `new_process`） |
| `ToolSession` | Host 实现 plan / 用户输入 / token |
| `ToolResult` | `{content, is_error}` |

---

## 工具（32）

| 域 | 示例 |
|----|------|
| 隔离（11） | `Read`, `shell_command`, `apply_patch`, … |
| Host（21） | Session×5, MCP×3, `exec`/`wait`, Extension×8 |

详见文档 [工具清单](https://github.com/DeepLink-org/pulsing/blob/main/docs/src/forge/tools.zh.md)。

---

## Pulsing Actor 示例

```python
import pulsing as pul
from pulsing.forge import ToolWorkerActor, ToolWorkerConfig

await pul.init()
worker = await ToolWorkerActor.spawn(ToolWorkerConfig(cwd="."), public=False)
await worker.shell_command(cmd="pytest -q", workdir=".")
await pul.shutdown()
```

共享 worker gossip 名：`craft/ws/{workspace_id}/_tools`

---

## 生态

```text
Pulsing   → Actor 运行时
Forge     → 工具与环境（本包 pulsing.forge）
Craft     → Multi-Agent 参考应用
```

---

## 测试

```bash
pytest tests/python/test_pulsing_forge.py tests/python/test_hybrid_forge_callable.py -q
cargo test -p pulsing-forge
```

---

## 许可

Apache-2.0 · `crates/pulsing-forge/NOTICE`
