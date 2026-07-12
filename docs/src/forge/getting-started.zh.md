# 快速开始

## 安装

```bash
pip install pulsing
```

Rust 加速路径与 MCP runtime（仓库开发）：

```bash
uv run maturin develop
```

---

## 最小示例

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

`maturin develop` 后默认 **`HybridForgeRuntime`**：Rust 优先 + Python fallback，**32 工具均可调用**。

---

## Host 会话钩子

`request_user_input`、`update_plan` 等通过 **`ToolSession`** 回调 Host：

```python
from pulsing.forge import ForgeEnvironment, LocalToolSession

session = LocalToolSession()
session.user_input = lambda args: {"answers": {"confirm": "yes"}}

env = ForgeEnvironment(session=session)
env.runtime().call_tool("request_user_input", {"questions": [...]})
```

---

## 隔离 Worker（Pulsing Actor）

```python
import pulsing as pul
from pulsing.forge import ToolWorkerActor, ToolWorkerConfig

await pul.init()
try:
    worker = await ToolWorkerActor.spawn(ToolWorkerConfig(cwd="."), public=False)
    await worker.ping()
    await worker.Read(file_path="README.md")
finally:
    await pul.shutdown()
```

统一部署见 [Pulsing 部署](deployment.zh.md)（LOCAL / DEDICATED / SHARED）。

---

## REPL（无 LLM）

```bash
python -m pulsing.forge.repl
```

直接调工具、保存 JSONL trace、逐步 replay。设计说明：[Session REPL](../design/forge/session-repl.zh.md)。

---

## 验证

```bash
pytest tests/python/test_pulsing_forge.py tests/python/test_hybrid_forge_callable.py -q
```

示例：`examples/python/forge_minimal.py`

---

## 下一步

- [核心概念](concepts.zh.md)
- [工具清单（32）](tools.zh.md)
- [Pulsing 部署](deployment.zh.md)
- [Craft 集成](../design/forge/craft-architecture.zh.md)
