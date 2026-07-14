# Forge 测试体系

> **读者**：Forge 贡献者、CI 维护者
> **关联**：[engineering.md](./engineering.md) · [session-repl.md](./session-repl.md)

Forge 测试按 **L0 → L4** 分层，借鉴了业界 agent-tool 生态的 fixture 思路（尤其是 `apply_patch` 场景目录），但断言的是 **Pulsing Forge 自身行为**，不对外做产品级「迁移对照」宣传。

---

## 分层（Test Pyramid）

| 层 | 含义 | 位置 |
|----|------|------|
| **L0** | 32 工具已注册、Host/隔离分区正确 | `tests/python/forge/test_gates.py` |
| **L1** | 每个工具默认可调用（无 `Unknown tool`） | 同上 + `test_hybrid_forge_callable.py` |
| **L2** | Wire 形状与关键副作用（Read 内容、patch 落盘、session plan） | `tests/python/forge/test_wire.py` |
| **L3** | 场景 fixture、JSONL trace replay | `tests/python/forge/test_integration.py`、Rust patch scenarios |
| **L4** | Pulsing Actor（inbox、MCP hub、worker） | `test_forge_p0_actors.py` 等 |

共享基础设施：`python/pulsing/testing/forge_harness.py`（minimal args、smoke runner、wire check 注册表）。

内部 conformance 分数：`python/pulsing/forge/codex_parity.py`（CI 用，文档站不链出）。

---

## 怎么跑

```bash
# Forge 全部分层（推荐 CI）
pytest tests/python/forge/ -m forge -q

# 仅注册 + callable
pytest tests/python/forge/ -m "forge_l0 or forge_l1" -q

# Rust：apply_patch 场景（复用 vendor codex-rs fixtures）
cargo test -p pulsing-forge apply_patch_scenarios

# Rust crate 单元测试
cargo test -p pulsing-forge

# 与 Craft 交叉
pytest tests/python/craft/test_forge_events.py tests/python/craft/test_tools.py -q
```

需要 Hybrid 路径时先 `maturin develop`。

---

## apply_patch 场景 fixture

Rust 集成测试读取：

```text
vendor/codex-rs/apply-patch/tests/fixtures/scenarios/
  001_add_file/
    input/
    expected/
    patch.txt
```

每个场景：复制 `input/` → 在临时目录执行 patch → 对比 `expected/` 文件树快照。布局与 codex-apply-patch 一致，便于跨语言复用。

实现：`crates/pulsing-forge/tests/apply_patch_scenarios.rs`

---

## 扩展 L2 wire check

在 `forge_harness.py` 中：

```python
from pulsing.testing.forge_harness import register_wire_check

def _check_my_tool(rt, tmp_path, out):
    assert not out.is_error
    ...

register_wire_check("my_tool", _check_my_tool)
```

`test_wire.py` 会自动 parametrized 跑所有已注册工具。

---

## Trace replay（L3 / L4）

JSONL 格式见 [session-repl.md](./session-repl.md)。Fixture：`tests/fixtures/forge_traces/`。

`replay --verify` 语义：重跑 tool call 并对比 recorded result — 用于 Agent 失败轨迹的最小复现。

---

## 新增工具时的清单

1. 加入 `integrated.py` 的 `FORGE_*_TOOL_NAMES`
2. 在 `forge_harness.minimal_tool_args` 补最小参数
3. L0 registry 自动覆盖；跑 L1 smoke
4. 若有 structured 输出，注册 L2 wire check
5. 复杂行为补 domain 测试（`test_forge_mcp.py` 等）或场景 fixture

---

## 与 Codex 参考实现的关系

- **可以**复用其 **fixture 布局**与 **场景数据**（如 apply_patch）
- **不必**在公开文档中强调「parity 等级」；`codex_parity.py` 仅供贡献者与 CI 追踪缺口
- 行为分歧时以 Forge 设计文档 §设计取舍 为准，测试应断言 Forge 承诺的行为
