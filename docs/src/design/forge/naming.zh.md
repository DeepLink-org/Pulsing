# Pulsing Forge — 命名定案

> **对外品牌**：**Pulsing Forge**
>
> **代码包名**：Rust `pulsing-forge`，Python `pulsing.forge`

---

## 1. 定案

| 层级 | 名称 |
|------|------|
| **产品 / 文档** | **Pulsing Forge** |
| **Python import** | `pulsing.forge` |
| **Rust crate** | `pulsing-forge`（lib：`pulsing_forge`） |
| **pip extra** | `pulsing[forge]`（计划；当前可用 `pip install pulsing`） |
| **兼容** | `pulsing.tools` → 已废弃，re-export `pulsing.forge` |

---

## 2. 路径对照（rename 后）

| 旧路径 | 新路径 |
|--------|--------|
| `crates/pulsing-tools/` | `crates/pulsing-forge/` |
| `python/pulsing/tools/` | `python/pulsing/forge/` |
| `docs/design/pulsing-tools.md` | `do../design/forge/engineering.md` |
| `scripts/sync-codex-tools.sh` | `scripts/sync-codex-forge.sh` |
| `tests/python/test_pulsing_tools.py` | `tests/python/test_pulsing_forge.py` |
| `examples/python/tools_minimal.py` | `examples/python/forge_minimal.py` |

---

## 3. 生态名称

```text
Pulsing        → pulsing           Actor 运行时
Pulsing Forge  → pulsing.forge     Tool + Sandbox
Craft          → pulsing.craft     Multi-Agent 应用
```

---

## 4. 文档

- [../../forge/index.md](../../forge/index.md) — 产品介绍
- [do../../forge/abstractions.md](../../forge/abstractions.md) — 抽象模型

---

## 5. Checklist

- [x] 品牌：**Pulsing Forge**
- [x] Rust crate / Python 包路径 rename
- [x] `pulsing.tools` 兼容 shim（DeprecationWarning）
- [x] 主仓 README.zh 增加 Pulsing Forge 小节
- [x] `pyproject.toml` 增加 `pulsing[forge]` extra
- [x] 产品文档 `docs/forge/`
