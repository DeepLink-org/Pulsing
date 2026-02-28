# Contributing to Pulsing

感谢你对 Pulsing 的贡献兴趣！本文档介绍如何搭建开发环境、运行测试以及提交代码。

## 前置要求

| 工具 | 版本 | 安装方式 |
|------|------|----------|
| Rust | ≥ 1.75 | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| Python | ≥ 3.10 | [python.org](https://python.org) 或 `uv python install 3.11` |
| uv | 最新 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| just | 最新 | `cargo install just` 或 `brew install just` |

## 快速开始（三步）

```bash
# 1. 克隆仓库
git clone https://github.com/DeepLink-org/Pulsing.git && cd Pulsing

# 2. 创建并激活 Python 虚拟环境，安装开发依赖
uv sync --extra dev

# 3. 编译 Rust 核心并安装到当前环境
uv run maturin develop
```

完成后可以验证安装：

```bash
uv run python -c "import pulsing; print(pulsing.__version__)"
```

## 常用开发命令

项目使用 [just](https://github.com/casey/just) 作为任务运行器，所有常用命令都在 `Justfile` 中定义。

```bash
just dev          # 编译并安装（开发模式，等同于 maturin develop）
just test         # 运行全部测试（Rust + Python）
just test-python  # 仅运行 Python 测试
just test-rust    # 仅运行 Rust 测试
just fmt          # 格式化代码（Rust + Python）
just lint         # 代码检查
just check        # 提交前完整检查（格式 + lint + 测试）
just cov          # 生成覆盖率报告
just clean        # 清理构建产物
```

> **提示**：提交代码前请运行 `just check`，确保所有检查通过。

## 项目结构

```
Pulsing/
├── crates/
│   ├── pulsing-actor/    # Rust 核心：Actor、Cluster、Transport
│   └── pulsing-py/       # PyO3 绑定：将 Rust 类型暴露给 Python
├── python/pulsing/       # Python 包
│   ├── core/             # @remote 装饰器、ActorProxy
│   ├── serving/          # LLM 服务路由、调度
│   ├── streaming/        # 分布式队列与发布/订阅
│   ├── agent/            # Agent 运行时工具
│   └── integrations/     # Ray / AutoGen / LangGraph 集成
├── tests/python/         # Python 测试
├── examples/             # 示例代码
└── docs/                 # 文档（MkDocs）
```

## 开发工作流

### 修改 Python 代码

Python 代码无需重新编译，修改后直接运行测试：

```bash
just test-python
# 或者运行单个文件
uv run pytest tests/python/test_remote_decorator.py -v
```

### 修改 Rust 代码

修改 Rust 代码后需要重新编译：

```bash
just dev
just test
```

### 添加新特性

1. 在 `crates/pulsing-actor/` 实现 Rust 逻辑
2. 在 `crates/pulsing-py/src/` 添加 PyO3 绑定
3. 在 `python/pulsing/` 添加 Python 封装（如需要）
4. 在 `tests/python/` 添加测试
5. 运行 `just check` 确认无误

## 代码规范

### Rust

- 使用 `cargo fmt` 格式化（`just fmt` 会自动运行）
- 通过 `cargo clippy` 检查（`just lint` 会自动运行）
- 公共 API 必须有文档注释（`///`）
- 错误类型使用 `thiserror` 定义，避免使用 `anyhow` 做为库的公共 API

### Python

- 使用 `ruff format` 格式化（行宽 88）
- 使用 `ruff check` 检查（遵循 E/F/W/I/UP/B 规则集）
- 类型注解尽量完整
- 异步函数优先使用 `async def`

### 测试

- Python 测试使用 `pytest-asyncio`，配置 `asyncio_mode = "auto"`
- 测试函数命名：`test_<功能描述>_<场景>`
- 避免测试间共享全局状态（每个测试通过 fixture 独立初始化系统）

## 运行文档

```bash
cd docs
uv run mkdocs serve
# 访问 http://localhost:8000
```

## 提交 PR

1. Fork 仓库并创建特性分支：`git checkout -b feat/your-feature`
2. 编写代码和测试
3. 运行 `just check` 确保全部通过
4. Push 并在 GitHub 上创建 Pull Request
5. PR 描述中说明改动目的和测试方式

## 常见问题

**Q: `maturin develop` 报错 `linker 'cc' not found`**

Linux 上需要安装 gcc：

```bash
# Ubuntu/Debian
sudo apt install build-essential
# CentOS/Fedora
sudo dnf install gcc gcc-c++
```

**Q: 运行测试时报 `ImportError: cannot import name '_core' from 'pulsing'`**

需要先编译 Rust 核心：

```bash
just dev
```

**Q: macOS 上 `maturin develop` 很慢**

可以尝试只编译当前架构：

```bash
maturin develop --target $(rustc -vV | grep host | cut -d' ' -f2)
```

**Q: 如何只跑某一个测试？**

```bash
uv run pytest tests/python/test_remote_decorator.py::test_spawn_actor -v
```
