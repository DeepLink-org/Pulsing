# Contributing to Pulsing

感谢你对 Pulsing 的兴趣！我们欢迎各种形式的贡献。

## 开发环境设置

### 前置要求

- Rust 1.75+
- Python 3.10+
- maturin (`pip install maturin`)

### 构建项目

```bash
# 克隆仓库
git clone https://github.com/DeepLink-org/Pulsing.git
cd pulsing

# 安装 Python 依赖
pip install -e .
# 或使用 maturin
maturin develop

# 运行测试
cargo test
pytest tests/
```

## 贡献流程

### 1. 创建 Issue

在开始工作之前，请先创建一个 Issue 讨论你想要做的改动。这有助于避免重复工作并确保你的贡献与项目方向一致。

### 2. Fork 和 Clone

```bash
git clone https://github.com/YOUR_USERNAME/pulsing.git
cd pulsing
git remote add upstream https://github.com/DeepLink-org/Pulsing.git
```

### 3. 创建分支

```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/your-fix-name
```

### 4. 开发

- 遵循现有的代码风格
- 添加必要的测试
- 更新相关文档

### 5. 提交

我们使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
feat: 添加新功能
fix: 修复 bug
docs: 更新文档
test: 添加或修改测试
refactor: 代码重构
chore: 构建过程或辅助工具的变动
```

示例：
```bash
git commit -m "feat: add streaming support to ActorRef"
git commit -m "fix: resolve memory leak in mailbox"
git commit -m "docs: update README with new examples"
```

### 6. 提交 PR

- 确保所有测试通过
- 确保代码格式正确 (`cargo fmt`, `ruff format`)
- 提供清晰的 PR 描述

## 代码风格

### Rust

- 使用 `cargo fmt` 格式化代码
- 使用 `cargo clippy` 检查代码质量
- 遵循 [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)

### Python

- 使用 `ruff` 进行格式化和检查
- 遵循 PEP 8
- 使用类型注解

## 测试

### Rust 测试

```bash
# 运行所有测试
cargo test

# 运行特定测试
cargo test test_name

# 运行 Actor System 测试
cargo test -p pulsing-actor
```

### Python 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/actor_system/
```

## 文档

- API 文档使用 rustdoc / docstring
- 设计文档放在 `docs/design/`
- 示例代码放在 `examples/`

## 行为准则

请阅读并遵守我们的 [行为准则](CODE_OF_CONDUCT.md)。

## 许可证

通过贡献代码，你同意你的贡献将在 Apache-2.0 许可证下发布。

## 问题？

如果你有任何问题，请通过 [GitHub Issues](https://github.com/DeepLink-org/Pulsing/issues) 联系我们。
