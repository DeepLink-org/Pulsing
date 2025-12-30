# 安装指南

本指南介绍如何安装 Pulsing 并开始开发。

## 前置条件

安装 Pulsing 之前，请确保您有：

- **Python 3.10+** - Pulsing 需要 Python 3.10 或更高版本
- **Rust 工具链** - 用于构建原生扩展
- **Linux/macOS** - 当前支持的平台

## 安装方法

### 方法 1：从源码安装（开发）

用于开发或获取最新功能：

```bash
# 克隆仓库
git clone https://github.com/reiase/pulsing.git
cd pulsing

# 如果尚未安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装 maturin 用于构建 Python 扩展
pip install maturin

# 以开发模式构建和安装
maturin develop
```

### 方法 2：从 PyPI 安装（即将推出）

```bash
pip install pulsing
```

## 验证安装

安装后，验证一切正常：

```python
import asyncio
from pulsing.actor import as_actor, create_actor_system, SystemConfig

@as_actor
class HelloActor:
    def greet(self, name: str) -> str:
        return f"Hello, {name}!"

async def main():
    # 创建 Actor 系统
    system = await create_actor_system(SystemConfig.standalone())

    # 创建 Actor
    hello = await HelloActor.local(system)

    # 调用方法
    result = await hello.greet("World")
    print(result)  # Hello, World!

    # 清理
    await system.shutdown()

asyncio.run(main())
```

如果您看到打印出 "Hello, World!"，则安装成功！

## 开发环境设置

如需为 Pulsing 做贡献：

### 1. 克隆和设置

```bash
git clone https://github.com/reiase/pulsing.git
cd pulsing

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# 安装开发依赖
pip install maturin pytest pytest-asyncio
```

### 2. 构建

```bash
# 开发构建（带调试符号）
maturin develop

# 发布构建（优化）
maturin develop --release
```

### 3. 运行测试

```bash
# 运行 Python 测试
pytest tests/

# 运行 Rust 测试
cargo test --workspace
```

## 平台特定说明

### Linux

无特殊要求。完全支持所有功能。

### macOS

- Apple Silicon (M1/M2)：完全支持
- Intel Mac：完全支持

### Windows

Windows 支持是实验性的。为获得最佳效果，请使用带有 Ubuntu 的 WSL2。

## 故障排除

### 构建错误

**问题：** `cargo build` 因缺少依赖而失败

**解决方案：**

```bash
# Ubuntu/Debian
sudo apt-get install build-essential pkg-config libssl-dev

# macOS
xcode-select --install
```

**问题：** `maturin develop` 失败

**解决方案：**

```bash
# 确保您有最新的 maturin
pip install --upgrade maturin

# 尝试使用详细输出
maturin develop -v
```

### 运行时错误

**问题：** 安装后导入错误

**解决方案：**

```bash
# 重新构建扩展
maturin develop --release
```

**问题：** 端口已被使用

**解决方案：**

```python
# 使用不同的端口
config = SystemConfig.with_addr("0.0.0.0:8001")  # 更改端口
```

## 下一步

现在您已经安装了 Pulsing：

1. **快速开始** - 按照[快速开始指南](quickstart/index.md)操作
2. **Actor 指南** - 了解 [Actors](guide/actors.md)
3. **示例** - 探索[示例代码](examples/index.zh.md)
4. **API 参考** - 查看 [API 文档](api_reference.md)
