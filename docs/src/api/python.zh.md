# Python API 参考

此页面包含 Pulsing Python 接口的完整自动生成 API 文档。

## 安装

Pulsing 需要 Python 3.10+，可以通过 pip 安装：

```bash
pip install pulsing
```

开发时，克隆仓库并以开发模式安装：

```bash
git clone https://github.com/DeepLink-org/pulsing
cd pulsing
pip install -e .
```

## 核心模块

::: pulsing

## Actor 模块

::: pulsing.core

## Agent 模块

::: pulsing.agent

## 队列模块

::: pulsing.streaming

## Subprocess 模块

`pulsing.subprocess` 提供与标准库 `subprocess` 兼容的同步 API。
不传 `resources` 时会回退到 Python 原生 `subprocess`；
传入 `resources` 且设置 `USE_POLSING_SUBPROCESS=1` 时，
命令会通过 Pulsing 后端执行。

::: pulsing.subprocess
