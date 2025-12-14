# Pulsing

Pulsing 是一个负载感知、KV 缓存感知的 LLM 推理服务系统，专注于多租户和高并发场景。通过动态感知请求成本、内存布局和缓存命中率，Pulsing 能够提升整体吞吐量并降低尾延迟，同时支持 vLLM、SGLang 等主流推理后端。

## 项目定位

Pulsing 的目标是提供一个**轻量、易用、可扩展**的 LLM 推理服务框架：

- **统一的命令行入口**：通过 `pulsing` 命令一键启动 frontend、backend、router 等组件
- **灵活的参数配置**：基于 [hyperparameter](https://github.com/reiase/hyperparameter) 的声明式参数系统
- **可插拔的后端**：支持 vLLM、Transformers 等多种推理后端
- **智能路由**：内置 KV 缓存感知路由，优化请求分发

## 快速上手

### 安装

```bash
pip install pulsing
```

### 启动服务

**1. 启动 vLLM 后端**

```bash
pulsing vllm --model Qwen/Qwen2.5-0.5B
```

**2. 启动 Frontend（HTTP 服务）**

```bash
pulsing frontend --model_name Qwen2.5-0.5B
```

**3. 启动独立 Router（可选，用于分离式部署）**

```bash
pulsing router --endpoint dynamo.prefill.generate --block_size 64
```

### 高级配置

Pulsing 使用 hyperparameter 进行参数管理，支持通过 `-D` 传递高级参数：

```bash
# 配置 runtime
pulsing frontend --model_name MyModel \
    -D runtime.request_plane=nats \
    -D runtime.store_kv=etcd

# 配置 KV Router 高级选项
pulsing router --endpoint dynamo.prefill.generate \
    -D router.kv.reset_states=true \
    -D router.kv.track_active_blocks=false
```

### 可用命令

| 命令 | 说明 |
|------|------|
| `pulsing frontend` | 启动 HTTP/gRPC 前端服务 |
| `pulsing vllm` | 启动 vLLM 后端 worker |
| `pulsing transformers` | 启动 Transformers 后端 worker |
| `pulsing router` | 启动独立 KV 感知路由器 |
| `pulsing bench` | 运行性能基准测试 |

使用 `pulsing <command> --help` 查看各命令的详细参数。

## 与 NVIDIA Dynamo 的关系

Pulsing 是 [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) 项目的独立维护分支（当前基线版本：v0.7.0）。

**继承**：
- Dynamo 的解耦推理架构
- Rust + Python 协同设计
- 分布式 runtime 和服务发现机制

**演进**：
- 重新设计的命令行接口（`pulsing` CLI）
- 基于 hyperparameter 的声明式参数系统
- 更清晰的模块划分和代码组织

**致谢**：感谢 ai-dynamo/dynamo 项目所有贡献者的高质量开源工作。本项目遵循 Apache-2.0 协议，在所有衍生分发中保留原始版权和署名声明。

## 构建项目

### 从源码构建

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 构建 wheel
cd lib/bindings/python
uv build --wheel --python 3.12
```

### ABI 兼容构建

构建可在不同 Linux 发行版上运行的 manylinux 兼容 wheel：

```bash
# 安装构建工具
pip install maturin ziglang

# 构建
cd lib/bindings/python
maturin pep517 build-wheel \
    --auditwheel repair --manylinux \
    --zig --compatibility manylinux_2_24
```

## 许可证

Apache-2.0
