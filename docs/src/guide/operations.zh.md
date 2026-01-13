# CLI 运维操作

Pulsing 内置 CLI 工具，用于运行、检查和基准测试分布式系统。

---

## 运行服务

### Router（OpenAI 兼容 HTTP API）

```bash
pulsing actor router --addr 0.0.0.0:8000 --http_port 8080 --model_name my-llm
```

### Transformers Worker

```bash
pulsing actor transformers --model gpt2 --addr 0.0.0.0:8001 --seeds 127.0.0.1:8000
```

### vLLM Worker

```bash
pulsing actor vllm --model Qwen/Qwen2 --addr 0.0.0.0:8002 --seeds 127.0.0.1:8000
```

---

## Actor List

`pulsing actor list` 是轻量级 **观察者**，通过 HTTP 查询 actor — **无需加入集群**。

### 单节点

```bash
pulsing actor list --endpoint 127.0.0.1:8000
```

### 集群（通过 Seeds）

```bash
pulsing actor list --seeds 127.0.0.1:8000,127.0.0.1:8001
```

### 选项

| 参数 | 描述 |
|------|------|
| `--all_actors True` | 包含内部/系统 actor |
| `--json True` | JSON 格式输出 |

!!! note
    使用 HTTP/2 (h2c)。节点需暴露 HTTP 端点。

---

## Inspect

`pulsing inspect` 加入集群（通过 seeds）并打印成员和 actor 的可读快照。

```bash
pulsing inspect --seeds 127.0.0.1:8000
```

输出包含：

- **集群成员**：节点 id、地址、状态
- **命名 Actor**：跨节点分布

!!! tip
    本地 seeds（`127.0.0.1`）时，CLI 自动绑定到 `127.0.0.1:0`。

---

## Bench

`pulsing bench` 对 OpenAI 兼容推理端点进行负载测试。

```bash
pulsing bench gpt2 --url http://localhost:8080
```

!!! note "可选扩展"
    若提示 `pulsing._bench module not found`：

    ```bash
    maturin develop --manifest-path crates/pulsing-bench-py/Cargo.toml
    ```

---

## 快速参考

| 任务 | 命令 |
|------|------|
| 启动 router | `pulsing actor router --addr 0.0.0.0:8000 --http_port 8080` |
| 启动 worker | `pulsing actor transformers --model gpt2 --seeds ...` |
| 列出 actor | `pulsing actor list --endpoint 127.0.0.1:8000` |
| 检查集群 | `pulsing inspect --seeds 127.0.0.1:8000` |
| 基准测试 | `pulsing bench gpt2 --url http://localhost:8080` |

---

## 下一步

- [LLM 推理](../examples/llm_inference.zh.md) — 可运行的端到端教程
- [安全](security.zh.md) — mTLS 和集群隔离
