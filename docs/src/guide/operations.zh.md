# 运维（CLI）

本页提供使用 Pulsing CLI 进行运行、巡检与排障的最小入口。

## 你能做什么

- **启动服务**：Router / 推理 Worker
- **巡检集群**：查看节点 + 命名 actors 分布
- **列出 actors**：通过 HTTP 观察者模式查询（无需加入集群）
- **压测**：对 OpenAI 兼容端点做基准测试

## 常用命令

## 快速入口

- [Actor 列表](actor_list.zh.md)
- [巡检](inspect.zh.md)
- [压测](bench.zh.md)

### 启动服务（router / workers）

- Router（OpenAI 兼容 HTTP API）：

```bash
pulsing actor router --addr 0.0.0.0:8000 --http_port 8080 --model_name my-llm
```

- Transformers Worker：

```bash
pulsing actor transformers --model gpt2 --addr 0.0.0.0:8001 --seeds 127.0.0.1:8000
```

- vLLM Worker：

```bash
pulsing actor vllm --model Qwen/Qwen2 --addr 0.0.0.0:8002 --seeds 127.0.0.1:8000
```

### 巡检集群

```bash
pulsing inspect --seeds 127.0.0.1:8000
```

### 列出 actors（观察者模式）

```bash
# 单节点
pulsing actor list --endpoint 127.0.0.1:8000

# 集群（通过 seeds）
pulsing actor list --seeds 127.0.0.1:8000,127.0.0.1:8001
```

### 压测端点

```bash
pulsing bench gpt2 --url http://localhost:8080
```

## 下一步

- 想按步骤跑通完整链路：见 [LLM 推理](../examples/llm_inference.zh.md)。

