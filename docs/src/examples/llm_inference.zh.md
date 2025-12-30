# LLM 推理（概览）

Pulsing 正在变成一个**通用的分布式 Actor 框架**，同时也很适合用于 **LLM 推理服务**，尤其是需要：

- router + worker 架构
- 分布式调度 / 负载感知
- 流式响应（`ask_stream`）

本页目前是概览（Draft）。相关设计可先看：

- `docs/src/design/http2-transport.md`：HTTP/2 流式协议设计
- `docs/src/design/load_sync.md`：负载同步机制

## 推荐架构

- **Router**：接入请求，选择 worker，转发请求
- **Worker**：承载模型副本，对外提供 `generate` / `generate_stream`

## 下一步

如果你希望把这里做成可运行示例，请告诉我你希望使用哪种后端：

- `transformers` + `torch`
- `vllm`
- `triton` / 自研引擎
