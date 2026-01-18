# 术语与风格约定

本页定义 Pulsing 文档和代码中的术语与风格规范，确保一致性。

## 核心术语

| 术语 | 用法 | 说明 |
|------|------|------|
| `ActorSystem` | 代码符号 | Rust/Python 类名，用于代码引用 |
| actor system | 概念描述 | 一般性描述时使用小写 |
| `Actor` | 代码符号 | 基类名 |
| actor | 概念描述 | 一般性描述 |
| `ActorRef` | 代码符号 | 底层 actor 引用 |
| `ActorProxy` | 代码符号 | `@remote` 装饰器返回的高级代理 |

## 组件命名

| 组件 | CLI actor 类路径 | 说明 |
|------|------------------|------|
| Router | `pulsing.actors.Router` | OpenAI 兼容 HTTP 路由 |
| TransformersWorker | `pulsing.actors.TransformersWorker` | Transformers 推理 Worker |
| VllmWorker | `pulsing.actors.VllmWorker` | vLLM 推理 Worker |

**注意**：文档中提到"Router"时，通常指 LLM 推理服务的 HTTP 路由组件。示例代码中若需要任务分发逻辑，应使用 `Dispatcher` 等名称以避免混淆。

## CLI 命令格式

### 启动 Actor

```bash
pulsing actor <完整类路径> [选项]

# 示例
pulsing actor pulsing.actors.Router --http_port 8080 --model_name my-llm
pulsing actor pulsing.actors.TransformersWorker --model_name gpt2 --device cpu
```

### 检查命令（观察者模式）

`pulsing inspect` 使用子命令结构：

```bash
# 集群状态
pulsing inspect cluster --seeds <地址>

# Actor 分布
pulsing inspect actors --seeds <地址> [--top N] [--filter ...]
pulsing inspect actors --endpoint <地址> [--detailed]

# 指标
pulsing inspect metrics --seeds <地址> [--raw]

# 实时监视
pulsing inspect watch --seeds <地址> [--interval ...] [--kind ...]
```

**已移除**：`pulsing actor list` → 请使用 `pulsing inspect actors`

## 代码风格

- **Python**：遵循 PEP 8，优先使用类型注解
- **Rust**：使用 `cargo fmt` 和 `cargo clippy`
- **文档**：中英双语（`.zh.md` 为中文版）

## 文档引用

- 代码符号使用反引号：`` `ActorSystem` ``
- 命令使用代码块
- 文件路径使用反引号：`` `python/pulsing/actor/` ``
