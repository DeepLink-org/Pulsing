# 可靠性实践（超时 / 重试 / 重启）

本页汇总在 Pulsing 上做生产化时，最容易踩坑、也最值得统一口径的可靠性实践。

## TL;DR

- 网络与节点都会失败：**显式加超时**，并按失败场景设计。
- Pulsing 不提供端到端 exactly-once：**业务必须幂等**。
- Pulsing 支持 **actor 级别重启**（不引入 supervision tree）：用它做“崩溃恢复”，不要把它当“正确性保证”。

## 超时

对 `ask` 建议显式加超时：

```python
from pulsing.actor import ask_with_timeout

result = await ask_with_timeout(ref, {"op": "compute"}, timeout=10.0)
```

## 重试（放在业务层）

Pulsing 不会替你“隐式重试”。一旦你做重试，就要默认可能出现重复处理。

推荐模式：

- 每个请求都带 **幂等键（idempotency key）**
- actor 内部（或外部存储）做 **去重（dedup）**

## actor 级别重启（supervision）

你可以在 Python 的 `@remote` 上配置重启策略：

```python
from pulsing.actor import remote

@remote(restart_policy="on-failure", max_restarts=5, min_backoff=0.2, max_backoff=10.0)
class Worker:
    def work(self, x: int) -> int:
        return 100 // x
```

### 它是什么 / 不是什么

- **是**：actor 实例崩溃后的自动恢复（带退避与重启上限）
- **不是**：supervision tree，也**不是** exactly-once 保证

## 流式响应的韧性

对流式响应要默认可能“部分输出后中断”。建议每个 chunk 自包含：

- 每个 chunk 带 `seq` / offset / id
- 客户端可恢复或去重

