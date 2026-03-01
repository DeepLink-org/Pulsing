# 可靠性实践（超时 / 重试 / 重启）

本页汇总在 Pulsing 上做生产化时，最容易踩坑、也最值得统一口径的可靠性实践。

## TL;DR

- 网络与节点都会失败：**显式加超时**，并按失败场景设计。
- Pulsing 不提供端到端 exactly-once：**业务必须幂等**。
- Pulsing 支持 **actor 级别重启**（不引入 supervision tree）：用它做“崩溃恢复”，不要把它当“正确性保证”。

## 超时

对 `ask` 建议显式加超时：

```python
result = await asyncio.wait_for(ref.ask({"op": "compute"}), timeout=10.0)
```

对 proxy 方法调用：`await asyncio.wait_for(proxy.compute(), timeout=10.0)`。

## 重试（放在业务层）

Pulsing 不会替你“隐式重试”。一旦你做重试，就要默认可能出现重复处理。

推荐模式：

- 每个请求都带 **幂等键（idempotency key）**
- actor 内部（或外部存储）做 **去重（dedup）**

## actor 级别重启（supervision）

你可以在 Python 的 `@pul.remote` 上配置重启策略：

```python
import pulsing as pul

@pul.remote(restart_policy="on_failure", max_restarts=5, min_backoff=0.2, max_backoff=10.0)
class Worker:
    def work(self, x: int) -> int:
        return 100 // x
```

### 它是什么 / 不是什么

- **是**：actor 实例崩溃后的自动恢复（带退避与重启上限）
- **不是**：supervision tree，也**不是** exactly-once 保证

## 错误处理

Pulsing 区分框架错误和 Actor 执行错误，支持适当的恢复策略。

### 错误分类

- **框架错误** (`PulsingRuntimeError`): 网络故障、集群问题、配置错误、Actor 系统错误
- **Actor 错误** (`PulsingActorError`): 用户代码错误
  - **业务错误** (`PulsingBusinessError`): 用户输入验证失败（可恢复，返回给调用者）
  - **系统错误** (`PulsingSystemError`): 内部处理失败（可能触发 Actor 重启）
  - **超时错误** (`PulsingTimeoutError`): 操作超时（可重试）

### 错误恢复策略

1. **业务错误**: 返回给调用者，不重试
   ```python
   except PulsingBusinessError as e:
       # 用户输入问题 - 返回错误给调用者
       return {"error": e.message, "code": e.code}
   ```

2. **系统错误**: 检查 `recoverable` 标志，可能触发 Actor 重启
   ```python
   except PulsingSystemError as e:
       if e.recoverable:
           # 可以重试或等待 Actor 重启
           # 如果配置了 restart_policy，Actor 会重启
           pass
       else:
           # 不可恢复 - 记录日志并失败
           logger.error(f"不可恢复错误: {e.error}")
   ```

3. **超时错误**: 使用退避策略重试
   ```python
   except PulsingTimeoutError as e:
       # 使用指数退避重试
       await asyncio.sleep(backoff_seconds)
       return await retry_operation()
   ```

4. **框架错误**: 在应用层记录日志并处理
   ```python
   except PulsingRuntimeError as e:
       # 网络/集群问题 - 记录日志并在应用层处理
       logger.error(f"框架错误: {e}")
       # 可能需要重试或故障转移
   ```

### 示例：综合错误处理

```python
from pulsing.exceptions import (
    PulsingBusinessError,
    PulsingSystemError,
    PulsingTimeoutError,
    PulsingRuntimeError,
)

async def process_with_retry(actor, data, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await actor.process(data)
        except PulsingBusinessError as e:
            # 不重试业务错误
            raise
        except PulsingSystemError as e:
            if not e.recoverable:
                raise
            # 等待 Actor 重启，然后重试
            await asyncio.sleep(2 ** attempt)
        except PulsingTimeoutError:
            # 重试超时错误
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise
        except PulsingRuntimeError as e:
            # 框架错误 - 可能需要故障转移
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise
```

## 流式响应的韧性

对流式响应要默认可能“部分输出后中断”。建议每个 chunk 自包含：

- 每个 chunk 带 `seq` / offset / id
- 客户端可恢复或去重
