# 通信范式指南

本指南解释 Pulsing 中不同通信范式的**设计原理**和**使用场景**，帮助您理解"为什么"需要这些范式，以及"何时"使用它们。

## 为什么需要不同的通信范式？

### Actor 的核心特性

在 Actor 模型中，每个 Actor **一次只处理一条消息**。这是 Actor 模型的基础保证，确保状态更新的安全性。

```
Actor 邮箱（FIFO 队列）
    ↓
[消息1] → Actor 处理 → 响应1
[消息2] → Actor 处理 → 响应2  ← 必须等待消息1完成
[消息3] → Actor 处理 → 响应3  ← 必须等待消息2完成
```

### 问题：阻塞 vs 非阻塞

如果 Actor 在处理一条消息时被阻塞（例如等待网络响应），那么：

```
❌ 同步阻塞模式：
消息1: [等待HTTP响应...████████] 500ms  ← 阻塞中
消息2: [等待中...]                      ← 无法处理！
消息3: [等待中...]                      ← 无法处理！
```

**结果**：Actor 无法处理其他消息，吞吐量极低。

**解决方案**：使用异步非阻塞模式：

```
✅ 异步非阻塞模式：
消息1: [等待HTTP...] 500ms  ← 在后台等待
消息2: [处理中...] 10ms     ← 可以同时处理！
消息3: [处理中...] 10ms     ← 可以同时处理！
```

**结果**：Actor 可以并发处理多个请求，吞吐量大幅提升。

### 为什么需要流式响应？

对于需要长时间生成结果的操作（如 LLM token 生成），如果等待全部完成：

```
❌ 等待全部完成：
用户: [等待...████████████████] 10秒后看到结果
```

**问题**：用户体验差，需要等待很久。

**解决方案**：流式传输，边生成边返回：

```
✅ 流式传输：
用户: [token1][token2][token3]...  ← 立即看到结果
```

**结果**：用户立即看到进度，体验更好。

---

## 四种通信范式

基于上述原理，Pulsing 提供了四种通信范式：

| 范式 | 方法类型 | 为什么需要 | 使用场景 |
|------|----------|------------|----------|
| **同步** | `def method()` | 快速操作不需要并发，简单直接 | 快速 CPU 工作、状态变更 |
| **异步** | `async def method()` | 避免阻塞，允许并发处理 | I/O 操作、外部 API 调用 |
| **流式** | `async def method()` 带 `yield` | 增量返回，提升用户体验 | LLM token 生成、大数据传输 |
| **发送即忘** | `tell()` | 不需要响应，最大化吞吐量 | 日志记录、通知 |

## 1. 同步方法 (`def method`)

### 为什么需要同步方法？

**原理**：对于快速操作（< 10ms），并发带来的开销大于收益。

- ✅ **简单直接**：不需要 `async/await`，代码更简洁
- ✅ **无并发开销**：快速操作不需要并发，顺序执行即可
- ✅ **可预测**：严格顺序执行，易于理解和调试

**适用场景**：操作足够快，阻塞时间可以忽略不计。

### 行为特性

- **顺序执行**：Actor 一次处理一个请求
- **阻塞 Actor**：处理时，Actor 无法处理其他消息
- **简单可预测**：无并发问题

### 何时使用

✅ **最适合：**
- 快速 CPU 密集型操作（计算、状态更新）
- 简单状态变更（递增计数器、更新字典）
- 在微秒到毫秒内完成的操作（< 10ms）

❌ **避免用于：**
- 网络请求（HTTP、数据库查询）
- 文件 I/O 操作
- 可能耗时 > 10ms 的任何操作

### 示例

```python
@pul.remote
class Counter:
    def __init__(self):
        self.value = 0
        self.history = []

    # ✅ 好：快速状态变更
    def increment(self, n: int = 1) -> int:
        self.value += n
        self.history.append(self.value)
        return self.value

    # ✅ 好：简单计算
    def get_average(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    # ❌ 差：网络 I/O 会阻塞 Actor
    def fetch_data(self, url: str) -> dict:
        # 这会阻塞 Actor 整个 HTTP 请求期间！
        response = requests.get(url)  # 不要这样做！
        return response.json()
```

### 性能特征

```
请求 1: [████████████] 2ms
请求 2:              [████████████] 2ms
请求 3:                            [████████████] 2ms
总计: 6ms（顺序执行）
```

---

## 2. 异步方法 (`async def method`)

### 为什么需要异步方法？

**核心问题**：如果使用同步方法处理 I/O 操作，Actor 会被阻塞，无法处理其他消息。

**原理**：
- 异步方法在 `await` 时会**让出控制权**
- Actor 可以在等待期间**处理其他消息**
- 多个异步操作可以**并发执行**

**对比**：

```python
# ❌ 同步：阻塞 Actor
def fetch_data(self, url: str) -> dict:
    response = requests.get(url)  # 阻塞 500ms
    return response.json()
# 结果：Actor 在这 500ms 内无法处理任何其他消息

# ✅ 异步：非阻塞
async def fetch_data(self, url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)  # 等待期间可以处理其他消息
        return response.json()
# 结果：Actor 可以在等待 HTTP 响应时处理其他请求
```

### 行为特性

- **非阻塞执行**：Actor 可以在等待时处理其他消息
- **并发处理**：多个异步方法可以同时运行
- **后台任务**：方法作为 Actor 上的后台任务运行

### 何时使用

✅ **最适合：**
- I/O 操作（HTTP 请求、数据库查询、文件 I/O）
- 外部 API 调用
- 可能耗时 > 10ms 的操作
- 需要并发处理多个请求

❌ **避免用于：**
- 快速 CPU 密集型操作（使用同步方法更简单）
- 简单状态变更（同步方法更简单）

### 示例

```python
@pul.remote
class DataService:
    def __init__(self):
        self.cache = {}

    # ✅ 好：网络 I/O - 不阻塞 Actor
    async def fetch_user(self, user_id: str) -> dict:
        # 等待 HTTP 响应时，Actor 可以处理其他请求
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://api.example.com/users/{user_id}")
            return response.json()

    # ✅ 好：数据库查询
    async def get_orders(self, user_id: str) -> list[dict]:
        # 等待数据库时，Actor 可以处理其他请求
        async with database.transaction() as tx:
            return await tx.fetch("SELECT * FROM orders WHERE user_id = $1", user_id)

    # ✅ 好：多个并发操作
    async def fetch_user_profile(self, user_id: str) -> dict:
        # 这些操作并发运行，不是顺序运行
        user, orders, preferences = await asyncio.gather(
            self.fetch_user(user_id),
            self.get_orders(user_id),
            self.get_preferences(user_id),
        )
        return {"user": user, "orders": orders, "preferences": preferences}

    # ❌ 差：快速操作 - 同步更简单
    async def get_cache(self, key: str) -> dict:
        # 这个操作足够快，适合同步方法
        return self.cache.get(key, {})
```

### 性能特征

```
请求 1: [████████████████████] 50ms（等待 HTTP）
请求 2: [████████████████████] 50ms（等待 HTTP）  ← 并发！
请求 3: [████████████████████] 50ms（等待 HTTP）  ← 并发！
总计: ~50ms（并发，不是 150ms！）
```

### 使用模式

#### 模式 1：等待最终结果

```python
service = await DataService.spawn()

# 等待最终结果
result = await service.fetch_user("user123")
print(result)
```

#### 模式 2：发送即忘（后台任务）

```python
# 启动异步操作，不等待
task = asyncio.create_task(service.fetch_user("user123"))

# 做其他工作...
await other_operations()

# 稍后获取结果
result = await task
```

---

## 3. 流式响应 (`async def method` 带 `yield`)

### 为什么需要流式响应？

**核心问题**：某些操作需要很长时间才能完成（如 LLM 生成 1000 个 token），如果等待全部完成再返回：

```
❌ 等待全部完成：
用户请求 → [生成中...████████] 10秒 → 返回全部结果
问题：用户需要等待 10 秒才能看到任何内容
```

**原理**：
- 使用 `yield` **增量返回**结果
- 客户端可以**立即开始处理**第一个结果
- 提升用户体验，减少感知延迟

```
✅ 流式返回：
用户请求 → [token1] → [token2] → [token3]... → 完成
结果：用户立即看到第一个 token，无需等待
```

**额外好处**：
- 可以**提前取消**（如果用户不需要了）
- 可以显示**进度更新**
- 可以处理**大数据集**（不需要全部加载到内存）

### 行为特性

- **增量交付**：结果在可用时立即发送
- **非阻塞**：Actor 可以在生成流时处理其他消息
- **背压**：通过有界通道自然流控
- **可取消**：客户端可以取消流消费

### 何时使用

✅ **最适合：**
- LLM token 生成（用户希望立即看到输出）
- 大数据传输（分块处理，避免内存溢出）
- 实时数据流（传感器数据、日志）
- 进度更新（长时间任务需要反馈）

❌ **避免用于：**
- 小的完整响应（使用常规异步方法）
- 需要原子结果时（全有或全无）

### 示例

```python
@pul.remote
class LLMService:
    # ✅ 好：流式 LLM token
    async def generate(self, prompt: str):
        # 在生成时流式传输 token
        async for token in self.llm_client.stream(prompt):
            yield {"token": token, "type": "token"}

        # 最终结果
        yield {"type": "done", "total_tokens": count}

    # ✅ 好：大文件处理
    async def process_large_file(self, file_path: str):
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                processed = process_line(line)
                yield {"line": i, "data": processed}

                # 允许处理其他消息
                await asyncio.sleep(0)  # 让出控制权

    # ✅ 好：进度更新
    async def long_running_task(self, task_id: str):
        for step in range(100):
            result = await do_work(step)
            yield {"progress": step, "result": result}
```

### 使用模式

#### 模式 1：增量消费流

```python
service = await LLMService.spawn()

# 在 token 到达时处理
async for chunk in service.generate("Hello, world!"):
    if chunk["type"] == "token":
        print(chunk["token"], end="", flush=True)
    elif chunk["type"] == "done":
        print(f"\n总 token 数: {chunk['total_tokens']}")
```

#### 模式 2：等待最终结果（跳过中间结果）

```python
# 如果只关心最终结果
result = await service.generate("Hello, world!")
# Pulsing 自动收集所有块并返回最终值
```

#### 模式 3：提前取消流

```python
async def consume_with_timeout():
    async with asyncio.timeout(5.0):
        async for chunk in service.generate("很长的提示..."):
            process(chunk)
    # 超时时自动取消流
```

### 性能特征

```
客户端:     [chunk1][chunk2][chunk3]...
            ↓       ↓       ↓
网络:      [████][████][████]...
            ↓       ↓       ↓
Actor:      [gen][gen][gen]...  ← 非阻塞生成
            ↓       ↓       ↓
LLM API:    [████████████████]...  ← 持续生成

总延迟: 第一个块快速到达，不等待所有块
```

---

## 4. Ask vs Tell

### 为什么需要两种模式？

**核心区别**：是否需要等待响应。

- **`ask()`**：需要响应，等待结果返回
- **`tell()`**：不需要响应，发送后立即继续

**为什么重要**：

```
❌ 所有操作都用 ask()：
await logger.ask({"level": "info", "msg": "..."})  # 等待响应
await metrics.ask({"event": "..."})                # 等待响应
await notifier.ask({"user": "..."})                 # 等待响应
问题：即使不需要结果，也要等待，降低吞吐量

✅ 区分使用：
await logger.tell({"level": "info", "msg": "..."})  # 不等待
await metrics.tell({"event": "..."})                # 不等待
result = await service.get_user("123")               # 需要结果，使用 ask
好处：不需要响应的操作不阻塞，吞吐量更高
```

### `ask()` - 请求/响应

**为什么使用**：需要知道操作结果或是否成功。

**何时使用：**
- 需要响应进行后续处理
- 需要知道操作是否成功
- 需要错误处理

```python
# ✅ 好：需要结果
result = await counter.increment(10)
print(f"新值: {result}")

# ✅ 好：需要检查成功
try:
    user = await service.get_user("user123")
except PulsingActorError:
    print("用户未找到")
```

### `tell()` - 发送即忘

**为什么使用**：最大化吞吐量，不需要等待响应。

**何时使用：**
- 不需要响应（日志、指标）
- 操作可以安全丢弃
- 想要最大吞吐量

```python
# ✅ 好：日志记录 - 不需要响应
await logger.tell({"level": "info", "message": "用户已登录"})

# ✅ 好：指标 - 发送即忘
await metrics.tell({"event": "page_view", "page": "/home"})

# ✅ 好：通知 - 最终交付即可
await notifier.tell({"user_id": "123", "message": "新邮件"})
```

### 对比

| 方面 | `ask()` | `tell()` |
|------|---------|----------|
| **响应** | ✅ 返回值 | ❌ 无响应 |
| **错误处理** | ✅ 抛出异常 | ❌ 静默失败 |
| **吞吐量** | 较低（等待响应） | 较高（不等待） |
| **使用场景** | 需要结果的操作 | 可以丢弃的操作 |

---

## 5. 快速决策指南

### 决策流程

```
开始：你的操作需要什么？

1. 需要响应吗？
   ├─ 否 → 使用 `tell()`（发送即忘）
   │      原因：不需要等待，最大化吞吐量
   │
   └─ 是 → 继续下一步

2. 操作需要多长时间？
   ├─ < 10ms → 使用 `def method()`（同步）
   │           原因：足够快，不需要并发，代码更简单
   │
   └─ > 10ms → 继续下一步

3. 需要增量返回结果吗？
   ├─ 否 → 使用 `async def method()`（异步）
   │       原因：避免阻塞，允许并发处理
   │
   └─ 是 → 使用 `async def method()` 带 `yield`（流式）
           原因：立即返回部分结果，提升用户体验
```

### 为什么这样选择？

| 选择 | 原因 |
|------|------|
| `tell()` | 不需要响应，不等待可以最大化吞吐量 |
| `def method()` | 快速操作不需要并发，同步代码更简单 |
| `async def method()` | 避免阻塞 Actor，允许并发处理多个请求 |
| `async def method()` + `yield` | 立即返回部分结果，提升用户体验 |

---

## 6. 实际示例

### 示例 1：计数器服务

```python
@pul.remote
class Counter:
    def __init__(self):
        self.value = 0

    # ✅ 同步：快速状态变更
    def increment(self, n: int = 1) -> int:
        self.value += n
        return self.value

    # ✅ 同步：简单读取
    def get(self) -> int:
        return self.value

    # ✅ 同步：快速操作
    def reset(self) -> None:
        self.value = 0
```

**为什么使用同步？**
- 所有操作都很快（< 1ms）
- 无 I/O 操作，纯内存操作
- 不需要并发，顺序执行即可
- 同步代码更简单，易于理解

**如果改用异步会怎样？**
- ❌ 增加不必要的 `async/await` 开销
- ❌ 代码更复杂，但没有性能提升
- ❌ 操作太快，并发带来的收益为零

---

### 示例 2：HTTP API 客户端

```python
@pul.remote
class APIClient:
    # ✅ 异步：网络 I/O
    async def fetch_data(self, url: str) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)  # 等待期间，Actor 可以处理其他请求
            return response.json()

    # ✅ 异步：多个并发请求
    async def fetch_multiple(self, urls: list[str]) -> list[dict]:
        tasks = [self.fetch_data(url) for url in urls]
        return await asyncio.gather(*tasks)  # 并发执行，不是顺序执行
```

**为什么使用异步？**
- 网络请求需要时间（通常 50-500ms）
- 如果使用同步，Actor 会被阻塞，无法处理其他请求
- 使用异步，Actor 可以在等待 HTTP 响应时处理其他消息
- 多个请求可以并发执行，大幅提升吞吐量

**如果改用同步会怎样？**
- ❌ Actor 在等待 HTTP 响应时无法处理任何其他消息
- ❌ 吞吐量极低（一次只能处理一个请求）
- ❌ 用户体验差（所有请求排队等待）

---

### 示例 3：LLM 服务

```python
@pul.remote
class LLMService:
    # ✅ 流式：Token 增量到达
    async def generate(self, prompt: str):
        async for token in self.llm_client.stream(prompt):
            yield {"token": token}  # 立即返回每个 token
        yield {"done": True}

    # ✅ 异步：单次完成（不需要流式）
    async def embed(self, text: str) -> list[float]:
        return await self.llm_client.embed(text)  # 快速完成，不需要流式
```

**为什么 `generate` 使用流式？**
- LLM 生成需要时间（可能 5-30 秒）
- 如果等待全部完成，用户需要等待很久才能看到任何内容
- 使用流式，用户立即看到第一个 token，体验更好
- 用户可以提前取消（如果不需要了）

**为什么 `embed` 使用异步而不是流式？**
- Embedding 操作通常很快（< 1 秒）
- 结果是单个向量，不需要增量返回
- 使用异步避免阻塞即可，不需要流式

**如果 `generate` 不使用流式会怎样？**
- ❌ 用户需要等待 10-30 秒才能看到任何输出
- ❌ 无法提前取消（即使不需要了也要等待）
- ❌ 用户体验极差

---

### 示例 4：混合模式

```python
@pul.remote
class DataProcessor:
    def __init__(self):
        self.processed_count = 0  # 快速状态更新

    # ✅ 同步：快速计数器更新
    def get_stats(self) -> dict:
        return {"processed": self.processed_count}

    # ✅ 异步：I/O 操作
    async def fetch_from_db(self, query: str) -> list[dict]:
        return await database.query(query)

    # ✅ 流式：增量处理大数据集
    async def process_large_dataset(self, dataset_id: str):
        async for record in self.fetch_records(dataset_id):
            processed = await self.process_record(record)
            self.processed_count += 1  # 快速更新
            yield {"record": processed, "count": self.processed_count}
```

**为什么混合？** 不同操作有不同的特性 - 为每个操作使用正确的工具。

---

## 7. 性能对比：理解差异

### 场景：处理 1000 个请求

#### 同步方法（顺序执行）

```python
def process(self, data: str) -> str:
    return process_data(data)  # 每个 2ms
```

**执行时间线**：
```
请求1: [████] 2ms
请求2:      [████] 2ms
请求3:          [████] 2ms
...
请求1000:                    [████] 2ms
总计: 2000ms（2秒）
```

**为什么慢？** 必须等待前一个请求完成才能处理下一个。

#### 异步方法（并发执行）

```python
async def process(self, data: str) -> str:
    result = await external_api(data)  # 每个 50ms（等待网络）
    return result
```

**执行时间线**：
```
请求1-1000: [████████████████████████████████] 50ms（全部并发）
总计: ~50ms（不是 50秒！）
```

**为什么快？** 所有请求并发执行，Actor 在等待网络响应时可以处理其他请求。

#### 流式（增量返回）

```python
async def process(self, data: str):
    for chunk in split_data(data):
        result = await process_chunk(chunk)
        yield result  # 立即返回
```

**执行时间线**：
```
客户端收到第一个结果: [██] 10ms  ← 立即看到！
客户端收到所有结果:   [████████████████████] 50ms
```

**为什么更好？** 用户不需要等待全部完成，可以立即开始处理第一个结果。

### 关键理解

- **同步**：顺序执行，简单但慢（适合快速操作）
- **异步**：并发执行，快但需要 `async/await`（适合 I/O 操作）
- **流式**：增量返回，用户体验好（适合长时间操作）

---

## 8. 常见陷阱：理解为什么错误

### ❌ 陷阱 1：对 I/O 使用同步

**问题**：阻塞 Actor，无法处理其他消息。

```python
# ❌ 差：在 HTTP 请求期间阻塞 Actor
def fetch_data(self, url: str) -> dict:
    response = requests.get(url)  # 阻塞数秒！
    return response.json()
# 结果：Actor 在这几秒内无法处理任何其他消息
```

**为什么错误？**
- Actor 被阻塞，无法处理其他请求
- 吞吐量极低（一次只能处理一个请求）
- 用户体验差（所有请求排队）

```python
# ✅ 好：非阻塞异步
async def fetch_data(self, url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)  # 等待期间可以处理其他请求
        return response.json()
# 结果：Actor 可以并发处理多个请求
```

### ❌ 陷阱 2：对快速操作使用异步

**问题**：增加不必要的复杂度，没有性能提升。

```python
# ❌ 差：不必要的异步开销
async def increment(self, n: int) -> int:
    self.value += n  # 这个操作只需要 < 1ms
    return self.value
# 问题：操作太快，并发带来的收益为零，但代码更复杂
```

**为什么错误？**
- 操作太快（< 1ms），不需要并发
- 增加 `async/await` 语法复杂度
- 没有性能提升

```python
# ✅ 好：简单同步方法
def increment(self, n: int) -> int:
    self.value += n
    return self.value
# 结果：代码更简单，性能相同
```

### ❌ 陷阱 3：LLM 不使用流式

**问题**：用户体验差，需要等待很久。

```python
# ❌ 差：等待所有 token
async def generate(self, prompt: str) -> str:
    tokens = []
    async for token in self.llm_client.stream(prompt):
        tokens.append(token)
    return "".join(tokens)  # 用户等待 10-30 秒才能看到任何内容
# 问题：用户需要等待全部完成，无法提前取消
```

**为什么错误？**
- 用户需要等待 10-30 秒才能看到任何输出
- 无法提前取消（即使不需要了）
- 用户体验极差

```python
# ✅ 好：token 到达时流式传输
async def generate(self, prompt: str):
    async for token in self.llm_client.stream(prompt):
        yield token  # 用户立即看到 token
# 结果：用户立即看到输出，可以提前取消
```

### ❌ 陷阱 4：对发送即忘使用 Ask

**问题**：不必要的等待，降低吞吐量。

```python
# ❌ 差：不必要的等待
await logger.ask({"level": "info", "msg": "..."})  # 等待响应
# 问题：即使不需要结果，也要等待，降低吞吐量
```

**为什么错误？**
- 不需要响应，但还是要等待
- 降低吞吐量（所有日志操作都要等待）
- 增加延迟

```python
# ✅ 好：发送即忘
await logger.tell({"level": "info", "msg": "..."})  # 不等待
# 结果：最大化吞吐量，不阻塞
```

---

## 9. 最佳实践总结

### 核心原则

1. **快速操作（< 10ms）**：使用 `def method()`（同步）
   - **原因**：足够快，不需要并发，代码更简单

2. **I/O 操作（> 10ms）**：使用 `async def method()`（异步）
   - **原因**：避免阻塞 Actor，允许并发处理

3. **增量结果**：使用 `async def method()` 带 `yield`（流式）
   - **原因**：立即返回部分结果，提升用户体验

4. **不需要响应**：使用 `tell()`（发送即忘）
   - **原因**：最大化吞吐量，不阻塞

5. **需要响应**：使用 `ask()` 或方法调用
   - **原因**：需要知道操作结果或是否成功

6. **LLM token 生成**：始终使用流式
   - **原因**：生成时间长，用户希望立即看到输出

7. **多个并发操作**：使用 `async def` 配合 `asyncio.gather()`
   - **原因**：并发执行，而不是顺序执行

---

## 10. 快速参考

| 操作类型 | 范式 | 示例 |
|----------|------|------|
| 计数器递增 | `def increment()` | 快速状态更新 |
| HTTP 请求 | `async def fetch()` | 网络 I/O |
| 数据库查询 | `async def query()` | I/O 操作 |
| LLM 生成 | `async def generate()` 带 `yield` | 流式 token |
| 文件处理 | `async def process()` 带 `yield` | 大数据 |
| 日志记录 | `tell()` | 发送即忘 |
| 指标收集 | `tell()` | 发送即忘 |
| 获取结果 | `ask()` 或 `await method()` | 需要响应 |

---

## 下一步

- 了解[错误处理](error_handling.md)以实现健壮的通信
- 查看[可靠性指南](reliability.md)了解超时和重试模式
- 查看[示例](../examples/index.md)了解更多实际模式
