# 语义与保证（Semantics）

本文明确 Pulsing 在以下方面**保证什么**、**不保证什么**，避免在生产环境里误用更强的语义假设：

- Actor 执行语义
- 远程消息（`ask` / `tell`）
- 流式响应（`StreamMessage`）
- 分布式内存队列（`pulsing.queue`）

## TL;DR（快速结论）

- **单个 Actor 内部是串行处理**：同一个 actor 的 `receive` 不会并发执行。
- **没有 exactly-once**：Pulsing 不提供端到端“恰好一次”投递/处理保证。
- **队列没有 consumer group**：读取是基于 offset 的，且**默认不删除数据**；要做到“只消费一次”需要应用层实现 ack/去重。
- **流式响应有背压**：`StreamMessage.create(..., buffer_size=32)` 使用有界队列，写端 `await write()` 会在缓冲满时阻塞。

## 术语

- **投递（delivery）**：消息到达目标 actor。
- **处理（processing）**：目标 actor 执行 `receive`。
- **At-most-once**：0 或 1 次（可能丢）。
- **At-least-once**：至少 1 次（可能重复）。
- **Exactly-once**：恰好 1 次（通常需要去重 + 提交协议）。

## Actor 执行语义

### 顺序与并发（单 Actor 内）

- **保证**：同一个 actor 的消息处理是**串行**的（不会同时跑两个 `receive`）。
- **不保证**：跨 actor 的全局顺序；不同 actor 之间没有统一的顺序关系。

实践建议：

- 任何强依赖顺序的状态变化都应封装在 actor 内，通过消息串行化。

### 异常与失败

- `receive` 内抛异常或返回错误（如 `"Error"`）时，调用方会观察到失败（通常表现为异常或错误消息）。
- Pulsing 支持 **actor 级别重启**（可配置重启策略 + 退避），但 **不会引入 supervision tree**。
  - 参见：[可靠性实践](reliability.zh.md)

## 远程消息语义（`ask` / `tell`）

### `ask`（请求-响应）

`ask` 会返回：

- 单条 `Message`；或
- 流式 `Message`（`msg.is_stream == True`），通过 `msg.stream_reader()` 消费。

**你可以安全依赖的保证**：

- 如果你**收到了响应**，该响应对应目标 actor 实例的一次 `receive` 执行结果。

**你不应该假设**：

- 端到端 exactly-once（应用层重试会引入重复）。
- 多个并发请求在网络上的稳定顺序（网络/连接复用/调度都会影响到达顺序）。

关于超时/重试：

- API 层面不应默认假设“自动重试”。
- 超时与失败可能来自网络分区、节点故障、过载等；应用层需要显式制定重试/退避策略，并配合幂等设计。

### `tell`（单向发送）

- `tell` 无响应，因此**没有投递确认**。
- 适用于“可丢弃”或“允许最终对账/纠偏”的消息。

## 流式语义（`StreamMessage`）

`StreamMessage.create(msg_type, buffer_size=32)` 返回 `(stream_msg, writer)`。

### 流的组成

- 流由 **Message 对象**组成，而非原始字节。
- 流中的每个 chunk 都是完整的 `Message`，有自己的 `msg_type` 和 payload。
- 这使得**异构流**成为可能，不同的 chunk 可以有不同的类型。

### 透明的 Python 对象流式传输

对于 Python 到 Python 的通信，流式传输是**完全透明的**：

```python
# 写入端 - 直接写入 Python 对象
async def generate_stream():
    stream_msg, writer = StreamMessage.create("tokens")
    for token in tokens:
        await writer.write({"token": token, "index": i})  # dict 自动 pickle
    await writer.close()
    return stream_msg

# 读取端 - 直接接收 Python 对象
async for chunk in response.stream_reader():
    token = chunk["token"]  # chunk 已经是 Python dict
    print(token)
```

### 背压与缓冲

- 底层是一个**有界**的 channel（大小为 `buffer_size`）。
- `writer.write(...)` 在 buffer 满时会 `await` → 自然背压。

### 生命周期

流可能因以下原因结束：

- `writer.close()`：正常结束
- `writer.error("...")`：带错误结束
- 消费端 `reader.cancel()` 或 reader 被丢弃

### 投递语义

- 流式 chunk 是 best-effort，可能出现"部分输出后中断"。
- 建议每个 chunk 带上 `seq` / offset / id，让消费端可恢复/去重。

## 队列语义（`pulsing.queue`）

队列按 bucket 分片：

- `bucket_id = md5(str(bucket_value)) % num_buckets`
- 集群内每个 bucket 由一个 `BucketStorage` actor 服务（通过一致性哈希选择 owner 节点）

### 可见性模型（内存缓冲 + 持久化）

`BucketStorage` 同时维护：

- 持久化部分（Lance dataset，若可用）
- 内存缓冲部分（未 flush 的记录）

**保证**：

- `put` 成功后，记录会**立即对读者可见**（至少在内存缓冲中可读到）。

**不保证**：

- 没有安装 `lance/pyarrow` 或 flush 失败时，不保证落盘持久化。

### 读取语义：基于 offset，默认不删除

- 读取是 `offset + limit` 模式，本质是“日志式读取”，不会删除数据。
- `QueueReader` 在客户端侧为**每个 bucket**维护 offset。

含义：

- 多个 reader 可能读取到同一批数据（除非你实现协调/ack）。
- 不存在内建的 consumer group / exactly-once 处理语义。

### 阻塞读取（`wait` / `timeout`）

- `GetStream` 支持 `wait=True` 等待新数据。
- 超时会结束流，返回少量或 0 条记录。

### 顺序

- 单 bucket 内：按 offset 顺序读取（追加顺序）。
- 多 bucket：**没有全局顺序**；“读所有 bucket”相当于多路合并。

### 需要应用层自行处理的失败模式

- 节点故障/网络分区可能让某个 bucket 暂时不可用，直到路由/成员信息收敛。
- 应用层重试可能导致重复写入/重复处理 → 用幂等与去重来兜底。

## 推荐的应用层模式

- **幂等键（idempotency key）**：写入/处理时携带稳定 `id`，消费端去重。
- **显式 ack**：用 actor 状态（或单独的 commit log）记录“已处理到哪个 offset”。
- **显式超时 + 重试策略**：把策略放在业务层，避免隐式重试带来的语义误解。
