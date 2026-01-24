# 常见问题解答

此页面解答用户在使用 Pulsing 时遇到的常见问题和问题。

## 一般问题

### 什么是 Pulsing？

Pulsing 是一个分布式 actor 框架，为构建分布式系统提供通信骨干，并为 AI 应用提供专门支持。

### Pulsing 与 Ray 有何区别？

Ray 专注于通用分布式计算和基于任务的并行性，而 Pulsing 专门针对 Actor 模型：

- **位置透明性**：本地和远程 actor 使用相同 API
- **真正的 actor 语义**：Actor 一次处理一条消息
- **零外部依赖**：纯 Rust + Tokio 实现
- **流式支持**：原生支持流式响应

### 何时应该使用 Pulsing 而不是 Ray？

当你需要以下特性时选择 Pulsing：

- 基于 actor 的编程和位置透明性
- 流式响应（LLM 应用）
- 最小的运维复杂度（无需外部服务）
- 高性能 actor 通信

当你需要以下特性时选择 Ray：

- 通用分布式计算任务
- 复杂的依赖管理
- 与现有 Ray 生态系统集成

## 安装问题

### ImportError: No module named 'pulsing'

**问题**：Pulsing 包未安装或不在 Python 路径中。

**解决方案**：

1. **安装 Pulsing**：
   ```bash
   pip install pulsing
   ```

2. **开发环境**：
   ```bash
   git clone https://github.com/DeepLink-org/pulsing
   cd pulsing
   pip install -e .
   ```

3. **检查 Python 路径**：
   ```python
   import sys
   print(sys.path)
   ```

### macOS/Linux 上的构建失败

**问题**：Rust 编译问题。

**解决方案**：

1. **安装 Rust**：
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   ```

2. **安装系统依赖**（Ubuntu/Debian）：
   ```bash
   sudo apt-get install build-essential pkg-config libssl-dev
   ```

3. **安装系统依赖**（macOS）：
   ```bash
   brew install openssl pkg-config
   ```

## 运行时问题

### Actor 不响应消息

**问题**：Actor 似乎卡住或不处理消息。

**可能原因**：

1. **阻塞操作**：Actor 在同步 I/O 上阻塞
2. **无限循环**：Actor 代码包含无限循环
3. **死锁**：Actor 正在等待永远不会到达的消息

**解决方案**：

```python
# ❌ 错误：在 actor 中使用阻塞 I/O
@pul.remote
class BadActor:
    def process(self, url):
        response = requests.get(url)  # 阻塞 actor！
        return response.text

# ✅ 正确：使用异步 I/O
@pul.remote
class GoodActor:
    async def process(self, url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.text()
```

### 连接拒绝错误

**问题**：无法连接到远程 actor。

**可能原因**：

1. **地址错误**：Actor 系统监听不同的地址
2. **防火墙**：网络流量被阻塞
3. **TLS 问题**：证书验证失败

**解决方案**：

1. **检查 actor 系统地址**：
   ```python
   # 确保地址匹配
   system1 = await pul.actor_system(addr="0.0.0.0:8000")
   system2 = await pul.actor_system(addr="0.0.0.0:8001", seeds=["127.0.0.1:8000"])
   ```

2. **测试时禁用 TLS**：
   ```python
   # 仅用于开发环境
   system = await pul.actor_system(addr="0.0.0.0:8000", passphrase=None)
   ```

### 内存泄漏

**问题**：内存使用量随时间增长。

**可能原因**：

1. **消息积累**：消息处理不够快
2. **大型消息负载**：消息包含大型数据结构
3. **Actor 泄漏**：Actor 未正确清理

**解决方案**：

1. **监控邮箱大小**：
   ```python
   # 检查 actor 邮箱大小
   mailbox_size = await system.get_mailbox_size("actor_name")
   ```

2. **对大型数据使用流式处理**：
   ```python
   @pul.remote
   class StreamingActor:
       async def process_large_data(self, data_stream):
           async for chunk in data_stream:
               # 分块处理
               yield self.process_chunk(chunk)
   ```

## 性能问题

### 高延迟

**问题**：消息往返耗时太长。

**优化方案**：

1. **尽可能使用本地 actor**：
   ```python
   # 本地 actor（快速）
   local_actor = await MyActor.spawn()

   # 远程 actor（较慢）
   remote_actor = await MyActor.resolve("remote_actor")
   ```

2. **批处理消息**：
   ```python
   # 不要进行多次调用
   results = []
   for item in items:
       result = await actor.process(item)
       results.append(result)

   # 批处理
   results = await actor.process_batch(items)
   ```

3. **对无需响应的操作使用 tell()**：
   ```python
   # 如果不需要响应，不要等待
   await actor.log_event(event_data)  # 在内部使用 ask()
   await actor.tell({"action": "log", "data": event_data})  # 发射后不管
   ```

### 序列化开销

**问题**：消息序列化很慢。

**解决方案**：

1. **使用高效的数据格式**：
   ```python
   # ✅ 良好：使用简单类型
   await actor.process({"numbers": [1, 2, 3], "text": "hello"})

   # ❌ 错误：复杂的嵌套对象
   await actor.process({"data": very_complex_nested_object})
   ```

2. **避免发送大型负载**：
   ```python
   # 发送引用而不是数据
   await actor.process_data(data_id)  # 发送 ID，而不是数据本身
   ```

## 部署问题

### 集群无法正常工作

**问题**：多个节点无法相互发现。

**解决方案**：

1. **检查种子节点配置**：
   ```python
   # 节点 1（种子）
   system1 = await pul.actor_system(addr="192.168.1.100:8000")

   # 节点 2（加入集群）
   system2 = await pul.actor_system(
       addr="192.168.1.101:8000",
       seeds=["192.168.1.100:8000"]
   )
   ```

2. **验证网络连接**：
   ```bash
   # 测试端口是否开放
   telnet 192.168.1.100 8000
   ```

3. **检查防火墙设置**：
   ```bash
   # Linux
   sudo ufw status
   sudo ufw allow 8000

   # macOS
   sudo pfctl -s rules
   ```

### 负载均衡问题

**问题**：请求未在集群中均匀分布。

**解决方案**：

1. **使用轮询解析**：
   ```python
   # 默认行为在实例间分布
   actor = await MyActor.resolve("service_name")
   ```

2. **检查 actor 分布**：
   ```python
   # 监控集群成员
   members = await system.members()
   print(f"Cluster has {len(members)} nodes")
   ```

## 迁移问题

### 从 Ray 迁移

**常见问题**：

1. **API 差异**：
   ```python
   # Ray
   @ray.remote
   class MyActor:
       def __init__(self, value):
           self.value = value

   actor = MyActor.remote(42)
   result = ray.get(actor.method.remote())

   # Pulsing
   @pul.remote
   class MyActor:
       def __init__(self, value):
           self.value = value

   actor = await MyActor.spawn(value=42)
   result = await actor.method()
   ```

2. **处处需要 async/await**：
   ```python
   # Pulsing 需要 async/await
   async def main():
       await pul.init()
       actor = await MyActor.spawn()
       result = await actor.method()
       await pul.shutdown()

   asyncio.run(main())
   ```

## 获取帮助

如果在此处找不到答案：

1. **查看文档**：[用户指南](../guide/) 和 [API 参考](../api/overview.md)
2. **搜索现有问题**：[GitHub Issues](https://github.com/DeepLink-org/pulsing/issues)
3. **咨询社区**：[GitHub Discussions](https://github.com/DeepLink-org/pulsing/discussions)
4. **提交错误报告**：如果发现 bug，请[创建 issue](https://github.com/DeepLink-org/pulsing/issues/new)

## 贡献

发现此 FAQ 有问题？[帮助改进它！](https://github.com/DeepLink-org/pulsing/blob/main/docs/src/faq.md)
