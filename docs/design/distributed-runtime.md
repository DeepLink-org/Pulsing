# Pulsing 分布式 Runtime 设计文档

## 概述

本文档描述基于 Pulsing Actor System 的分布式 Runtime 设计。该设计采用纯 Actor 模型，所有组件（包括系统管理、路由、计算等）都以 Actor 形式存在，通过消息传递进行通信，实现一个轻量级、零外部依赖的分布式推理服务框架。

## 设计目标

1. **纯 Actor 模型** - 没有 Runtime 抽象层，所有组件都是 Actor
2. **零外部依赖** - 不依赖 etcd、NATS 等外部服务
3. **原生流式支持** - Actor 层面直接支持流式请求/响应
4. **Gossip 元数据同步** - 使用 Gossip 协议同步集群状态和元数据
5. **简洁的 API** - 重新设计的 API，更符合 Actor 模型思想

## 与 Dynamo Runtime 的对比

| 特性 | Dynamo Runtime | Pulsing Actor Runtime |
|------|---------------|----------------------|
| 外部依赖 | etcd + NATS（必需） | 无 |
| 核心抽象 | DistributedRuntime → Namespace → Component → Endpoint | ActorSystem → SystemActor + 业务 Actors |
| 服务发现 | etcd KV Store / Kubernetes | Gossip 协议 (SWIM) |
| 元数据存储 | etcd | Gossip 同步的分布式存储 |
| 通信模式 | NATS/HTTP/TCP + 流式 | Actor 消息 (ask/tell) + 统一 Message 类型 |
| 流式响应 | 外部 Stream 抽象 | Message::Stream 原生支持 |

## 核心架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Pulsing Actor Runtime                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         ActorSystem                                  │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │                    SystemActor (内置)                        │   │   │
│  │   │                    actor://localhost/system                  │   │   │
│  │   │                                                              │   │   │
│  │   │   • 集群成员管理 (Gossip/SWIM)                               │   │   │
│  │   │   • Named Actor 注册表同步                                   │   │   │
│  │   │   • 元数据存储与同步                                         │   │   │
│  │   │   • 诊断接口 (健康检查、状态查询)                            │   │   │
│  │   │                                                              │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                       │   │
│  │                              │ 管理                                  │   │
│  │                              ▼                                       │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │                  业务 Actors (用户创建)                      │   │   │
│  │   │                                                              │   │   │
│  │   │   ┌───────────┐  ┌───────────┐  ┌───────────┐              │   │   │
│  │   │   │ Frontend  │  │  Router   │  │  Worker   │   ...        │   │   │
│  │   │   │  Actor    │  │  Actor    │  │  Actor    │              │   │   │
│  │   │   └───────────┘  └───────────┘  └───────────┘              │   │   │
│  │   │                                                              │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Transport Layer (HTTP/2)                          │   │
│  │                                                                       │   │
│  │   • ask(msg) → Message     请求/响应 (Single 或 Stream)               │   │
│  │   • tell(msg)              单向消息 (fire-and-forget)                 │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 进程内部结构

每个进程包含：
- **1 个 ActorSystem 实例** - 管理传输层和 Actor 生命周期
- **1 个 SystemActor（自动创建）** - 处理系统级操作
- **N 个业务 Actor（用户创建）** - 处理具体业务逻辑

```
┌─────────────────────────────────────────┐
│  进程 (Process)                          │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  ActorSystem                        │ │
│  │                                     │ │
│  │  ┌───────────────────────────────┐ │ │
│  │  │  SystemActor (自动)           │ │ │
│  │  │  actor://localhost/system     │ │ │
│  │  └───────────────────────────────┘ │ │
│  │                                     │ │
│  │  ┌─────────┐ ┌─────────┐ ┌──────┐ │ │
│  │  │ Actor A │ │ Actor B │ │ ...  │ │ │
│  │  │ (业务)  │ │ (业务)  │ │      │ │ │
│  │  └─────────┘ └─────────┘ └──────┘ │ │
│  │                                     │ │
│  └────────────────────────────────────┘ │
│                                          │
└─────────────────────────────────────────┘
```

## SystemActor 设计

### 职责

SystemActor 是每个进程自动创建的系统级 Actor，负责：

1. **集群管理**
   - 成员发现和维护 (Gossip/SWIM)
   - 节点加入/离开处理
   - 故障检测

2. **Actor 注册表**
   - 本地 Actor 管理
   - Named Actor 全局注册表同步
   - Actor 位置查询

3. **元数据存储**
   - 分布式 Key-Value 存储
   - 元数据变更监听
   - Gossip 同步

4. **诊断接口**
   - 健康检查
   - 系统状态查询
   - 性能指标

### 数据结构

```rust
pub struct SystemActor {
    node_id: NodeId,
    
    // ========== 集群管理 ==========
    /// 集群成员信息
    members: HashMap<NodeId, MemberInfo>,
    /// SWIM 故障检测器
    swim: SwimDetector,
    /// Gossip 配置
    gossip_config: GossipConfig,
    
    // ========== Actor 注册表 ==========
    /// 本地 Actor 列表
    local_actors: HashMap<String, LocalActorHandle>,
    /// Named Actor 全局注册表
    named_registry: HashMap<String, NamedActorInfo>,
    
    // ========== 元数据存储 ==========
    /// 分布式元数据
    metadata: MetadataStore,
    /// 元数据观察者
    metadata_watchers: HashMap<String, Vec<WatcherInfo>>,
    
    // ========== 诊断 ==========
    /// 启动时间
    start_time: Instant,
    /// 系统指标
    metrics: SystemMetrics,
}
```

### 消息协议

#### 集群管理消息

```rust
/// 获取集群成员列表
pub struct GetMembers;
pub struct MemberList {
    pub members: Vec<MemberInfo>,
}

/// 加入集群
pub struct JoinCluster {
    pub seed_addrs: Vec<SocketAddr>,
}

/// 离开集群
pub struct LeaveCluster;
```

#### Actor 管理消息

```rust
/// 列出本地 Actor
pub struct ListLocalActors;
pub struct LocalActorList {
    pub actors: Vec<ActorInfo>,
}

/// 查询 Named Actor
pub struct LookupNamed {
    pub path: String,
}
pub struct NamedActorInfo {
    pub path: String,
    pub instances: Vec<NodeId>,
}
```

#### 元数据操作消息

```rust
/// 获取元数据
pub struct GetMetadata {
    pub namespace: String,
    pub key: String,
}

/// 设置元数据
pub struct SetMetadata {
    pub namespace: String,
    pub key: String,
    pub value: Vec<u8>,
    pub ttl: Option<Duration>,
}

/// 删除元数据
pub struct DeleteMetadata {
    pub namespace: String,
    pub key: String,
}

/// 列出元数据
pub struct ListMetadata {
    pub namespace: String,
    pub key_prefix: String,
}

/// 监听元数据变化 (流式响应)
pub struct WatchMetadata {
    pub namespace: String,
    pub key_prefix: String,
}

/// 元数据变更事件
pub struct MetadataChanged {
    pub namespace: String,
    pub key: String,
    pub value: Option<Vec<u8>>,  // None 表示删除
    pub version: u64,
}
```

#### 诊断消息

```rust
/// 健康检查
pub struct HealthCheck;
pub struct HealthStatus {
    pub healthy: bool,
    pub uptime_secs: u64,
    pub actor_count: usize,
    pub cluster_size: usize,
}

/// 获取系统状态
pub struct GetStatus;
pub struct SystemStatus {
    pub node_id: String,
    pub addr: SocketAddr,
    pub uptime_secs: u64,
    pub actors: Vec<ActorInfo>,
    pub cluster: ClusterStatus,
    pub metrics: MetricsSnapshot,
}

/// Ping
pub struct Ping { pub timestamp: u64 }
pub struct Pong { pub timestamp: u64, pub node_id: String }
```

## 消息类型与流式响应设计

### 设计理念

核心思想：**统一的 Message 类型支持单次和流式两种模式**，无需区分 `ask` 和 `ask_stream`。

传统回调模式的问题：
- 每个 token 一次网络调用，开销大
- 需要临时 Actor 管理生命周期
- 背压和取消机制复杂

**解决方案**：使用 `Message` 枚举统一表示单次消息和流式消息

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     消息类型：统一的 Message 枚举                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Message 枚举:                                                              │
│  ─────────────                                                              │
│  • Message::Single { msg_type, data }     → 单次请求/响应                   │
│  • Message::Stream { msg_type, stream }   → 流式请求/响应                   │
│                                                                             │
│  通信模式:                                                                  │
│  ───────────                                                                │
│  • Single → Single   传统 RPC (查询、命令)                                  │
│  • Single → Stream   服务端流式 (LLM 生成)                                  │
│  • Stream → Single   客户端流式 (上传)                                      │
│  • Stream → Stream   双向流式                                               │
│                                                                             │
│  示例 - LLM 推理流程:                                                       │
│  ───────────────────────                                                    │
│  Client → Frontend → Router → Worker                                        │
│              │                    │                                         │
│              │  ask(Generate)     │     // 发送 Single 请求                 │
│              │═══════════════════►│                                         │
│              │◄═══ Message::Stream│     // 返回 Stream 响应                 │
│              │    (token stream)  │                                         │
│                                                                             │
│  优势:                                                                      │
│  • 统一 API：ask() 同时支持单次和流式                                       │
│  • 类型自描述：响应类型由 Message 变体决定                                  │
│  • 背压由 Stream 机制自然处理                                               │
│  • Drop stream = 取消请求                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Message 类型定义

```rust
/// 统一消息类型，支持单次和流式两种模式
pub enum Message {
    /// 单次数据消息
    Single {
        /// 消息类型标识 (用于路由和反序列化)
        msg_type: String,
        /// 消息数据 (序列化的 payload)
        data: Vec<u8>,
    },
    /// 流式数据消息
    Stream {
        /// 消息类型标识
        msg_type: String,
        /// 数据流
        stream: Pin<Box<dyn Stream<Item = Result<Vec<u8>>> + Send>>,
    },
}

impl Message {
    /// 创建单次消息
    pub fn single(msg_type: impl Into<String>, data: impl Into<Vec<u8>>) -> Self {
        Message::Single {
            msg_type: msg_type.into(),
            data: data.into(),
        }
    }
    
    /// 打包可序列化对象为 Single 消息
    pub fn pack<T: Serialize + 'static>(value: &T) -> Result<Self> {
        Ok(Message::Single {
            msg_type: std::any::type_name::<T>().to_string(),
            data: bincode::serialize(value)?,
        })
    }
    
    /// 从 Single 消息解包
    pub fn unpack<T: DeserializeOwned>(self) -> Result<T> {
        match self {
            Message::Single { data, .. } => Ok(bincode::deserialize(&data)?),
            Message::Stream { .. } => Err(anyhow!("Cannot unpack stream message")),
        }
    }
    
    /// 创建流式消息
    pub fn stream<S>(msg_type: impl Into<String>, stream: S) -> Self
    where
        S: Stream<Item = Result<Vec<u8>>> + Send + 'static,
    {
        Message::Stream {
            msg_type: msg_type.into(),
            stream: Box::pin(stream),
        }
    }
    
    /// 检查是否为流式消息
    pub fn is_stream(&self) -> bool {
        matches!(self, Message::Stream { .. })
    }
    
    /// 获取消息类型
    pub fn msg_type(&self) -> &str {
        match self {
            Message::Single { msg_type, .. } => msg_type,
            Message::Stream { msg_type, .. } => msg_type,
        }
    }
}
```

### ActorRef API

```rust
impl ActorRef {
    /// 发送消息并等待响应
    /// 
    /// 返回的 Message 可以是 Single 或 Stream，调用方根据业务逻辑处理
    pub async fn send(&self, msg: Message) -> Result<Message>;
    
    /// 发送消息，不等待响应 (fire-and-forget)
    pub async fn fire(&self, msg: Message) -> Result<()>;
    
    /// 类型化的请求-响应 (便捷方法)
    /// 
    /// 自动将请求序列化为 Single 消息，将响应反序列化
    /// 如果响应是 Stream，会报错
    pub async fn ask<M, R>(&self, msg: M) -> Result<R>
    where
        M: Serialize + 'static,
        R: DeserializeOwned,
    {
        let request = Message::pack(&msg)?;
        let response = self.send(request).await?;
        response.unpack()
    }
    
    /// 类型化的单向消息 (便捷方法)
    pub async fn tell<M>(&self, msg: M) -> Result<()>
    where
        M: Serialize + 'static,
    {
        let message = Message::pack(&msg)?;
        self.fire(message).await
    }
}
```

### Actor Trait

```rust
#[async_trait]
pub trait Actor: Send + Sync + 'static {
    /// 返回 Actor 元数据 (可选，用于诊断)
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }
    
    /// Actor 启动时调用
    async fn on_start(&mut self, ctx: &mut ActorContext) -> Result<()> {
        Ok(())
    }
    
    /// Actor 停止时调用
    async fn on_stop(&mut self, ctx: &mut ActorContext) -> Result<()> {
        Ok(())
    }
    
    /// 处理消息
    /// 
    /// 输入和输出都是 Message，可以是 Single 或 Stream：
    /// - Single → Single: 传统 RPC
    /// - Single → Stream: 服务端流式 (如 LLM 生成)
    /// - Stream → Single: 客户端流式
    /// - Stream → Stream: 双向流式
    async fn receive(
        &mut self,
        msg: Message,
        ctx: &mut ActorContext,
    ) -> Result<Message>;
}

## Python Actor 流式支持

### 设计目标

Python Actor 需要支持流式请求和流式响应，同时保持简洁的 API 和与 Python 异步生态的良好集成。

### 核心问题

1. **入站流（接收流式请求）**：Rust 的 `Message::Stream` 如何暴露给 Python？
2. **出站流（返回流式响应）**：Python 如何返回 `Message::Stream`？
3. **任务管理**：流式处理应该 offload 到哪里？

### 设计方案

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Python 流式消息处理                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  入站流 (接收):                                                             │
│  ───────────────                                                            │
│  Rust Message::Stream  →  PyStreamReader  →  Python async for              │
│       (PayloadStream)      (async iterator)                                 │
│                                                                             │
│  出站流 (发送):                                                             │
│  ───────────────                                                            │
│  Python async generator  →  PyStreamMessage  →  Rust Message::Stream       │
│       or channel              (wraps channel)    (wraps receiver)          │
│                                                                             │
│  任务模型:                                                                  │
│  ──────────                                                                 │
│  • receive() 快速返回，不阻塞                                               │
│  • 流的生产/消费在 Python asyncio task 中进行                               │
│  • Rust runtime 负责网络传输                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Python 绑定类型

```python
# ============================================================================
# 流式读取器（入站流）
# ============================================================================

class StreamReader:
    """
    异步流读取器，用于从 Rust 流中读取数据。
    
    实现 Python 异步迭代器协议，可在 async for 中使用。
    """
    
    def __aiter__(self):
        return self
    
    async def __anext__(self) -> bytes:
        """读取下一个 chunk，流结束时抛出 StopAsyncIteration"""
        ...
    
    async def read_json(self) -> Any:
        """读取下一个 chunk 并解析为 JSON"""
        ...
    
    def cancel(self) -> None:
        """取消流（通知发送方停止）"""
        ...


# ============================================================================
# 流式消息（出站流）
# ============================================================================

class StreamMessage:
    """
    流式响应消息，用于 Python Actor 返回流式数据。
    
    两种使用方式：
    1. 基于 channel: 创建 channel，在 task 中写入
    2. 基于 async generator: 直接包装 async generator
    """
    
    @staticmethod
    def from_channel(msg_type: str) -> Tuple['StreamMessage', 'StreamWriter']:
        """
        创建基于 channel 的流式消息。
        
        Returns:
            (message, writer): message 用于返回，writer 用于写入数据
        """
        ...
    
    @staticmethod
    def from_generator(msg_type: str, gen: AsyncGenerator[bytes, None]) -> 'StreamMessage':
        """
        从 async generator 创建流式消息。
        
        generator 的执行会被 offload 到 Python task 中。
        """
        ...


class StreamWriter:
    """
    流式写入器，用于向流中写入数据。
    """
    
    async def write(self, data: bytes) -> None:
        """写入一个 chunk"""
        ...
    
    async def write_json(self, obj: Any) -> None:
        """将对象序列化为 JSON 并写入"""
        ...
    
    def close(self) -> None:
        """关闭流（正常结束）"""
        ...
    
    def error(self, msg: str) -> None:
        """以错误关闭流"""
        ...


# ============================================================================
# 扩展的 Message 类型
# ============================================================================

class Message:
    """统一消息类型，支持单次和流式"""
    
    @property
    def is_stream(self) -> bool:
        """是否为流式消息"""
        ...
    
    def stream_reader(self) -> StreamReader:
        """
        获取流读取器（仅对流式消息有效）
        
        Raises:
            ValueError: 如果是单次消息
        """
        ...
```

### 使用示例

#### 1. 返回流式响应（基于 channel）

```python
class LLMWorkerActor(Actor):
    async def receive(self, msg: Message) -> Message:
        if msg.msg_type == "Generate":
            req = msg.to_json()
            
            # 创建 channel-based 流式响应
            stream_msg, writer = StreamMessage.from_channel("TokenStream")
            
            # 启动生产者 task（offload 到 Python 线程）
            async def produce():
                try:
                    async for token in self.backend.generate(req["prompt"]):
                        await writer.write_json({"text": token.text})
                    writer.close()
                except Exception as e:
                    writer.error(str(e))
            
            asyncio.create_task(produce())
            
            # 立即返回，不阻塞
            return stream_msg
        
        return Message.empty()
```

#### 2. 返回流式响应（基于 async generator）

```python
class LLMWorkerActor(Actor):
    async def receive(self, msg: Message) -> Message:
        if msg.msg_type == "Generate":
            req = msg.to_json()
            
            # 定义 async generator
            async def token_generator():
                async for token in self.backend.generate(req["prompt"]):
                    yield json.dumps({"text": token.text}).encode()
            
            # 直接包装为流式消息
            return StreamMessage.from_generator("TokenStream", token_generator())
        
        return Message.empty()
```

#### 3. 消费流式请求

```python
class AggregatorActor(Actor):
    async def receive(self, msg: Message) -> Message:
        if msg.is_stream:
            # 获取流读取器
            reader = msg.stream_reader()
            
            # 在 Python task 中消费流
            results = []
            async for chunk in reader:
                data = json.loads(chunk)
                results.append(data)
            
            # 返回聚合结果
            return Message.from_json("AggregateResult", {"items": results})
        
        return Message.empty()
```

#### 4. 流式请求 + 流式响应（双向流）

```python
class StreamProcessorActor(Actor):
    async def receive(self, msg: Message) -> Message:
        if msg.is_stream:
            reader = msg.stream_reader()
            stream_msg, writer = StreamMessage.from_channel("ProcessedStream")
            
            # 处理 task：读取输入流，处理后写入输出流
            async def process():
                try:
                    async for chunk in reader:
                        processed = self.process_chunk(chunk)
                        await writer.write(processed)
                    writer.close()
                except Exception as e:
                    writer.error(str(e))
            
            asyncio.create_task(process())
            return stream_msg
        
        return Message.empty()
```

### Rust 实现要点

```rust
// ============================================================================
// PyStreamReader - 入站流的 Python 包装
// ============================================================================

#[pyclass]
pub struct PyStreamReader {
    receiver: Arc<Mutex<Option<PayloadStream>>>,
    event_loop: PyObject,
}

#[pymethods]
impl PyStreamReader {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let receiver = self.receiver.clone();
        let event_loop = self.event_loop.clone();
        
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = receiver.lock().await;
            if let Some(stream) = guard.as_mut() {
                match stream.next().await {
                    Some(Ok(data)) => {
                        Python::with_gil(|py| {
                            Ok(PyBytes::new(py, &data).into())
                        })
                    }
                    Some(Err(e)) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        e.to_string()
                    )),
                    None => Err(PyErr::new::<pyo3::exceptions::PyStopAsyncIteration, _>(
                        "Stream ended"
                    )),
                }
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyStopAsyncIteration, _>(
                    "Stream consumed"
                ))
            }
        })
    }
    
    fn cancel(&self) {
        // 通过 drop receiver 来取消流
        let receiver = self.receiver.clone();
        tokio::spawn(async move {
            let mut guard = receiver.lock().await;
            *guard = None;
        });
    }
}

// ============================================================================
// PyStreamMessage - 出站流的 Python 包装
// ============================================================================

#[pyclass]
pub struct PyStreamMessage {
    msg_type: String,
    receiver: Option<mpsc::Receiver<anyhow::Result<Vec<u8>>>>,
}

#[pyclass]
pub struct PyStreamWriter {
    sender: Option<mpsc::Sender<anyhow::Result<Vec<u8>>>>,
}

#[pymethods]
impl PyStreamWriter {
    fn write<'py>(&self, py: Python<'py>, data: Vec<u8>) -> PyResult<Bound<'py, PyAny>> {
        let sender = self.sender.clone();
        
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            if let Some(tx) = sender {
                tx.send(Ok(data)).await
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "Stream closed"
                    ))?;
            }
            Ok(())
        })
    }
    
    fn close(&mut self) {
        self.sender = None;  // Drop sender to signal end
    }
    
    fn error(&mut self, msg: String) {
        if let Some(tx) = self.sender.take() {
            let _ = tx.try_send(Err(anyhow::anyhow!(msg)));
        }
    }
}

// ============================================================================
// PythonActorWrapper 扩展
// ============================================================================

#[async_trait]
impl Actor for PythonActorWrapper {
    async fn receive(&mut self, msg: Message, ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let handler = self.handler.clone();
        let event_loop = self.event_loop.clone();
        
        // 将 Message 转换为 Python 可用的类型
        let py_msg = match msg {
            Message::Single { msg_type, data } => {
                PyMessage::Single { msg_type, payload: data }
            }
            Message::Stream { msg_type, stream } => {
                // 包装为 PyStreamReader
                let reader = PyStreamReader::new(stream, event_loop.clone());
                PyMessage::Stream { msg_type, reader }
            }
        };
        
        // 调用 Python receive
        let response = python_executor()
            .execute(move || {
                Python::with_gil(|py| -> PyResult<PyMessageResponse> {
                    let result = handler.call_method1(py, "receive", (py_msg,))?;
                    
                    // 处理 coroutine
                    let asyncio = py.import("asyncio")?;
                    let is_coro = asyncio.call_method1("iscoroutine", (&result,))?
                        .extract::<bool>()?;
                    
                    let py_result = if is_coro {
                        let future = asyncio.call_method1(
                            "run_coroutine_threadsafe", 
                            (&result, &event_loop)
                        )?;
                        future.call_method0("result")?
                    } else {
                        result
                    };
                    
                    // 检查返回类型
                    if py_result.is_instance_of::<PyStreamMessage>(py) {
                        let stream_msg: PyStreamMessage = py_result.extract(py)?;
                        Ok(PyMessageResponse::Stream(stream_msg))
                    } else {
                        let msg: PyMessage = py_result.extract(py)?;
                        Ok(PyMessageResponse::Single(msg))
                    }
                })
            })
            .await??;
        
        // 转换回 Rust Message
        match response {
            PyMessageResponse::Single(msg) => Ok(msg.to_message()),
            PyMessageResponse::Stream(stream_msg) => {
                Ok(Message::from_channel(
                    stream_msg.msg_type,
                    stream_msg.receiver.unwrap()
                ))
            }
        }
    }
}
```

### 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| **入站流暴露方式** | AsyncIterator | 符合 Python 惯用法，易于在 async for 中使用 |
| **出站流创建方式** | Channel + async generator | 灵活，支持 push 和 pull 两种模式 |
| **任务执行位置** | Python asyncio | 让 Python 代码在熟悉的环境中执行，便于调试 |
| **背压机制** | Channel 缓冲区 | 自然的背压，写入时会等待消费者 |
| **取消机制** | Drop receiver/sender | 惯用的 Rust 模式，自动传播取消 |

### 生命周期与资源管理

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     流式响应生命周期                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. receive() 被调用                                                        │
│      │                                                                      │
│  2. Python 创建 StreamMessage + StreamWriter                                │
│      │                                                                      │
│  3. Python 启动 asyncio.create_task(produce())                              │
│      │                                                                      │
│  4. receive() 返回 StreamMessage                                            │
│      │                                                                      │
│  5. Rust 取出 receiver，包装为 Message::Stream                              │
│      │                                                                      │
│  6. Rust 开始消费 stream (传输到网络)                                       │
│      │                           │                                          │
│  7. Python task 写入 writer ────►│ (channel buffer)                         │
│      │                           │                                          │
│  8. Rust 读取 stream ◄───────────┘                                          │
│      │                                                                      │
│  9. Python task 调用 writer.close()                                         │
│      │                                                                      │
│  10. Rust 读到 None，流结束                                                 │
│                                                                             │
│  取消场景:                                                                  │
│  • 客户端断开 → Rust drop receiver → Python write 失败 → task 退出         │
│  • Python 出错 → writer.error() → Rust 收到 Err → 传播到客户端             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 元数据同步机制

### Gossip 协议扩展

元数据通过 Gossip 协议在集群中同步，采用最终一致性模型。

```rust
pub enum GossipMessage {
    // 现有消息
    Ping { ... },
    Pong { ... },
    Sync { ... },
    NamedActorRegistered { ... },
    NamedActorUnregistered { ... },
    
    // 元数据同步
    MetadataSync {
        entries: Vec<MetadataEntry>,
    },
    MetadataUpdate {
        entry: MetadataEntry,
    },
}

pub struct MetadataEntry {
    pub namespace: String,
    pub key: String,
    pub value: Vec<u8>,
    pub version: u64,           // Lamport 时间戳
    pub expires_at: Option<Instant>,
    pub origin: NodeId,
}
```

### 冲突解决

采用 Last-Write-Wins (LWW) 策略，基于 Lamport 时间戳：

```rust
impl MetadataStore {
    fn merge(&mut self, entry: MetadataEntry) {
        let key = format!("{}/{}", entry.namespace, entry.key);
        
        match self.entries.get(&key) {
            Some(existing) if existing.version >= entry.version => {
                // 本地版本更新，忽略
            }
            _ => {
                // 远程版本更新，采纳
                self.entries.insert(key, entry);
                self.notify_watchers(&entry);
            }
        }
    }
}
```

### 元数据使用场景

1. **Worker 注册**：Worker 启动时将自己的信息写入元数据
2. **模型信息**：存储模型部署卡片、配置等
3. **路由策略**：Router 从元数据获取可用 Worker 列表
4. **动态配置**：运行时配置变更

## 集群部署示例

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           集群部署示例                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Node A (Frontend 节点)           Node B (Router 节点)                      │
│  ┌──────────────────────┐        ┌──────────────────────┐                  │
│  │ ActorSystem          │        │ ActorSystem          │                  │
│  │ 192.168.1.10:8000    │        │ 192.168.1.11:8000    │                  │
│  │                      │        │                      │                  │
│  │ [SystemActor]◄──────────Gossip──────►[SystemActor]   │                  │
│  │      │               │        │      │               │                  │
│  │ [FrontendActor]      │        │ [RouterActor]        │                  │
│  │ services/http/api    │        │ services/llm/router  │                  │
│  └──────────────────────┘        └──────────────────────┘                  │
│           │                               │                                 │
│           │  send(Generate) → Stream      │                                 │
│           └──────────────────────────────►│                                 │
│                                           │                                 │
│                                           │  send(Generate) → Stream        │
│                                           ▼                                 │
│  Node C (Worker 节点)            Node D (Worker 节点)                       │
│  ┌──────────────────────┐        ┌──────────────────────┐                  │
│  │ ActorSystem          │        │ ActorSystem          │                  │
│  │ 192.168.1.12:8000    │        │ 192.168.1.13:8000    │                  │
│  │                      │        │                      │                  │
│  │ [SystemActor]◄──────────Gossip──────►[SystemActor]   │                  │
│  │      │               │        │      │               │                  │
│  │ [WorkerActor]        │        │ [WorkerActor]        │                  │
│  │ workers/llm/vllm-0   │        │ workers/llm/vllm-1   │                  │
│  └──────────────────────┘        └──────────────────────┘                  │
│                                                                             │
│  Gossip 同步内容:                                                           │
│  • 成员列表: [node_a, node_b, node_c, node_d]                              │
│  • Named Actor 注册表:                                                      │
│    - services/http/api   → [node_a]                                        │
│    - services/llm/router → [node_b]                                        │
│    - workers/llm/vllm-0  → [node_c]                                        │
│    - workers/llm/vllm-1  → [node_d]                                        │
│  • 元数据:                                                                  │
│    - workers/llama-70b/node_c → {capacity, load, ...}                      │
│    - workers/llama-70b/node_d → {capacity, load, ...}                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## HTTP API 设计

### 路由规则

```
SystemActor 专用路由 (管理/诊断):
─────────────────────────────────────────────────────────────────────────
GET  /system/health              → HealthCheck → HealthStatus
GET  /system/status              → GetStatus → SystemStatus
GET  /system/members             → GetMembers → MemberList
GET  /system/actors              → ListLocalActors → LocalActorList

POST   /system/metadata/{ns}/{key}   → SetMetadata
GET    /system/metadata/{ns}/{key}   → GetMetadata
DELETE /system/metadata/{ns}/{key}   → DeleteMetadata
GET    /system/metadata/{ns}         → ListMetadata (prefix)

业务 Actor 路由:
─────────────────────────────────────────────────────────────────────────
POST /actors/{actor_name}        → Actor 消息
POST /named/{path...}            → Named Actor 消息

Headers:
  X-Message-Mode: ask | tell        // ask 等待响应，tell 不等待
  X-Message-Type: {message_type}    // 消息类型标识
  
响应可能是:
  - Single: 普通响应体
  - Stream: chunked/ndjson 流式响应

Gossip 内部协议:
─────────────────────────────────────────────────────────────────────────
POST /internal/gossip            → Gossip 协议消息
```

### 流式响应格式

请求始终使用普通 POST，响应类型由 Actor 的 `receive()` 返回值决定：

```
POST /named/services/llm/router
X-Message-Mode: ask
X-Message-Type: Generate
Content-Type: application/octet-stream

[bincode serialized Generate request]

---

HTTP/2 200 OK
X-Response-Type: stream                      // 标识响应为流式
Transfer-Encoding: chunked
Content-Type: application/x-ndjson

{"seq":0,"data":"base64(StreamToken)"}       // 序列化的 token
{"seq":1,"data":"base64(StreamToken)"}
{"seq":2,"data":"base64(StreamToken)"}
{"seq":3,"data":"","end":true}               // 流结束标记
```

**单次响应格式对比**：

```
POST /named/services/llm/router
X-Message-Mode: ask
X-Message-Type: HealthCheck

---

HTTP/2 200 OK
X-Response-Type: single                      // 标识响应为单次
Content-Type: application/octet-stream

[bincode serialized HealthStatus response]
```

## 使用示例

### 启动 ActorSystem

```rust
use pulsing_actor::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // 创建 ActorSystem（自动创建 SystemActor）
    let system = ActorSystem::new(
        SystemConfig::new()
            .bind("0.0.0.0:8000")
            .seeds(vec!["seed1:8000", "seed2:8000"])
    ).await?;
    
    // 创建业务 Actor
    let router_path = ActorPath::new("services/llm/router")?;
    system.spawn_named(router_path, RouterActor::new()).await?;
    
    system.wait_shutdown().await
}
```

### 与 SystemActor 交互

```rust
// 获取 SystemActor 引用
let sys = system.system_ref();

// 查询集群成员 (Single → Single)
let members: MemberList = sys.ask(GetMembers).await?;

// 设置元数据 (fire-and-forget)
sys.tell(SetMetadata {
    namespace: "models".into(),
    key: "llama-70b".into(),
    value: serde_json::to_vec(&model_info)?,
    ttl: None,
}).await?;

// 监听元数据变化 (Single → Stream)
let request = Message::pack(&WatchMetadata {
    namespace: "workers".into(),
    key_prefix: "".into(),
})?;
let response = sys.send(request).await?;

// 响应是 Stream，迭代处理
if let Message::Stream { mut stream, .. } = response {
    while let Some(Ok(data)) = stream.next().await {
        let change: MetadataChanged = bincode::deserialize(&data)?;
        println!("Metadata changed: {:?}", change);
    }
}
```

### 实现 Worker Actor

```rust
pub struct WorkerActor {
    backend: Box<dyn LLMBackend>,
    model_name: String,
}

#[async_trait]
impl Actor for WorkerActor {
    async fn on_start(&mut self, ctx: &mut ActorContext) -> Result<()> {
        let sys = ctx.system_ref();
        
        // 注册到元数据
        sys.tell(SetMetadata {
            namespace: "workers".into(),
            key: format!("{}/{}", self.model_name, ctx.node_id()),
            value: serde_json::to_vec(&WorkerInfo {
                addr: ctx.self_address(),
                model: self.model_name.clone(),
                capacity: self.backend.capacity(),
            })?,
            ttl: Some(Duration::from_secs(30)),
        }).await?;
        
        // 启动心跳
        ctx.spawn_interval(Duration::from_secs(10), || HeartbeatTick);
        
        Ok(())
    }
    
    async fn receive(&mut self, msg: Message, ctx: &mut ActorContext) -> Result<Message> {
        // 检查消息类型
        if msg.msg_type().ends_with("Generate") {
            let req: Generate = msg.unpack()?;
            let cancel = ctx.cancellation_token();
            
            // 返回流式响应
            let stream = self.backend
                .generate_stream(req.prompt, req.params)
                .take_until(cancel.cancelled())
                .map(|token| {
                    bincode::serialize(&StreamToken {
                        text: token.text,
                        logprob: token.logprob,
                    })
                    .map_err(|e| anyhow::anyhow!("{}", e))
                });
            
            return Ok(Message::stream("StreamToken", stream));
        }
        
        Err(anyhow::anyhow!("Unknown message type: {}", msg.msg_type()))
    }
}
```

### 实现 Router Actor

```rust
pub struct RouterActor {
    workers: Vec<WorkerInfo>,
    strategy: LoadBalanceStrategy,
}

#[async_trait]
impl Actor for RouterActor {
    async fn on_start(&mut self, ctx: &mut ActorContext) -> Result<()> {
        let sys = ctx.system_ref();
        
        // 查询现有 Worker (Single → Single)
        let workers: MetadataList = sys.ask(ListMetadata {
            namespace: "workers".into(),
            key_prefix: "".into(),
        }).await?;
        
        for entry in workers.entries {
            let info: WorkerInfo = serde_json::from_slice(&entry.value)?;
            self.workers.push(info);
        }
        
        // 监听 Worker 变化 (Single → Stream)
        let request = Message::pack(&WatchMetadata {
            namespace: "workers".into(),
            key_prefix: "".into(),
        })?;
        let response = sys.send(request).await?;
        
        if let Message::Stream { stream, .. } = response {
            ctx.spawn_task(async move {
                let mut stream = stream;
                while let Some(Ok(data)) = stream.next().await {
                    let _change: MetadataChanged = bincode::deserialize(&data)?;
                    // 更新 worker 列表
                }
                Ok::<_, anyhow::Error>(())
            });
        }
        
        Ok(())
    }
    
    async fn receive(&mut self, msg: Message, ctx: &mut ActorContext) -> Result<Message> {
        if msg.msg_type().ends_with("Generate") {
            let req: GenerateRequest = msg.unpack()?;
            
            // 选择 Worker
            let worker_info = self.select_worker(&req)?;
            let worker = ctx.resolve(&worker_info.addr).await?;
            
            // 转发请求到 Worker，直接返回 Worker 的响应（可能是 Stream）
            let request = Message::pack(&Generate {
                prompt: req.prompt,
                params: req.params,
            })?;
            
            return worker.send(request).await;
        }
        
        Err(anyhow::anyhow!("Unknown message type: {}", msg.msg_type()))
    }
}
```

## 项目结构

```
pulsing/
├── actor_system/                    # Actor 核心框架
│   └── src/
│       ├── lib.rs
│       ├── actor/
│       │   ├── mod.rs
│       │   ├── traits.rs            # Actor, Message 枚举
│       │   ├── context.rs           # ActorContext
│       │   ├── reference.rs         # ActorRef (send/fire/ask/tell)
│       │   ├── mailbox.rs
│       │   └── address.rs           # ActorAddress, ActorPath
│       │
│       ├── system/
│       │   ├── mod.rs
│       │   ├── actor_system.rs      # ActorSystem
│       │   ├── system_actor.rs      # SystemActor 实现
│       │   └── messages.rs          # 系统消息定义
│       │
│       ├── cluster/
│       │   ├── mod.rs
│       │   ├── gossip.rs            # Gossip 协议
│       │   ├── swim.rs              # SWIM 故障检测
│       │   ├── member.rs            # 成员管理
│       │   └── metadata.rs          # 元数据存储
│       │
│       └── transport/
│           ├── mod.rs
│           ├── http.rs              # HTTP/2 传输
│           ├── codec.rs             # 消息编解码
│           └── stream.rs            # 流式传输支持
│
├── actors/                          # 可复用的业务 Actor
│   └── src/
│       ├── lib.rs
│       ├── frontend.rs              # HTTP Frontend Actor
│       ├── router.rs                # LLM Router Actor
│       └── worker.rs                # LLM Worker Actor
│
└── examples/
    ├── standalone/                  # 单节点示例
    ├── cluster/                     # 集群示例
    └── llm_service/                 # LLM 服务完整示例
```

## 后续工作

### 传输层增强

当前已支持 HTTP/2 (h2c)，后续可增强：
- TLS 支持 (h2)
- 更精细的流控和背压
- 连接池优化

### 可靠性增强

- 消息重试机制
- 死信队列
- 熔断器

### 可观测性

- 分布式追踪集成 (OpenTelemetry)
- 指标暴露 (Prometheus)
- 日志关联

### 安全性

- TLS 支持
- 认证/授权

## 总结

| 组件 | 职责 | 创建方式 |
|------|------|---------|
| **ActorSystem** | 进程入口，管理传输层 | 用户显式创建 |
| **SystemActor** | 集群管理、元数据、诊断 | 自动创建，每进程一个 |
| **业务 Actor** | 具体业务逻辑 | 用户按需创建 |

**核心设计原则**：
1. **Everything is Actor** - 所有组件都是 Actor
2. **零外部依赖** - Gossip 替代 etcd/NATS
3. **统一 Message 类型** - `Message::Single` 和 `Message::Stream` 统一处理，无需区分 `ask`/`ask_stream`
4. **简洁 API** - 通过 SystemActor 统一系统操作入口

**消息模式**：
| 模式 | 说明 | 使用场景 |
|------|------|---------|
| Single → Single | 传统 RPC | 查询、命令 |
| Single → Stream | 服务端流式 | LLM 推理、Watch |
| Stream → Single | 客户端流式 | 文件上传 |
| Stream → Stream | 双向流式 | 实时通信 |

