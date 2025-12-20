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
| 通信模式 | NATS/HTTP/TCP + 流式 | Actor 消息 (ask/tell/ask_stream) |
| 流式响应 | 外部 Stream 抽象 | Actor 原生支持 |

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
│  │   • ask()        → 单次请求/响应                                      │   │
│  │   • tell()       → 单向消息                                           │   │
│  │   • ask_stream() → 流式请求/响应                                      │   │
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

## 流式响应设计

### 设计理念

传统回调模式的问题：
- 每个 token 一次网络调用，开销大
- 需要临时 Actor 管理生命周期
- 背压和取消机制复杂

**解决方案**：在 Actor 层原生支持流式响应

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     流式响应：原生支持 vs 回调模式                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  回调模式 (复杂，已弃用):                                                    │
│  ─────────────────────────                                                  │
│  Client → Frontend → Router → Worker                                        │
│              │                    │                                         │
│              │  创建 StreamActor  │                                         │
│              │◄──────────────────│  tell(Token) × N                         │
│              ▼                                                              │
│         StreamActor (临时)                                                  │
│                                                                             │
│  原生流式 (简洁，采用):                                                     │
│  ───────────────────────                                                    │
│  Client → Frontend → Router → Worker                                        │
│              │                    │                                         │
│              │  ask_stream(req)   │                                         │
│              │═══════════════════►│                                         │
│              │◄═══ Stream<Token> ═│                                         │
│                                                                             │
│  优势:                                                                      │
│  • 无临时 Actor，无需管理生命周期                                           │
│  • 背压由 Stream 机制自然处理                                               │
│  • Drop stream = 取消请求                                                   │
│  • 代码量减少 90%                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ActorRef API 扩展

```rust
impl ActorRef {
    /// 请求-响应模式
    pub async fn ask<M, R>(&self, msg: M) -> Result<R>
    where
        M: Message,
        R: Message;
    
    /// 单向消息
    pub async fn tell<M>(&self, msg: M) -> Result<()>
    where
        M: Message;
    
    /// 流式请求-响应模式
    pub async fn ask_stream<M, R>(&self, msg: M) -> Result<MessageStream<R>>
    where
        M: Message,
        R: Message;
}
```

### MessageStream

```rust
/// 消息流，支持背压和取消
pub struct MessageStream<R> {
    inner: Pin<Box<dyn Stream<Item = Result<R>> + Send>>,
    cancel: CancellationToken,
}

impl<R> MessageStream<R> {
    /// 显式取消流
    pub fn cancel(&self) {
        self.cancel.cancel();
    }
}

impl<R> Stream for MessageStream<R> {
    type Item = Result<R>;
    
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) 
        -> Poll<Option<Self::Item>> 
    {
        self.inner.as_mut().poll_next(cx)
    }
}

impl<R> Drop for MessageStream<R> {
    fn drop(&mut self) {
        // Drop 时自动取消
        self.cancel.cancel();
    }
}
```

### Actor Trait 扩展

```rust
#[async_trait]
pub trait Actor: Send + 'static {
    fn id(&self) -> &ActorId;
    
    async fn on_start(&mut self, ctx: &mut ActorContext) -> Result<()> {
        Ok(())
    }
    
    async fn on_stop(&mut self, ctx: &mut ActorContext) -> Result<()> {
        Ok(())
    }
    
    /// 处理普通消息
    async fn receive(
        &mut self,
        msg: RawMessage,
        ctx: &mut ActorContext,
    ) -> Result<RawMessage>;
    
    /// 处理流式请求
    async fn receive_stream(
        &mut self,
        msg: RawMessage,
        ctx: &mut ActorContext,
    ) -> Result<RawMessageStream> {
        Err(anyhow::anyhow!("Streaming not supported"))
    }
}
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
│           │       ask_stream(Generate)    │                                 │
│           └──────────────────────────────►│                                 │
│                                           │                                 │
│                                           │ ask_stream(Generate)            │
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
POST /actors/{actor_name}        → 普通 Actor 消息
POST /named/{path...}            → Named Actor 消息

Headers:
  X-Message-Mode: ask | tell | stream
  X-Message-Type: {message_type}

Gossip 内部协议:
─────────────────────────────────────────────────────────────────────────
POST /internal/gossip            → Gossip 协议消息
```

### 流式响应格式

```
POST /named/services/llm/router
X-Message-Mode: stream
X-Message-Type: Generate
Content-Type: application/json

{"prompt": "Hello", "params": {...}}

---

HTTP/1.1 200 OK
Transfer-Encoding: chunked
Content-Type: application/x-ndjson

{"type":"StreamToken","text":"Hello"}
{"type":"StreamToken","text":" World"}
{"type":"StreamToken","text":"!"}
{"type":"StreamEnd","usage":{"prompt_tokens":1,"completion_tokens":3}}
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

// 查询集群成员
let members: MemberList = sys.ask(GetMembers).await?;

// 设置元数据
sys.tell(SetMetadata {
    namespace: "models".into(),
    key: "llama-70b".into(),
    value: serde_json::to_vec(&model_info)?,
    ttl: None,
}).await?;

// 监听元数据变化
let mut changes = sys.ask_stream::<_, MetadataChanged>(WatchMetadata {
    namespace: "workers".into(),
    key_prefix: "".into(),
}).await?;

while let Some(change) = changes.next().await {
    println!("Metadata changed: {:?}", change);
}
```

### 实现 Worker Actor

```rust
pub struct WorkerActor {
    id: ActorId,
    backend: Box<dyn LLMBackend>,
    model_name: String,
}

#[async_trait]
impl Actor for WorkerActor {
    fn id(&self) -> &ActorId { &self.id }
    
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
    
    async fn receive_stream(&mut self, msg: RawMessage, ctx: &mut ActorContext) 
        -> Result<RawMessageStream> 
    {
        match msg.msg_type.as_str() {
            "Generate" => {
                let req: Generate = msg.into_message()?;
                let cancel = ctx.cancellation_token();
                
                let stream = self.backend
                    .generate_stream(req.prompt, req.params)
                    .take_until(cancel.cancelled())
                    .map(|token| RawMessage::from_message(&StreamToken {
                        text: token.text,
                        logprob: token.logprob,
                    }));
                
                Ok(Box::pin(stream))
            }
            _ => Err(anyhow::anyhow!("Unknown")),
        }
    }
}
```

### 实现 Router Actor

```rust
pub struct RouterActor {
    id: ActorId,
    workers: Vec<WorkerInfo>,
    strategy: LoadBalanceStrategy,
}

#[async_trait]
impl Actor for RouterActor {
    fn id(&self) -> &ActorId { &self.id }
    
    async fn on_start(&mut self, ctx: &mut ActorContext) -> Result<()> {
        let sys = ctx.system_ref();
        
        // 查询现有 Worker
        let workers: MetadataList = sys.ask(ListMetadata {
            namespace: "workers".into(),
            key_prefix: "".into(),
        }).await?;
        
        for entry in workers.entries {
            let info: WorkerInfo = serde_json::from_slice(&entry.value)?;
            self.workers.push(info);
        }
        
        // 监听 Worker 变化
        let mut changes = sys.ask_stream::<_, MetadataChanged>(WatchMetadata {
            namespace: "workers".into(),
            key_prefix: "".into(),
        }).await?;
        
        ctx.spawn_task(async move {
            while let Some(Ok(change)) = changes.next().await {
                // 更新 worker 列表
            }
            Ok::<_, anyhow::Error>(())
        });
        
        Ok(())
    }
    
    async fn receive_stream(&mut self, msg: RawMessage, ctx: &mut ActorContext) 
        -> Result<RawMessageStream> 
    {
        match msg.msg_type.as_str() {
            "Generate" => {
                let req: GenerateRequest = msg.into_message()?;
                
                // 选择 Worker
                let worker_info = self.select_worker(&req)?;
                let worker = ctx.resolve(&worker_info.addr).await?;
                
                // 转发流式请求
                let stream = worker.ask_stream::<_, StreamToken>(Generate {
                    prompt: req.prompt,
                    params: req.params,
                }).await?;
                
                Ok(Box::pin(stream.map(|r| 
                    r.and_then(|t| RawMessage::from_message(&t))
                )))
            }
            _ => Err(anyhow::anyhow!("Unknown")),
        }
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
│       │   ├── traits.rs            # Actor, Message traits
│       │   ├── context.rs           # ActorContext
│       │   ├── reference.rs         # ActorRef (支持 ask_stream)
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

### 传输层重构

当前使用 HTTP/1.1，需要重构以支持：
- HTTP/2 多路复用
- 流式响应（chunked transfer / server-sent events）
- 背压机制
- 连接复用

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
1. Everything is Actor - 所有组件都是 Actor
2. 零外部依赖 - Gossip 替代 etcd/NATS
3. 原生流式支持 - ask_stream() 一等公民
4. 简洁 API - 通过 SystemActor 统一系统操作入口

