# Pulsing Actor System

一个轻量级、零外部依赖的分布式 Actor 框架。

## 特性

- **零外部依赖**：不依赖 etcd、nats 或 redis
- **Gossip 服务发现**：使用 SWIM 协议自动管理集群成员
- **位置透明的 ActorRef**：本地和远程 Actor 使用相同的 API
- **原生 Async/Await**：基于 tokio 构建

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         ActorSystem                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Actor 1    │  │  Actor 2    │  │      Cluster Module      │  │
│  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────────────────┐  │  │
│  │  │Mailbox│  │  │  │Mailbox│  │  │  │  Gossip Protocol  │  │  │
│  │  └───────┘  │  │  └───────┘  │  │  │  (SWIM-like)      │  │  │
│  └─────────────┘  └─────────────┘  │  └───────────────────┘  │  │
│         ↑               ↑          │           ↑              │  │
│         └───────┬───────┘          │           │              │  │
│                 │                  │           │              │  │
│        ┌────────┴────────┐         │  ┌────────┴────────┐    │  │
│        │  Actor Registry │←────────┼──│  Member Registry │    │  │
│        └─────────────────┘         │  └─────────────────┘    │  │
│                                    └─────────────────────────┘  │
│                          ↕ TCP Transport                         │
└─────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 定义消息

```rust
use pulsing_actor::prelude::*;

#[derive(Serialize, Deserialize)]
struct Ping { value: i32 }

impl Message for Ping {
    fn type_id() -> &'static str { "Ping" }
}

#[derive(Serialize, Deserialize)]
struct Pong { result: i32 }

impl Message for Pong {
    fn type_id() -> &'static str { "Pong" }
}
```

### 定义 Actor

```rust
struct CounterActor {
    id: ActorId,
    count: i32,
}

#[async_trait]
impl Actor for CounterActor {
    fn id(&self) -> &ActorId { &self.id }

    async fn receive(
        &mut self,
        msg: RawMessage,
        _ctx: &mut ActorContext,
    ) -> anyhow::Result<RawMessage> {
        match msg.msg_type.as_str() {
            "Ping" => {
                let ping: Ping = msg.into_message()?;
                self.count += ping.value;
                RawMessage::from_message(&Pong { result: self.count })
            }
            _ => Err(anyhow::anyhow!("Unknown message"))
        }
    }
}
```

### 创建 Actor System

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 单机模式
    let system = ActorSystem::new(SystemConfig::standalone()).await?;

    // 创建 Actor
    let actor = CounterActor {
        id: ActorId::local("counter"),
        count: 0,
    };
    let actor_ref = system.spawn(actor).await?;

    // 发送消息并等待响应 (ask pattern)
    let pong: Pong = actor_ref.ask(Ping { value: 42 }).await?;
    println!("Result: {}", pong.result);

    // 发送消息不等待响应 (tell pattern)
    actor_ref.tell(Ping { value: 10 }).await?;

    system.shutdown().await?;
    Ok(())
}
```

## 集群模式

```rust
// 节点 1 - 启动第一个节点
let config = SystemConfig::with_addrs(
    "0.0.0.0:8000".parse()?,  // TCP 地址
    "0.0.0.0:7000".parse()?,  // Gossip 地址
);
let system1 = ActorSystem::new(config).await?;

// 节点 2 - 加入现有集群
let config = SystemConfig::with_addrs(
    "0.0.0.0:8001".parse()?,
    "0.0.0.0:7001".parse()?,
).with_seeds(vec!["192.168.1.100:7000".parse()?]);

let system2 = ActorSystem::new(config).await?;

// 获取远程 Actor 的引用
let remote_ref = system2.actor_ref(&actor_id).await?;
let result: Pong = remote_ref.ask(Ping { value: 10 }).await?;
```

## 核心组件

### Actor Trait

```rust
#[async_trait]
pub trait Actor: Send + Sync + 'static {
    /// Actor 唯一标识
    fn id(&self) -> &ActorId;

    /// 启动回调
    async fn on_start(&mut self, ctx: &mut ActorContext) -> Result<()>;

    /// 停止回调
    async fn on_stop(&mut self, ctx: &mut ActorContext) -> Result<()>;

    /// 处理消息
    async fn receive(&mut self, msg: RawMessage, ctx: &mut ActorContext) -> Result<RawMessage>;
}
```

### Message Trait

所有消息必须实现 `Message` trait，要求可序列化：

```rust
pub trait Message: Serialize + DeserializeOwned + Send + Sync + 'static {
    fn type_id() -> &'static str where Self: Sized;
}
```

### ActorRef

ActorRef 是 Actor 的句柄，用于发送消息：

```rust
// ask - 发送并等待响应
let response: R = actor_ref.ask(message).await?;

// tell - 发送不等待响应
actor_ref.tell(message).await?;
```

## 运行示例

```bash
# Ping-Pong 示例
cargo run --example ping_pong -p pulsing-actor

# 集群示例 (需要两个终端)
# 终端 1:
cargo run --example cluster -p pulsing-actor -- --node 1

# 终端 2:
cargo run --example cluster -p pulsing-actor -- --node 2
```

## 配置

### SystemConfig

```rust
pub struct SystemConfig {
    /// TCP 地址 (用于 Actor 通信)
    pub tcp_addr: SocketAddr,

    /// Gossip 地址 (用于集群成员管理)
    pub gossip_addr: SocketAddr,

    /// 种子节点 (加入集群时使用)
    pub seed_nodes: Vec<SocketAddr>,

    /// Gossip 配置
    pub gossip_config: GossipConfig,

    /// TCP 传输配置
    pub tcp_config: TcpTransportConfig,
}
```

### GossipConfig

```rust
pub struct GossipConfig {
    /// Gossip 间隔 (默认 200ms)
    pub gossip_interval: Duration,

    /// 每轮 Gossip 的目标节点数 (默认 3)
    pub fanout: usize,

    /// SWIM 故障检测配置
    pub swim: SwimConfig,
}
```

## 设计原则

1. **简单性**：API 简洁，易于理解
2. **零依赖**：不依赖外部服务
3. **最终一致性**：使用 Gossip 协议，适合大多数场景
4. **故障检测**：基于 SWIM 协议的故障检测
5. **位置透明**：本地和远程 Actor 使用相同的 API

## License

Apache-2.0
