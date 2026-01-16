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
│                          ↕ HTTP/2 Transport                      │
└─────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 定义 Actor

```rust
use pulsing_actor::prelude::*;

struct Echo;

#[async_trait]
impl Actor for Echo {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let s: String = msg.unpack()?;
        Message::pack(&format!("echo: {}", s))
    }
}
```

### 创建 Actor System

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 单机模式
    let system = ActorSystem::builder().build().await?;

    // 创建 Actor
    let actor = system.spawn("echo", Echo).await?;

    // 发送消息并等待响应 (ask pattern)
    let resp: String = actor.ask("hello".to_string()).await?;
    println!("{}", resp);  // echo: hello

    system.shutdown().await
}
```

## 集群模式

```rust
// 节点 1 - 启动第一个节点
let system1 = ActorSystem::builder()
    .addr("0.0.0.0:8001")
    .build()
    .await?;

// 创建命名 Actor (可跨节点发现)
system1.spawn_named("services/echo", "echo", Echo).await?;

// 节点 2 - 加入现有集群
let system2 = ActorSystem::builder()
    .addr("0.0.0.0:8002")
    .seeds(&["127.0.0.1:8001"])
    .build()
    .await?;

// 通过路径解析远程 Actor
let remote = system2.resolve_named("services/echo", None).await?;
let resp: String = remote.ask("hello".to_string()).await?;
```

## Behavior API (类型安全)

除了传统 Actor trait，还提供函数式 Behavior API：

```rust
use pulsing_actor::behavior::{stateful, BehaviorAction, BehaviorSpawner};

fn counter(initial: i32) -> Behavior<i32> {
    stateful(initial, |count, n| {
        *count += n;
        println!("count = {}", *count);
        BehaviorAction::Same
    })
}

let system = ActorSystem::builder().build().await?;
let counter_ref = system.spawn_behavior("counter", counter(0)).await?;

counter_ref.tell(5).await?;  // count = 5
counter_ref.tell(3).await?;  // count = 8
```

## 核心组件

### Actor Trait

```rust
#[async_trait]
pub trait Actor: Send + 'static {
    async fn on_start(&mut self, ctx: &mut ActorContext) -> Result<()> { Ok(()) }
    async fn on_stop(&mut self, ctx: &mut ActorContext) -> Result<()> { Ok(()) }
    async fn receive(&mut self, msg: Message, ctx: &mut ActorContext) -> Result<Message>;
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

# Behavior 示例
cargo run --example behavior_counter -p pulsing-actor
```

## 配置 (Builder 模式)

```rust
let system = ActorSystem::builder()
    .addr("0.0.0.0:8001")           // 绑定地址
    .seeds(&["127.0.0.1:8000"])     // 种子节点
    .build()
    .await?;
```

## 设计原则

1. **简单性**：API 简洁，易于理解
2. **零依赖**：不依赖外部服务
3. **最终一致性**：使用 Gossip 协议，适合大多数场景
4. **故障检测**：基于 SWIM 协议的故障检测
5. **位置透明**：本地和远程 Actor 使用相同的 API

## License

Apache-2.0
