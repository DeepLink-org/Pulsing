# Rust API 参考

此页面提供 Pulsing Rust API 的概述，包括示例和使用模式。

## 安装

在 `Cargo.toml` 中添加 Pulsing：

```toml
[dependencies]
pulsing-actor = "0.1"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
```

## 核心概念

Rust API 通过 trait 层组织，提供不同级别的功能：

- **ActorSystemCoreExt**：主要 API，用于生成和解析 actor
- **ActorSystemAdvancedExt**：高级功能，如监督和基于工厂的生成
- **ActorSystemOpsExt**：运维、诊断和生命周期管理

## 快速开始

```rust
use pulsing_actor::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Ping(i32);

#[derive(Serialize, Deserialize)]
struct Pong(i32);

struct Echo;

#[async_trait::async_trait]
impl Actor for Echo {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let Ping(x) = msg.unpack()?;
        Message::pack(&Pong(x))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let system = ActorSystem::builder().build().await?;

    // 生成命名 actor
    let actor = system.spawn_named("services/echo", Echo).await?;

    // 发送消息并等待响应
    let Pong(x): Pong = actor.ask(Ping(42)).await?;

    println!("Received: {}", x);

    system.shutdown().await?;
    Ok(())
}
```

## 核心 API

### ActorSystem

Actor 系统的主要入口点。

```rust
pub struct ActorSystem { /* fields omitted */ }

impl ActorSystem {
    pub async fn builder() -> ActorSystemBuilder {
        // 创建新的 actor 系统构建器
    }
}
```

#### ActorSystemBuilder

构建器模式用于配置 actor 系统。

```rust
pub struct ActorSystemBuilder { /* fields omitted */ }

impl ActorSystemBuilder {
    pub fn addr<A: Into<String>>(self, addr: A) -> Self {
        // 设置绑定地址
    }

    pub fn seeds<I: IntoIterator<Item = String>>(self, seeds: I) -> Self {
        // 设置集群发现的种子节点
    }

    pub fn build(self) -> impl Future<Output = anyhow::Result<ActorSystem>> {
        // 构建 actor 系统
    }
}
```

### ActorSystemCoreExt

核心生成和解析功能。

```rust
#[async_trait::async_trait]
pub trait ActorSystemCoreExt {
    async fn spawn<A>(&self, actor: A) -> anyhow::Result<TypedRef<A::Message>>
    where
        A: Actor + 'static;

    async fn spawn_named<A>(
        &self,
        name: &str,
        actor: A
    ) -> anyhow::Result<TypedRef<A::Message>>
    where
        A: Actor + 'static;

    async fn actor_ref(&self, id: &ActorId) -> anyhow::Result<ActorRef>;

    async fn resolve(&self, name: &str) -> anyhow::Result<ActorRef>;
}
```

### Actor Trait

所有 actor 必须实现的 core trait。

```rust
#[async_trait::async_trait]
pub trait Actor: Send + 'static {
    type Message: Serialize + for<'de> Deserialize<'de> + Send + 'static;

    async fn receive(
        &mut self,
        msg: Message,
        ctx: &mut ActorContext
    ) -> anyhow::Result<Message>;

    fn on_start(&mut self, _id: ActorId, _ctx: &mut ActorContext) {}

    fn on_stop(&mut self, _ctx: &mut ActorContext) {}
}
```

### TypedRef

Actor 的类型安全引用。

```rust
pub struct TypedRef<M> { /* fields omitted */ }

impl<M> TypedRef<M>
where
    M: Serialize + for<'de> Deserialize<'de> + Send + 'static,
{
    pub async fn ask(&self, msg: M) -> anyhow::Result<M> {
        // 发送消息并等待类型化响应
    }

    pub async fn tell(&self, msg: M) -> anyhow::Result<()> {
        // 发送消息而不等待响应
    }
}
```

## 高级功能

### 监督

Actor 可以配置重启策略以实现容错。

```rust
use pulsing_actor::system::SupervisionSpec;

let options = SpawnOptions::default()
    .supervision(SupervisionSpec::on_failure().max_restarts(3));

// 基于工厂的生成，支持监督
system.spawn_named_factory("services/worker", || Ok(Worker::new()), options).await?;
```

### Behavior（类型安全 Actor）

使用行为模式的更高级别 API。

```rust
use pulsing_actor::prelude::*;

fn counter(init: i32) -> Behavior<i32> {
    stateful(init, |count, n, _ctx| {
        *count += n;
        BehaviorAction::Same
    })
}

// Behavior 实现 IntoActor trait
let counter = system.spawn(counter(0)).await?;
let result: i32 = counter.ask(5).await?; // Result is 5
```

### 消息类型

#### 常规消息

```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct MyMessage {
    action: String,
    data: Vec<i32>,
}

// 打包/解包消息
let msg = Message::pack(&MyMessage {
    action: "process".to_string(),
    data: vec![1, 2, 3],
})?;

let MyMessage { action, data } = msg.unpack()?;
```

#### 流式消息

```rust
// 流式响应
let stream_msg = Message::Stream(Stream::from_iter(items));

// 在 actor 中处理流式
async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
    match msg {
        Message::Stream(stream) => {
            // 处理流
            let result = process_stream(stream).await?;
            Message::pack(&result)
        }
        _ => Message::pack(&"Unsupported message type")
    }
}
```

## 集群管理

### 节点发现

Pulsing 使用 SWIM 协议进行自动集群发现。

```rust
// 单节点
let system = ActorSystem::builder()
    .addr("0.0.0.0:8000")
    .build()
    .await?;

// 加入现有集群
let system = ActorSystem::builder()
    .addr("0.0.0.0:8001")
    .seeds(vec!["127.0.0.1:8000".to_string()])
    .build()
    .await?;
```

### ActorSystemOpsExt

运维和诊断。

```rust
#[async_trait::async_trait]
pub trait ActorSystemOpsExt {
    fn node_id(&self) -> NodeId;

    fn addr(&self) -> &str;

    async fn members(&self) -> Vec<NodeInfo>;

    async fn all_named_actors(&self) -> HashMap<String, ActorId>;

    async fn stop(&self, name: &str) -> anyhow::Result<()>;

    async fn shutdown(self) -> anyhow::Result<()>;
}
```

## 错误处理

Pulsing 在整个 API 中使用 `anyhow::Result<T>` 进行错误处理。

```rust
use anyhow::{Result, Context};

async fn my_actor_logic(system: &ActorSystem) -> Result<()> {
    let actor = system.spawn_named("my_actor", MyActor)
        .await
        .context("Failed to spawn actor")?;

    let response = actor.ask(MyMessage::default())
        .await
        .context("Failed to send message")?;

    Ok(())
}
```

## 示例

### HTTP 服务器 Actor

```rust
use pulsing_actor::prelude::*;
use warp::Filter;

struct HttpServer {
    system: ActorSystem,
}

#[async_trait::async_trait]
impl Actor for HttpServer {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        // 通过转发到其他 actor 来处理 HTTP 请求
        let request: HttpRequest = msg.unpack()?;
        let processor = self.system.resolve("request_processor").await?;
        processor.ask(request).await
    }
}
```

### Worker 池

```rust
use pulsing_actor::prelude::*;

struct WorkerPool {
    workers: Vec<ActorRef>,
}

#[async_trait::async_trait]
impl Actor for WorkerPool {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        // 轮询任务分发
        let worker = &self.workers[self.next_worker()];
        worker.ask(msg).await
    }
}
```

## 性能考虑

- **零拷贝消息传递**：消息尽可能通过引用传递
- **异步运行时**：基于 Tokio 实现高并发
- **二进制序列化**：高效的 bincode 序列化
- **连接池化**：HTTP/2 连接重用

## 集成

### 与 Axum/Warp 集成

```rust
use axum::{routing::post, Router};
use pulsing_actor::prelude::*;

async fn handle_request(
    Extension(system): Extension<ActorSystem>,
    Json(payload): Json<MyRequest>,
) -> Json<MyResponse> {
    let actor = system.resolve("request_handler").await?;
    let response: MyResponse = actor.ask(payload).await?;
    Ok(Json(response))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let system = ActorSystem::builder().build().await?;

    let app = Router::new()
        .route("/api", post(handle_request))
        .layer(Extension(system.clone()));

    // 同时启动 HTTP 服务器和 actor 系统
    tokio::select! {
        _ = serve(app, ([127, 0, 0, 1], 3000)) => {},
        _ = system.run() => {},
    }

    Ok(())
}
```

## 后续步骤

- **[Python API](python.md)**: Python 接口文档
- **[示例](../../examples/)**: Rust 示例代码
- **[设计文档](../../design/)**: 架构和设计决策
