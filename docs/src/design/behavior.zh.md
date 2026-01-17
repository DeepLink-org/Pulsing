# Behavior API (Rust)

Pulsing 提供类型安全的函数式 Actor 编程接口，灵感来自 Akka Typed。Behavior API 提供了传统 `Actor` trait 之外的另一种选择，具备编译时消息类型检查。

## 设计理念

**类型安全优先。**

- `TypedRef<M>` 确保消息在编译时进行类型检查
- 通过 `BehaviorAction::Become` 进行状态转换，实现清晰的状态机模式
- 函数式风格：Actor 就是消息处理函数

## 核心概念

### Behavior&lt;M&gt;

Actor 定义为处理类型 `M` 消息的函数：

```rust
use pulsing_actor::behavior::{stateful, Behavior, BehaviorAction};

fn counter(initial: i32) -> Behavior<i32> {
    stateful(initial, |count, n, _ctx| {
        *count += n;
        println!("count = {}", *count);
        BehaviorAction::Same
    })
}
```

### TypedRef&lt;M&gt;

类型安全的 Actor 引用，只接受类型 `M` 的消息：

```rust
// 类型安全：只能发送 i32
let counter: TypedRef<i32> = system.spawn_behavior("counter", counter(0)).await?;

counter.tell(5).await?;  // OK
counter.tell(3).await?;  // OK
// counter.tell("hello").await?;  // 编译错误！
```

### BehaviorAction

控制 Actor 生命周期和状态转换：

| Action | 说明 |
|--------|------|
| `Same` | 保持当前 behavior |
| `Become(behavior)` | 切换到新的 behavior（状态机转换） |
| `Stop(reason)` | 优雅停止 Actor |

## 创建 Behavior

### 有状态 Behavior

使用 `stateful()` 创建有内部状态的 Actor：

```rust
use pulsing_actor::behavior::{stateful, Behavior, BehaviorAction};

fn counter(initial: i32) -> Behavior<i32> {
    stateful(initial, |count, msg, ctx| {
        *count += msg;
        println!("[{}] count = {}", ctx.name(), *count);
        BehaviorAction::Same
    })
}
```

处理函数接收：
- `&mut S` - 状态的可变引用
- `M` - 消息
- `&BehaviorContext<M>` - Actor 上下文

### 无状态 Behavior

使用 `stateless()` 创建无内部状态的 Actor：

```rust
use pulsing_actor::behavior::{stateless, Behavior, BehaviorAction};

fn echo() -> Behavior<String> {
    stateless(|msg, ctx| {
        Box::pin(async move {
            println!("[{}] Received: {}", ctx.name(), msg);
            BehaviorAction::Same
        })
    })
}
```

## 状态机模式

使用 `BehaviorAction::Become` 实现状态机：

```rust
use pulsing_actor::behavior::{stateful, Behavior, BehaviorAction};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
enum Signal {
    Next,
    Query,
}

// 带共享数据的状态
#[derive(Clone)]
struct Stats {
    cycles: u32,
    transitions: u32,
}

fn red(stats: Stats) -> Behavior<Signal> {
    stateful(stats, |stats, msg, ctx| match msg {
        Signal::Next => {
            stats.transitions += 1;
            println!("[{}] 🔴 红灯 -> 🟢 绿灯", ctx.name());
            BehaviorAction::Become(green(stats.clone()))
        }
        Signal::Query => {
            println!("[{}] 当前: 🔴 红灯", ctx.name());
            BehaviorAction::Same
        }
    })
}

fn green(stats: Stats) -> Behavior<Signal> {
    stateful(stats, |stats, msg, ctx| match msg {
        Signal::Next => {
            stats.transitions += 1;
            println!("[{}] 🟢 绿灯 -> 🟡 黄灯", ctx.name());
            BehaviorAction::Become(yellow(stats.clone()))
        }
        Signal::Query => {
            println!("[{}] 当前: 🟢 绿灯", ctx.name());
            BehaviorAction::Same
        }
    })
}

fn yellow(stats: Stats) -> Behavior<Signal> {
    stateful(stats, |stats, msg, ctx| match msg {
        Signal::Next => {
            stats.transitions += 1;
            stats.cycles += 1;
            println!("[{}] 🟡 黄灯 -> 🔴 红灯 (周期 #{})", ctx.name(), stats.cycles);
            BehaviorAction::Become(red(stats.clone()))
        }
        Signal::Query => {
            println!("[{}] 当前: 🟡 黄灯", ctx.name());
            BehaviorAction::Same
        }
    })
}
```

## BehaviorContext

上下文提供：

```rust
// 获取 Actor 名称
let name = ctx.name();

// 获取类型安全的自引用（用于 reply-to 模式）
let self_ref: TypedRef<M> = ctx.self_ref();

// 获取其他 Actor 的类型引用
let other: TypedRef<OtherMsg> = ctx.typed_ref("other_actor");

// 调度延迟消息给自己
ctx.schedule_self(msg, Duration::from_secs(5));

// 检查 Actor 是否应该停止
if ctx.is_cancelled() { ... }

// 访问底层 ActorSystem
let system = ctx.system();
```

## TypedRef 操作

```rust
// 发送不等待响应
counter.tell(5).await?;

// 请求-响应
let result: i32 = counter.ask(CounterMsg::Get).await?;

// 带超时
let result: i32 = counter.ask_timeout(msg, Duration::from_secs(5)).await?;

// 检查 Actor 是否存活
if counter.is_alive() { ... }

// 获取底层无类型 ActorRef
let actor_ref = counter.as_untyped()?;
```

## 完整示例

```rust
use pulsing_actor::behavior::{stateful, Behavior, BehaviorAction, BehaviorSpawner};
use pulsing_actor::system::ActorSystem;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
enum CounterMsg {
    Increment(i32),
    Decrement(i32),
    Get,
}

fn counter(initial: i32) -> Behavior<CounterMsg> {
    stateful(initial, |count, msg, ctx| match msg {
        CounterMsg::Increment(n) => {
            *count += n;
            println!("[{}] +{} = {}", ctx.name(), n, *count);
            BehaviorAction::Same
        }
        CounterMsg::Decrement(n) => {
            *count -= n;
            println!("[{}] -{} = {}", ctx.name(), n, *count);
            BehaviorAction::Same
        }
        CounterMsg::Get => {
            println!("[{}] 当前 = {}", ctx.name(), *count);
            BehaviorAction::Same
        }
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let system = ActorSystem::builder().build().await?;

    // 启动 behavior 风格的 Actor
    let counter = system.spawn_behavior("counter", counter(0)).await?;

    // 类型安全的消息发送
    counter.tell(CounterMsg::Increment(5)).await?;
    counter.tell(CounterMsg::Increment(3)).await?;
    counter.tell(CounterMsg::Decrement(2)).await?;
    counter.tell(CounterMsg::Get).await?;

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    system.shutdown().await
}
```

## Actor Trait vs Behavior

| 特性 | Actor Trait | Behavior API |
|------|-------------|--------------|
| 类型安全 | 运行时 (Message 类型) | 编译时 (TypedRef&lt;M&gt;) |
| 状态 | 结构体字段 | 封装在闭包中 |
| 状态机 | 手动实现 | `BehaviorAction::Become` |
| 风格 | OOP (impl trait) | 函数式 (函数) |
| 灵活性 | 更高 | 结构化 |

**何时使用 Behavior：**

- 需要编译时消息类型检查
- 构建状态机
- 偏好函数式编程风格

**何时使用 Actor trait：**

- 需要最大灵活性
- 复杂的初始化逻辑
- 现有 OOP 代码库

## 运行示例

```bash
# 计数器示例
cargo run --example behavior_counter -p pulsing-actor

# 状态机示例
cargo run --example behavior_fsm -p pulsing-actor
```

## 下一步

- [Actor 系统](actor-system.zh.md) — 核心 Actor 基础设施
- [架构概览](architecture.zh.md) — 系统设计概述
