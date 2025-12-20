# Actor Addressing 设计文档

## 概述

本文档定义了 Pulsing Actor System 的地址设计方案。该方案提供统一的 URI 格式来标识和定位集群中的 Actor，支持具名 Actor 的多实例部署和自动负载均衡。

## 设计目标

1. **统一的地址格式** - 使用 URI scheme 提供自描述、可扩展的地址表示
2. **位置透明** - 具名 Actor 可在集群任意节点访问，无需知道具体位置
3. **多实例支持** - 同一具名 Actor 可部署多个实例，支持负载均衡
4. **高效的服务发现** - 只有具名 Actor 通过 Gossip 广播，减少网络开销
5. **命名空间隔离** - 强制命名空间，提供逻辑分组和管理

## 地址格式

### 地址类型概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Actor Address Scheme                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 具名 Actor 服务地址（位置透明，多实例）                                 │
│     actor:///{namespace}/{path}/{name}                                      │
│     例: actor:///services/llm/router                                        │
│                                                                             │
│  2. 具名 Actor 实例地址（指定节点实例）                                     │
│     actor:///{namespace}/{path}/{name}@{node_id}                            │
│     例: actor:///services/llm/router@node_a                                 │
│                                                                             │
│  3. 全局 Actor 地址（集群中任意 actor 的精确地址）                          │
│     actor://{node_id}/{actor_id}                                            │
│     例: actor://node_a/worker_123                                           │
│                                                                             │
│  4. 本地快捷引用（当前节点上的 actor）                                      │
│     actor://localhost/{actor_id}                                            │
│     例: actor://localhost/worker_123                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 地址格式详解

#### 1. 具名 Actor 服务地址

```
actor:///{namespace}/{path}/{name}
```

- **格式特征**：三斜杠开头（`actor:///`），表示省略 host 部分
- **命名空间**：必须，路径的第一段
- **路径**：可选的中间层级，用于逻辑分组
- **名称**：路径的最后一段

**示例**：
```
actor:///services/llm/router        # 命名空间: services, 名称: router
actor:///workers/inference/pool     # 命名空间: workers, 名称: pool
actor:///system/cluster/monitor     # 命名空间: system, 名称: monitor
```

#### 2. 具名 Actor 实例地址

```
actor:///{namespace}/{path}/{name}@{node_id}
```

- 在服务地址后添加 `@node_id` 后缀
- 直接路由到指定节点的实例

**示例**：
```
actor:///services/llm/router@node_a
actor:///workers/pool/manager@node_b
```

#### 3. 全局 Actor 地址

```
actor://{node_id}/{actor_id}
```

- **格式特征**：两斜杠后直接跟 node_id
- 集群中任意 Actor 的精确定位
- 不通过 Gossip 注册，需要预先知道地址

**示例**：
```
actor://node_a/worker_123
actor://node_b/temp_handler_456
```

#### 4. 本地快捷引用

```
actor://localhost/{actor_id}
```

- 使用保留字 `localhost` 指代当前节点
- 运行时解析为实际的 node_id
- 适用于本地 Actor 间通信

**示例**：
```
actor://localhost/my_worker
actor://localhost/local_cache
```

### 地址分类对比

| 类型 | 格式 | Gossip 注册 | 多实例 | 使用场景 |
|------|------|-------------|--------|---------|
| 具名服务 | `actor:///ns/name` | ✅ | ✅ | 服务型 Actor，需要服务发现 |
| 具名实例 | `actor:///ns/name@node` | ✅ | - | 访问特定实例 |
| 全局地址 | `actor://node/id` | ❌ | ❌ | 临时/动态 Actor |
| 本地引用 | `actor://localhost/id` | ❌ | ❌ | 本地快捷访问 |

## 命名空间设计

### 强制命名空间

所有具名 Actor 必须属于一个命名空间。命名空间是路径的第一段，用于：

1. **逻辑分组** - 按功能或模块组织 Actor
2. **访问控制** - 未来可基于命名空间实现权限管理
3. **监控隔离** - 按命名空间聚合监控指标

### 预留命名空间

| 命名空间 | 用途 | 示例 |
|---------|------|------|
| `system` | 系统内置 Actor | `system/cluster/monitor` |
| `services` | 服务型 Actor | `services/llm/router` |
| `workers` | Worker 池 | `workers/inference/pool` |
| `user` | 用户自定义 | `user/myapp/handler` |

### 路径层级

命名空间下可以有多级路径，用于更细粒度的组织：

```
actor:///services/llm/router           # 2 级
actor:///services/llm/v2/router        # 3 级
actor:///workers/inference/gpu/pool    # 3 级
```

## 多实例支持

### 概念模型

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   具名 Actor: actor:///services/api                                      │
│                                                                          │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│   │    Instance     │  │    Instance     │  │    Instance     │          │
│   │    @node_a      │  │    @node_b      │  │    @node_c      │          │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘          │
│                                                                          │
│   访问 actor:///services/api 时自动负载均衡选择实例                      │
│   访问 actor:///services/api@node_b 时直接路由到 node_b                  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 实例注册

同一路径的具名 Actor 可在多个节点创建实例：

```rust
// Node A
let path = ActorPath::new("services/llm/router")?;
system.spawn_named(path.clone(), LLMRouter::new()).await?;

// Node B (同一路径，不同实例)
system.spawn_named(path.clone(), LLMRouter::new()).await?;

// Node C
system.spawn_named(path.clone(), LLMRouter::new()).await?;
```

### 负载均衡

访问服务地址时，系统自动从可用实例中选择：

```rust
// 自动负载均衡
let addr = ActorAddress::parse("actor:///services/llm/router")?;
let resp = system.ask(&addr, request).await?;  // 可能路由到 A、B 或 C
```

支持的负载均衡策略（可扩展）：
- Random（随机）- 默认
- RoundRobin（轮询）
- LeastConnections（最少连接）

## 地址解析

### 解析流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Address Resolution                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  actor:///services/api                                                      │
│       │                                                                     │
│       ├──→ 查询 Gossip Registry                                             │
│       │         │                                                           │
│       │         ↓                                                           │
│       │    instances: [node_a, node_b, node_c]                              │
│       │         │                                                           │
│       │         ↓ (负载均衡选择)                                            │
│       │    selected: node_b                                                 │
│       │         │                                                           │
│       └────────→ http://node_b_ip:port/named/services/api                   │
│                                                                             │
│  actor:///services/api@node_a                                               │
│       │                                                                     │
│       ├──→ 查询 node_a 地址                                                 │
│       │         │                                                           │
│       └────────→ http://node_a_ip:port/named/services/api                   │
│                                                                             │
│  actor://node_a/worker_123                                                  │
│       │                                                                     │
│       ├──→ 查询 node_a 地址                                                 │
│       │         │                                                           │
│       └────────→ http://node_a_ip:port/actors/worker_123                    │
│                                                                             │
│  actor://localhost/worker_123                                               │
│       │                                                                     │
│       ├──→ 替换 localhost → current_node_id                                 │
│       │         │                                                           │
│       └────────→ 本地直接调用（不走网络）                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### HTTP 映射

Actor 地址与 HTTP 端点的映射关系：

| Actor 地址 | HTTP 端点 |
|-----------|----------|
| `actor:///services/llm/router` | `POST http://{selected_node}/named/services/llm/router` |
| `actor:///services/llm/router@node_a` | `POST http://node_a/named/services/llm/router` |
| `actor://node_a/worker_123` | `POST http://node_a/actors/worker_123` |

## HTTP API 设计

### 路由规则

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HTTP API Routes                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  具名 Actor 路由                                                            │
│  ──────────────────────────────────────────────────────────────────────────│
│  POST /named/{namespace}/{path...}                                          │
│       发送消息给具名 Actor (ask/tell)                                       │
│       Headers:                                                              │
│         X-Actor-Instance: node_id  (可选，指定实例)                         │
│         X-Message-Type: string     (消息类型标识)                           │
│         X-Response-Required: bool  (是否需要响应，默认 true)                │
│       Body: 序列化的消息内容                                                │
│       Response: 序列化的响应内容                                            │
│                                                                             │
│  GET  /named/{namespace}/{path...}                                          │
│       查询具名 Actor 元信息                                                 │
│       Response: {                                                           │
│         "path": "services/llm/router",                                      │
│         "instances": ["node_a", "node_b"],                                  │
│         "metadata": {...}                                                   │
│       }                                                                     │
│                                                                             │
│  普通 Actor 路由                                                            │
│  ──────────────────────────────────────────────────────────────────────────│
│  POST /actors/{actor_id}                                                    │
│       发送消息给普通 Actor                                                  │
│       Headers: 同上                                                         │
│                                                                             │
│  GET  /actors/{actor_id}                                                    │
│       查询普通 Actor 元信息                                                 │
│                                                                             │
│  集群管理                                                                   │
│  ──────────────────────────────────────────────────────────────────────────│
│  POST /cluster/gossip                                                       │
│       Gossip 协议消息                                                       │
│                                                                             │
│  GET  /cluster/members                                                      │
│       查看集群成员列表                                                      │
│                                                                             │
│  GET  /cluster/registry                                                     │
│       查看具名 Actor 注册表                                                 │
│                                                                             │
│  GET  /cluster/registry/{namespace}                                         │
│       按命名空间过滤注册表                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### HTTP 语义

| 方法 | 语义 | 用途 |
|------|------|------|
| `POST` | 发送消息 | ask/tell 消息传递 |
| `GET` | 查询信息 | 获取 Actor 元信息、状态 |

## Gossip 注册表

### 数据结构

```rust
/// 具名 Actor 注册信息
pub struct NamedActorInfo {
    /// Actor 路径
    pub path: ActorPath,
    /// 所有实例所在的节点
    pub instances: HashSet<NodeId>,
    /// 版本号（用于冲突解决）
    pub version: u64,
}

/// 全局注册表
pub struct GlobalRegistry {
    /// 节点成员信息 (node_id -> MemberInfo)
    pub members: HashMap<NodeId, MemberInfo>,
    
    /// 具名 Actor 注册表 (path -> NamedActorInfo)
    /// 只有具名 Actor 在此注册
    pub named_actors: HashMap<String, NamedActorInfo>,
}
```

### Gossip 消息

```rust
pub enum GossipMessage {
    // ... 现有消息 ...
    
    /// 具名 Actor 实例注册
    NamedActorRegistered {
        path: ActorPath,
        node_id: NodeId,
    },
    
    /// 具名 Actor 实例注销
    NamedActorUnregistered {
        path: ActorPath,
        node_id: NodeId,
    },
    
    /// 同步消息
    Sync {
        from: NodeId,
        members: Vec<MemberInfo>,
        named_actors: Vec<NamedActorInfo>,
    },
}
```

### 注册流程

1. **创建具名 Actor**：调用 `spawn_named(path, actor)` 
2. **本地注册**：Actor 创建成功后注册到本地
3. **Gossip 广播**：发送 `NamedActorRegistered` 消息
4. **集群同步**：其他节点更新注册表

### 注销流程

1. **停止 Actor**：Actor 优雅关闭
2. **本地注销**：从本地注册表移除
3. **Gossip 广播**：发送 `NamedActorUnregistered` 消息
4. **节点故障**：SWIM 检测到节点故障时，自动清理该节点的所有实例

## 数据结构定义

### ActorPath

```rust
/// Actor 路径（命名空间 + 层级路径 + 名称）
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ActorPath {
    /// 路径段，如 ["services", "llm", "router"]
    segments: Vec<String>,
}

impl ActorPath {
    /// 预留的系统命名空间
    pub const SYSTEM_NAMESPACES: &[&str] = &["system"];
    
    /// 创建新路径（至少需要 namespace/name 两段）
    pub fn new(path: impl AsRef<str>) -> Result<Self, ParseError>;
    
    /// 获取命名空间（第一段）
    pub fn namespace(&self) -> &str;
    
    /// 获取名称（最后一段）
    pub fn name(&self) -> &str;
    
    /// 获取完整路径字符串
    pub fn as_str(&self) -> String;
    
    /// 检查是否为系统命名空间
    pub fn is_system(&self) -> bool;
}
```

### ActorAddress

```rust
/// 保留的特殊节点标识
pub const LOCALHOST: &str = "localhost";

/// Actor 地址
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum ActorAddress {
    /// 具名 Actor - actor:///namespace/path/name[@node]
    Named {
        path: ActorPath,
        instance: Option<NodeId>,
    },
    
    /// 全局地址 - actor://node_id/actor_id
    Global {
        node_id: NodeId,
        actor_id: String,
    },
}

impl ActorAddress {
    /// 解析 URI 格式地址
    pub fn parse(uri: &str) -> Result<Self, ParseError>;
    
    /// 转换为 URI 字符串
    pub fn to_uri(&self) -> String;
    
    /// 解析 localhost 为实际 node_id
    pub fn resolve_localhost(self, current_node: &NodeId) -> Self;
    
    /// 检查是否为本地引用
    pub fn is_localhost(&self) -> bool;
    
    /// 为具名地址添加实例定位
    pub fn with_instance(self, node_id: NodeId) -> Self;
}
```

## 使用示例

### 创建和访问具名 Actor

```rust
// === 创建具名 Actor ===

// 在 Node A 上创建
let path = ActorPath::new("services/llm/router")?;
let router_a = system.spawn_named(path.clone(), LLMRouter::new()).await?;

// 在 Node B 上创建同名实例
let router_b = system.spawn_named(path.clone(), LLMRouter::new()).await?;


// === 访问具名 Actor ===

// 服务地址访问（自动负载均衡）
let addr = ActorAddress::parse("actor:///services/llm/router")?;
let response: Response = system.ask(&addr, request).await?;

// 指定实例访问
let addr = ActorAddress::parse("actor:///services/llm/router@node_a")?;
let response: Response = system.ask(&addr, request).await?;

// 查询所有实例
let info = system.lookup_named(&path).await?;
println!("Instances: {:?}", info.instances);  // ["node_a", "node_b"]
```

### 创建和访问普通 Actor

```rust
// === 创建普通 Actor ===

// 创建（不注册到 Gossip）
let worker = system.spawn(TempWorker::new()).await?;
let worker_addr = worker.address();  // actor://node_a/worker_xyz123


// === 访问普通 Actor ===

// 通过完整地址访问
let addr = ActorAddress::parse("actor://node_a/worker_xyz123")?;
let result = system.ask(&addr, task).await?;

// 本地快捷访问
let local_addr = ActorAddress::parse("actor://localhost/worker_xyz123")?;
let result = system.ask(&local_addr, task).await?;


// === 地址传递 ===

// Worker 将自己的地址告诉 Manager
manager.tell(RegisterWorker { 
    addr: worker.address()  // actor://node_a/worker_xyz123
}).await?;

// Manager 可以用这个地址访问 Worker
```

### 跨节点通信

```rust
// Node A 上创建 Worker
let worker = system.spawn(Worker::new()).await?;
let worker_addr = worker.address();  // actor://node_a/worker_123

// 将地址发送给 Node B 上的 Manager
let manager_addr = ActorAddress::parse("actor:///services/task/manager")?;
system.tell(&manager_addr, RegisterWorker { addr: worker_addr }).await?;

// Node B 上的 Manager 收到后，可以直接访问 Node A 的 Worker
impl Handler<Task> for Manager {
    async fn handle(&mut self, task: Task, ctx: &mut ActorContext) {
        // worker_addr 是 "actor://node_a/worker_123"
        let worker_ref = ctx.actor_ref(&self.worker_addr).await?;
        worker_ref.tell(task).await?;
    }
}
```

## 总结

| 特性 | 设计决策 |
|------|---------|
| 地址格式 | URI scheme：`actor://` |
| 具名 vs 普通 | 具名通过 Gossip 注册，普通不注册 |
| 多实例 | 同一路径可在多节点部署，自动负载均衡 |
| 实例定位 | `@node_id` 后缀指定实例 |
| 本地引用 | `localhost` 保留字 |
| 命名空间 | 强制要求，提供逻辑隔离 |
| HTTP 语义 | POST=消息，GET=查询 |

## 附录：地址格式 BNF

```bnf
actor-address   = named-address | global-address

named-address   = "actor:///" path [ "@" node-id ]
global-address  = "actor://" node-id "/" actor-id

path            = namespace "/" name
                | namespace "/" sub-path "/" name
sub-path        = segment
                | segment "/" sub-path

namespace       = segment
name            = segment
segment         = 1*( ALPHA | DIGIT | "_" | "-" )

node-id         = "localhost" | identifier
actor-id        = identifier
identifier      = 1*( ALPHA | DIGIT | "_" | "-" )
```

