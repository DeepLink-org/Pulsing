# 架构概述

Pulsing Actor 系统架构概览。

## 系统组件

```mermaid
graph TB
    subgraph ActorSystem["ActorSystem"]
        subgraph LocalActors["Local Actors"]
            A["Actor A<br/>(Mailbox)"]
            B["Actor B<br/>(Mailbox)"]
            C["Actor C<br/>(Mailbox)"]
        end

        subgraph Transport["HTTP Transport"]
            T1["POST /actor/{name}<br/>Actor Messages"]
            T2["POST /cluster/gossip<br/>Cluster Protocol"]
        end

        subgraph Cluster["GossipCluster"]
            M["成员发现<br/>(Membership)"]
            R["Actor 位置<br/>(Actor Registry)"]
            S["故障检测<br/>(SWIM)"]
        end
    end

    A & B & C --> Transport
    Transport --> Cluster

    style ActorSystem fill:#f5f5f5,stroke:#333,stroke-width:2px
    style LocalActors fill:#e3f2fd,stroke:#1976d2
    style Transport fill:#fff3e0,stroke:#f57c00
    style Cluster fill:#e8f5e9,stroke:#388e3c
```

## 核心概念

### Actor

Actor 是一个计算单元，具有以下特性：
- 封装状态
- 异步处理消息
- 具有唯一标识符
- 可以是本地或远程

### Message

消息是 Actor 之间的通信机制：
- **单条消息**：请求-响应模式
- **流式消息**：连续数据流

### ActorRef

ActorRef 提供位置透明性：
- 本地和远程 Actor 使用相同 API
- 根据 Actor 位置自动路由
- 处理序列化/反序列化

### Cluster

集群提供：
- **节点发现**：通过 Gossip 协议自动发现
- **Actor 注册表**：跟踪跨节点的 Actor 位置
- **故障检测**：使用 SWIM 协议进行健康检查

## 消息流程

### 本地消息

```mermaid
sequenceDiagram
    participant S as Sender
    participant M as Mailbox
    participant A as Actor

    S->>M: ask(Ping)
    M->>A: recv()
    A->>A: handle()
    A-->>M: respond(Pong)
    M-->>S: Pong
```

### 远程消息

```mermaid
sequenceDiagram
    participant S as Sender
    participant R as ActorRef(Remote)
    participant N as Network
    participant A as Actor (Node B)

    S->>R: ask(Ping)
    R->>N: HTTP POST /actor/{name}
    Note over R,N: {msg_type, payload}
    N->>A: Envelope
    A->>A: handle()
    A-->>N: {result}
    N-->>R: HTTP Response
    R-->>S: Pong
```

## 设计原则

1. **零外部依赖**：无需 etcd、NATS 或其他外部服务
2. **位置透明**：本地和远程 Actor 使用相同 API
3. **高性能**：基于 Tokio 异步运行时构建
4. **简单 API**：易于使用的 Python 接口
5. **集群感知**：自动发现和路由

## 更多详情

- [Actor 系统设计](actor-system.md)
- [节点发现](node-discovery.md)
- [集群组网](cluster-networking.zh.md)
- [Actor 寻址](actor-addressing.md)
- [HTTP2 传输](http2-transport.md)
