# Architecture

Overview of Pulsing Actor System architecture.

## System Components

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

## Key Concepts

### Actor

An Actor is a computational unit that:
- Encapsulates state
- Processes messages asynchronously
- Has a unique identifier
- Can be local or remote

### Message

Messages are the communication mechanism between actors:
- **Single messages**: Request-response pattern
- **Streaming messages**: Continuous data flow

### ActorRef

ActorRef provides location transparency:
- Same API for local and remote actors
- Automatic routing based on actor location
- Handles serialization/deserialization

### Cluster

The cluster provides:
- **Node Discovery**: Automatic discovery via Gossip protocol
- **Actor Registry**: Track actor locations across nodes
- **Failure Detection**: SWIM protocol for health checks

## Message Flow

### Local Message

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

### Remote Message

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

## Design Principles

1. **Zero External Dependencies**: No need for etcd, NATS, or other external services
2. **Location Transparency**: Same API for local and remote actors
3. **High Performance**: Built on Tokio async runtime
4. **Simple API**: Easy to use Python interface
5. **Cluster Awareness**: Automatic discovery and routing

## For More Details

- [Actor System Design](../design/actor-system.md)
- [Node Discovery](../design/node-discovery.md)
- [Actor Addressing](../design/actor-addressing.md)
- [HTTP2 Transport](../design/http2-transport.md)
