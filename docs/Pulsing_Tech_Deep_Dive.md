# Pulsing 内部技术串讲 PPT 大纲

> **目标**: 深入讲解 Pulsing 的通信模型、架构设计和 API 取舍
> **受众**: 内部技术团队，具备分布式系统和 Python/Rust 基础；若听众偏 **K8s 运维/编排** 或 **LLM 训练/推理框架**，各 Slide 的【受众提示】给出对应类比与强调点
> **时长**: 60-90 分钟

---

## Part 1: 问题背景与通信范式演进 (20分钟)

### Slide 1: 封面

**标题**: Pulsing 技术深潜：分布式 Actor 通信模型的设计与实现

**议程**:
1. 通信范式演进：MPI → ZMQ → RPC → Actor
2. Pulsing 通信模型：四种通信范式详解
3. 架构设计：Gossip 组网、HTTP/2 传输、背压机制
4. API 设计：Typed Proxy、Ray 集成、性能权衡
5. **形式化模型**：因果序、一致性、会话类型（Part 5）
6. 总结与讨论

---

### Slide 2: 分布式通信的核心问题

**问题**: 如何让分布式进程可靠地协同工作？

**不确定性来源**:
- 网络延迟不可预测
- 节点随时可能故障
- 消息可能丢失、乱序
- 生产/消费速率不匹配

**四代技术的不同回答**:

| 代际 | 对不确定性的态度 | 代表技术 |
|------|-----------------|----------|
| MPI | 消除不确定性（假设世界确定） | MPI, NCCL |
| ZMQ | 暴露不确定性（给工具自己处理） | ZeroMQ |
| RPC | 掩盖不确定性（假装网络不存在） | gRPC, Thrift |
| Actor | 内化不确定性（纳入编程模型） | Erlang, Akka, Pulsing |

**【受众提示】** 偏 K8s/LLM：Part 1 回答的是「为什么选 Actor 而不是再堆一层 RPC」——和 K8s 互补（K8s 管调度与网络，Pulsing 管应用层发现与通信）；和训练/推理栈互补（NCCL 管数据面，Pulsing 管编排与流式）。

---

### Slide 3: 第一代 - MPI：静态世界的极致性能

**核心假设**: 所有节点启动时就绪，网络可靠，任务均匀

**BSP 模型**:
```
计算阶段 → Barrier → 通信阶段
```

**AllReduce 示例**:
```
Rank 0: [██████████] ─── Barrier ───▶ AllReduce
Rank 1: [████████]   ─── Barrier ───▶ AllReduce  ← 木桶效应
Rank 2: [████████████]               (最慢的节点决定速度)
```

**为什么 MPI 仍统治 AI 训练的数据面**:
- 通信拓扑预定义 → 可针对硬件优化 (NVLink, IB)
- Buffer 大小已知 → DMA 零拷贝、计算通信重叠
- 参与者固定 → 编译期调度优化

**局限性**:
- 刚性同步无法适应动态拓扑
- 容错几乎为零（一崩全崩）
- 无法处理不规则通信模式

---

### Slide 4: 第二代 - ZMQ：自由但危险的积木

**突破**: 从集合通信到任意点对点通信

**Socket 模式**:
```
REQ/REP: 请求响应
PUB/SUB: 发布订阅
PUSH/PULL: 负载均衡
DEALER/ROUTER: 异步路由
```

**问题：只有机制，没有策略**

**痛点 1：状态机约束容易破坏**:
```python
# REQ socket 必须严格遵循 send -> recv -> send -> recv
A(REQ): send(req1) → send(req2)  # 第二次 send 阻塞或报 EFSM
A(REQ): recv(resp1)              # 期望 resp1/resp2 顺序容易乱
```

**痛点 2：消息"看似丢失"**:
```python
# PUB/SUB 不为晚到订阅者保留历史
Pub: send(msg)        # 发布端发出
Sub: [尚未 connect]   # 订阅者还没准备好
Sub: recv()           # 这条 msg 不会补发
```

**痛点 3：背压处理困难**:
- HWM (高水位) 行为依赖 socket 类型
- 可能阻塞、可能返回 EAGAIN、可能直接丢弃
- 缺少端到端、与业务语义对齐的背压范式

> ZMQ 解决了"如何让任意节点随时通信"，但把"如何保证通信可靠"留给每个开发者。

---

### Slide 5: 第三代 - RPC：伪装成本地调用的代价

**核心思想**: Call 语义封装远程通信

```python
# ZMQ: 两步操作，手动配对
socket.send(request)
response = socket.recv()  # 忘记 recv 就卡死

# RPC: 一行代码，请求响应语法绑定
response = service.compute(request)
```

**分布式计算的八大误区**:
1. 网络是可靠的
2. 延迟为零（本地 vs 远程：10^3 ~ 10^6 倍差异）
3. 带宽无限
4. 网络是安全的
5. 拓扑不变
6. 只有一个管理员
7. 传输成本为零
8. 网络同质

**RPC 试图掩盖这些现实，诱导写出"假装网络不存在"的代码。**

**微服务税**:
```
[Service A] ──IPC── [Sidecar] ──network── [Sidecar] ──IPC── [Service B]
                          │                    │
                          └──── [Etcd/Consul] ─┘
                                    │
                          [Prometheus] [Jaeger]
```

RPC 本身只是点对点调用协议，需要外挂：
- 服务发现 (Etcd/Consul)
- 流量治理 (Envoy)
- Sidecar 模式剥离治理逻辑

**无状态设计的冲突**:
- AI 系统需要状态驻留：KV Cache、Agent Memory、模型权重
- 每次请求从 Redis/DB 拉取状态 → 延迟不可接受
- RPC 的无状态假设与 AI 场景天然冲突

---

### Slide 6: 第四代 - Actor：拥抱不确定性

**核心原则**:

**1. 消息传递，而非函数调用**:
```
Actor A              Actor B
   │─── Message ───▶   │
   │                   │ (放入 Mailbox)
   │                   │ (异步处理)
   │◀── Response ─────│ (可选)
```
- 发送是异步非阻塞的
- 每个 Actor 有 Mailbox 作为天然缓冲区
- "尽力而为"投递语义，迫使考虑失败场景

**2. 私有状态，而非共享内存**:
```python
@pul.remote
class InferenceWorker:
    def __init__(self, model_path):
        self.model = load_model(model_path)  # 常驻内存
        self.kv_cache = {}                    # 常驻内存
```
- 无锁、无竞态条件
- 状态内存驻留，天然适合 AI 场景

**3. "Let it crash" + 监督**:
```
       [Supervisor]
        /        \
 [Worker A]    [Worker B]
     ↓              │
   崩溃!            │
     ↓              │
  自动重启 ✓        │
```
- 承认故障是常态
- 父 Actor 监控子 Actor
- 崩溃隔离 + 自动重启 = 自愈能力

**【受众提示】** 类比：Actor ≈ 有**唯一身份**的有状态工作单元（类似 StatefulSet 的一格），一个进程里可有多 Actor，每个有自己的邮箱与顺序处理；Ask ≈ 发请求等响应（像 HTTP），Tell ≈ 发出去不管（像 fire-and-forget 日志）。

---

### Slide 7: 四代技术对比

| 维度 | MPI | ZMQ | RPC | Actor |
|------|-----|-----|-----|-------|
| 核心隐喻 | 军队同步行进 | 对讲机自由频道 | 电话一对一 | 邮件系统异步投递 |
| 控制平面 | 静态 | 手动 | 外挂 | 内建 |
| 状态管理 | 紧耦合 SPMD | 无 | 外部化 (Redis) | 内存驻留 |
| 通信基座 | TCP/RDMA | TCP Socket | HTTP/2/QUIC | HTTP/2 |
| 容错 | 一崩全崩 | 依赖开发者 | 重试+熔断 | Let it crash |
| 适用领域 | 数据面 (Tensor) | 传输层 | 业务面 (CRUD) | 控制面 (编排) |

**关键洞察**:
> MPI/NCCL 是 AI 基础设施的**高速公路**（张量搬运的数据面）
> Pulsing 是**智能交通指挥系统**（复杂编排的控制面）

**【受众提示】** 偏 LLM 训练/推理：数据面继续用 MPI/NCCL；编排、路由、流式、状态常驻用 Pulsing，和 Ray 定位类似但通信模型更清晰。

---

## Part 2: Pulsing 通信模型详解 (25分钟)

### Slide 8: Actor 的核心特性

**Actor 一次只处理一条消息**:
```
Actor 邮箱（FIFO 队列）
    ↓
[消息1] → Actor 处理 → 响应1
[消息2] → Actor 处理 → 响应2  ← 必须等待消息1完成
[消息3] → Actor 处理 → 响应3  ← 必须等待消息2完成
```

**澄清**：这里的“一次只处理一条”指同一时刻只**推进**一条消息的执行；当处理逻辑在 `await` 处让出执行权时，Actor 可以去推进其他消息（因此 I/O 场景下可提高吞吐，但并非多线程并行执行同一段同步代码）。

**为什么需要不同的通信范式？**

**阻塞 vs 非阻塞**:
```
❌ 同步阻塞模式：
消息1: [等待HTTP...████████] 500ms  ← 阻塞
消息2: [等待中...]                      ← 无法处理
消息3: [等待中...]                      ← 无法处理

✅ 异步非阻塞模式：
消息1: [等待HTTP...] 500ms  ← 后台等待
消息2: [处理中...] 10ms     ← 可以同时处理
消息3: [处理中...] 10ms     ← 可以同时处理
```

**流式 vs 等待全部**:
```
❌ 等待全部：
用户: [等待...████████████████] 10秒 → 看到结果

✅ 流式：
用户: [token1][token2][token3]...  ← 立即看到
```

**【受众提示】** 偏 LLM：训练里「取下一 batch」用同步或异步；**推理里 token 必须流式**，否则首 token 延迟和体验都差。

---

### Slide 9: 四种通信范式总览

| 范式 | 方法类型 | 为什么需要 | 使用场景 |
|------|----------|------------|----------|
| **同步** | `def method()` | 快速操作不需要并发 | 快速 CPU、状态变更 |
| **异步** | `async def method()` | 避免阻塞，提高吞吐（I/O 等待期间可推进其他消息） | I/O 操作、外部 API |
| **流式** | `async def method()` + `yield` | 增量返回，提升体验 | LLM token、大数据 |
| **发送即忘** | `tell()` | 不需要响应 | 日志、指标 |

---

### Slide 10: 范式 1 - 同步方法

**原理**: 对于快速操作（< 10ms），并发开销大于收益

**行为特性**:
- Actor 一次处理一个请求
- 处理时阻塞 Actor
- 严格顺序执行

**适用场景**:
✅ 快速 CPU 操作（计算、状态更新）
✅ 简单状态变更（计数器、字典）
❌ 网络请求、文件 I/O

**示例**:
```python
@pul.remote
class Counter:
    def __init__(self):
        self.value = 0

    # ✅ 好：快速状态变更
    def increment(self, n: int = 1) -> int:
        self.value += n
        return self.value

    # ❌ 差：网络 I/O 阻塞 Actor
    def fetch_data(self, url: str) -> dict:
        response = requests.get(url)  # 阻塞数秒！
        return response.json()
```

**性能特征**:
```
请求1: [████] 2ms
请求2:      [████] 2ms
请求3:          [████] 2ms
总计: 6ms（顺序执行）
```

---

### Slide 11: 范式 2 - 异步方法

**原理**: `await` 时让出控制权，Actor 可处理其他消息

**对比**:
```python
# ❌ 同步：阻塞 Actor
def fetch_data(self, url: str) -> dict:
    response = requests.get(url)  # 阻塞 500ms
    return response.json()
# 结果：Actor 在这 500ms 内无法处理其他消息

# ✅ 异步：非阻塞
async def fetch_data(self, url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)  # 可处理其他请求
        return response.json()
# 结果：Actor 可在 I/O 等待期间推进其他请求，提高整体吞吐
```

**适用场景**:
✅ I/O 操作（HTTP、数据库、文件）
✅ 外部 API 调用
✅ 耗时 > 10ms 的操作

**并发执行**:
```python
async def fetch_user_profile(self, user_id: str) -> dict:
    # 这些操作并发运行，不是顺序运行
    user, orders, preferences = await asyncio.gather(
        self.fetch_user(user_id),
        self.get_orders(user_id),
        self.get_preferences(user_id),
    )
    return {"user": user, "orders": orders, "preferences": preferences}
```

**性能特征**:
```
请求1: [████████████████████] 50ms（等待 HTTP）
请求2: [████████████████████] 50ms（等待 HTTP）← 并发！
请求3: [████████████████████] 50ms（等待 HTTP）← 并发！
总计: ~50ms（不是 150ms）
```

---

### Slide 12: 范式 3 - 流式响应

**问题**: LLM 生成 1000 个 token，等待全部完成用户体验差

**原理**: `yield` 增量返回结果

**适用场景**:
✅ LLM token 生成
✅ 大数据传输
✅ 实时数据流
✅ 进度更新

**示例**:
```python
@pul.remote
class LLMService:
    # ✅ 流式 LLM token
    async def generate(self, prompt: str):
        async for token in self.llm_client.stream(prompt):
            yield {"token": token, "type": "token"}
        yield {"type": "done", "total_tokens": count}

# 消费端
async for chunk in service.generate("Hello"):
    if chunk["type"] == "token":
        print(chunk["token"], end="", flush=True)
```

**行为特性**:
- 增量交付：结果可用立即发送
- 非阻塞：Actor 可处理其他消息
- 背压：有界通道自然流控
- 可取消：客户端可取消流消费

**性能特征**:
```
客户端收到第一个结果: [██] 10ms  ← 立即看到
客户端收到所有结果:   [████████████████████] 50ms
```

**【受众提示】** 流式 + 背压 ≈ 下游消费慢时上游自动慢下来，和训练里 DataLoader 的 prefetch 上限、推理里按 token 消费一致。

---

### Slide 13: 范式 4 - Ask vs Tell

**核心区别**: 是否需要等待响应

**Ask** - 请求/响应:
```python
# 需要结果进行后续处理
result = await counter.increment(10)
print(f"新值: {result}")

# 需要检查成功
try:
    user = await service.get_user("user123")
except PulsingActorError:
    print("用户未找到")
```

**Tell** - 发送即忘:
```python
# 日志记录 - 不需要响应
await logger.tell({"level": "info", "message": "用户已登录"})

# 指标 - 发送即忘
await metrics.tell({"event": "page_view", "page": "/home"})
```

**对比**:

| 方面 | `ask()` | `tell()` |
|------|---------|----------|
| 响应 | ✅ 返回值 | ❌ 无响应 |
| 错误处理 | ✅ 抛出异常 | ❌ 静默失败 |
| 吞吐量 | 较低（等待响应） | 较高（不等待） |
| 使用场景 | 需要结果 | 可丢弃 |

---

### Slide 14: 决策流程

```
开始：你的操作需要什么？

1. 需要响应吗？
   ├─ 否 → 使用 tell()（发送即忘）
   │
   └─ 是 → 继续

2. 操作需要多长时间？
   ├─ < 10ms → 使用 def method()（同步）
   │
   └─ > 10ms → 继续

3. 需要增量返回结果吗？
   ├─ 否 → 使用 async def method()（异步）
   │
   └─ 是 → 使用 async def method() + yield（流式）
```

**最佳实践总结**:
1. 快速操作（< 10ms）：同步
2. I/O 操作（> 10ms）：异步
3. 增量结果：流式
4. 不需要响应：tell()
5. LLM token 生成：始终流式

---

### Slide 15: 常见陷阱

**陷阱 1：对 I/O 使用同步**:
```python
# ❌ 差：阻塞 Actor 数秒
def fetch_data(self, url: str) -> dict:
    response = requests.get(url)
    return response.json()

# ✅ 好：非阻塞异步
async def fetch_data(self, url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

**陷阱 2：对快速操作使用异步**:
```python
# ❌ 差：不必要的复杂度
async def increment(self, n: int) -> int:
    self.value += n  # < 1ms
    return self.value

# ✅ 好：简单同步
def increment(self, n: int) -> int:
    self.value += n
    return self.value
```

**陷阱 3：LLM 不使用流式**:
```python
# ❌ 差：用户等待 10-30 秒
async def generate(self, prompt: str) -> str:
    tokens = []
    async for token in self.llm_client.stream(prompt):
        tokens.append(token)
    return "".join(tokens)

# ✅ 好：token 到达时流式传输
async def generate(self, prompt: str):
    async for token in self.llm_client.stream(prompt):
        yield token
```

---

## Part 3: Pulsing 架构设计 (25分钟)

### Slide 16: 系统架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                     ActorSystem                             │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Local Actor A│  │ Local Actor B│  │ Local Actor C│      │
│  │  (Mailbox)   │  │  (Mailbox)   │  │  (Mailbox)   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         └─────────────────┴─────────────────┘               │
│                           │                                 │
│              ┌────────────▼────────────┐                   │
│              │     HTTP/2 Transport    │                   │
│              │  (Actor RPC + Gossip)   │                   │
│              └────────────┬────────────┘                   │
│                           │                                 │
│         ┌─────────────────┼─────────────────┐              │
│         ▼                 ▼                 ▼              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │
│  │ Node A      │◄──►│ Node B      │◄──►│ Node C      │      │
│  └─────────────┘   └─────────────┘   └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

**关键设计**: 核心通信复用一个 HTTP/2 端口

**【受众提示】** 偏 K8s：健康检查、Actor RPC、Gossip 都走**单端口**，和 K8s 一个 Service 端口即可暴露。

---

### Slide 17: 无外置服务发现存储的组网 - Gossip + SWIM

**问题**: RPC 系统需要外挂 Etcd/Consul 做服务发现

**Pulsing 方案**: 内建集群能力

**启动流程**:
```
New Pod                Service IP              Existing Pods
   │                        │                      │
   ├── Probe 1 (Join) ────▶ ├── 路由到 Pod A ────▶ │
   │◀── Welcome [A] ───────┤                       │
   │                        │                      │
   ├── Probe 2 (Join) ────▶ ├── 路由到 Pod B ────▶ │
   │◀── Welcome [A,B] ─────┤                       │
   │                        │                      │
   └── 开始正常 Gossip ─────────────────────────────┘
        每 200ms 与随机节点同步状态
```

**SWIM 故障检测**:
```
Alive ──Ping超时──> Suspect ──Suspect超时──> Dead
  ▲                      │
  └────────Ack───────────┘
```

**单一端口**:
```
Port 8000:
  ├── POST /actor/{name}    Actor 消息
  ├── POST /cluster/gossip  集群协议
  └── GET /health           健康检查
```

**【受众提示】** 偏 K8s：**不依赖 Etcd/Consul**；新 Pod 通过 K8s Service IP probe 一次即可拿到成员列表。Gossip ≈ 应用层自己维护成员视图、最终一致，和 K8s Endpoints 传播类似但在应用层完成。

---

### Slide 18: 位置透明寻址

**URI 风格地址体系**:
```
actor:///services/llm/router           → 具名 Actor（集群路由）
actor:///services/llm/router@node_a    → 指定节点实例
actor://node_a/worker_123              → 全局精确地址
actor://localhost/worker_123           → 本地快捷引用
```

**负载均衡**:
```python
# Node A 部署
await system.spawn(LLMRouter(), name="services/llm/router")

# Node B 也部署同名 Actor
await system.spawn(LLMRouter(), name="services/llm/router")

# 从任意节点访问——自动选择实例
router = await system.resolve("services/llm/router")
result = await router.ask(request)  # 可能路由到 A 或 B
```

**【受众提示】** 偏 K8s：和 K8s Service 后多 Pod 一致——一个 Service 后多个 Pulsing 节点，`resolve(name)` 即负载均衡。

---

### Slide 19: HTTP/2 传输层设计

**为什么选择 HTTP/2 h2c**:
- h2c (HTTP/2 over cleartext)：内网**通常**可不启用 TLS（取决于部署安全要求）
- Prior Knowledge 模式：省掉升级协商
- 多路复用：单连接并行传输多个 Stream

**二进制帧协议**:
```
+----------------+-------+-------------+----------+------------+
| Length (4B BE) | Flags | MsgType Len | MsgType  | Raw Data   |
+----------------+-------+-------------+----------+------------+
```

**消息模式**:
```
Ask:    POST /actor/{name}  x-message-mode: ask    → 200 + response
Tell:   POST /actor/{name}  x-message-mode: tell   → 202 Accepted
Stream: POST /actor/{name}  x-message-mode: stream → 200 + stream
```

**流式帧格式**:
```
[4字节长度][flags][msg_type_len][msg_type][data]

Flags:
  - 0x01: FLAG_END   - 流结束
  - 0x02: FLAG_ERROR - 错误标志
```

---

### Slide 20: 背压传导机制（基于 HTTP/2 Flow Control）

**问题**: Token 生成速度 > 消费速度时怎么办？

**HTTP/2 Flow Control 解决方案**:
```
[LLM Actor]         [Network]           [Client]
     │                   │                   │
     │── Token ────────▶ │── Token ────────▶ │
     │── Token ────────▶ │── Token ────────▶ │ ← 处理变慢
     │── Token ────────▶ │   H2 窗口填满     │
     │   send() Pending  │◀─ 不再发送 WINDOW_UPDATE ─│ ← 窗口耗尽
     │   ← 自动暂停生成  │                   │
     │                   │                   │ ← 处理完成
     │   send() 恢复     │◀─ Window Update ──│ ← 释放窗口
     │── Token ────────▶ │── Token ────────▶ │
```

**关键**: 传输层背压无需用户手写流控代码，HTTP/2 + Rust Future 可自然传导背压（业务层仍建议显式处理取消/超时，并避免在应用层做无界预生成/缓冲）

**【受众提示】** 偏 K8s：流式时下游慢了**不用写限流代码**，上游自动被压住，避免 OOM；和 K8s resource limits 配合更好预估内存。

---

### Slide 21: 有状态编排

**Actor 作为状态的 Owner**:
```python
@pul.remote
class InferenceWorker:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)  # 常驻内存
        self.kv_cache = {}                    # 常驻内存

    async def generate(self, prompt: str):
        # 直接使用内存中的模型和缓存
        for token in self.model.generate(prompt, cache=self.kv_cache):
            yield {"token": token}
```

**优势**:
- KV Cache 常驻：请求间无需序列化/反序列化
- 模型权重常驻：加载一次，服务终身
- Agent 记忆常驻：多轮对话上下文直接存内存

**【受众提示】** 偏 K8s：有状态 = 状态在本地内存，**不依赖外部 Redis 存会话**。偏 LLM：KV Cache/模型权重常驻，和 vLLM、训练 checkpoint 的「状态在进程里」一致，不反复序列化。

---

## Part 4: API 设计取舍 (15分钟)

### Slide 22: API 设计原则

**目标**: 简洁、类型友好、与 Python 生态融合

**核心决策**:
1. **位置透明**: 本地和远程调用**形态一致**（同一套 `await` API），但性能与失败语义不同，需要显式使用超时/重试/幂等来面对分布式现实
2. **类型友好**: Typed Proxy 提供 IDE 补全与静态分析支持（mypy/pyright 视配置而定），运行时仍会做必要的校验与序列化处理
3. **渐进式**: 简单场景简单，复杂场景可行

---

### Slide 23: Python API 设计

**基础用法**:
```python
import pulsing as pul

@pul.remote
class Counter:
    def __init__(self, init=0):
        self.value = init

    def incr(self):                     # 同步方法
        self.value += 1
        return self.value

    async def fetch_and_add(self, url):  # 异步方法
        data = await http_get(url)
        self.value += data
        return self.value

# 创建和调用
counter = await Counter.spawn(name="counter")
result = await counter.incr()
```

**关键设计**:
- `@pul.remote` 装饰器：将类转为 Actor
- `spawn()` 创建实例
- `await` 直接调用：没有 `.remote()`，没有 `ray.get()`

---

### Slide 24: Typed vs Untyped Proxy

**Typed Proxy（推荐）**:
```python
# 通过类方法 resolve，返回类型化代理
proxy = await Counter.resolve("counter")
result = await proxy.incr()  # 类型检查、IDE 提示
```

**手动绑定**:
```python
ref = await pul.resolve("counter", timeout=30)
proxy = ref.as_type(Counter)
result = await proxy.incr()
```

**Untyped Proxy（类型未知时）**:
```python
ref = await pul.resolve("service_name")
proxy = ref.as_any()
result = await proxy.any_method(args)  # 运行时检查
```

**取舍**:
- Typed: 类型友好（IDE/静态分析支持）
- Untyped: 灵活性，用于动态场景

---

### Slide 25: Ray 集成设计

**问题**: Ray 强在调度与资源管理；在“会话型通信语义”（流式、背压、命名/解析、通信边界可形式化讨论）上，Pulsing 更聚焦

**方案**: `pul.mount()` 桥接
```python
@ray.remote
class Worker:
    def __init__(self, name):
        pul.mount(self, name=name)  # 一行代码桥接

    async def call_peer(self, peer_name, msg):
        proxy = (await pul.resolve(peer_name, timeout=30)).as_any()
        return await proxy.greet(msg)

# Ray 调度 + Pulsing 通信
ray.init()
workers = [Worker.remote(f"w{i}") for i in range(3)]
ray.get(workers[0].call_peer.remote("w1", "hi"))
```

**内部实现**:
1. 初始化 Pulsing（如果未初始化）
2. 将实例包装为 Pulsing Actor
3. 注册到网络，Gossip 广播名称

**【受众提示】** 偏 LLM 训练/推理：已有 Ray 调度时，可 **Ray 管调度、Pulsing 管进程间通信**；`pul.mount()` 把 Ray Worker 暴露成 Actor，无需再起一套服务发现。

---

### Slide 26: Low-level API

**显式 ActorSystem**:
```python
# 需要多个系统或精细控制时使用
system = await pul.actor_system(
    addr="0.0.0.0:8000",
    seeds=["node1:8000"]
)
```

**Low-level spawn**（需要 receive 方法）:
```python
actorref = await pul.spawn(
    actor: Actor,
    name: str | None = None,
    restart_policy: str = "never",
    max_restarts: int = 3,
)

# 消息传递
response = await actorref.ask(request)
await actorref.tell(msg)
```

**适用场景**:
- 需要自定义 Actor 生命周期
- 需要与 `@pul.remote` 之外的代码集成

---

### Slide 27: ZeroCopy 协议

**问题**: 大 Tensor/ndarray 序列化开销大

**方案**: `__zerocopy__` 协议
```python
from pulsing.core import ZeroCopyDescriptor

class MyTensorLike:
    def __zerocopy__(self, ctx):
        return ZeroCopyDescriptor(
            buffers=[memoryview(self.buffer)],
            dtype="float32",
            shape=[1024],
            strides=[4],
            transport="inline",
            checksum=None,
            version=1,
        )
```

**自动流式传输**:
- 缓冲区 > 64KB：自动使用流式传输
- 先发送描述符 header，再分块传输数据
- 接收方预分配缓冲区，增量填充

**Fallback**: 未实现协议的对象自动使用 pickle

---

## Part 5: 因果序、一致性与会话类型 (15分钟)

> 本 Part 给出 Pulsing 通信模型背后的形式化视角：**顺序**（因果序）、**可见性**（一致性）、**协议形状**（会话类型），并明确「我们保证什么、不保证什么」。

### Slide 28: 形式化模型导引——为什么需要三个维度

**为何这三个问题重要：通信更本质的一层**

协议格式（JSON/Protobuf）、传输层（TCP/HTTP/2）、API 形态（ask/tell）都是**表象**——它们决定「怎么发、怎么收」。但**分布式通信更本质**要回答的是三件事，它们直接决定系统能提供什么保证、不能提供什么，与具体实现无关：

| 本质问题 | 形式化对应 | 为何是「本质」 |
|----------|------------|----------------|
| **顺序** | 因果序 / happens-before | 通信一旦异步、多路、可重试，「谁先谁后」就不可回避；顺序错了，语义就错（重复执行、乱序生效）。 |
| **可见性** | 一致性模型 | 多副本、多节点下，「谁能看到什么状态」决定了读写的可推理性；选强一致还是最终一致，是系统边界而非实现细节。 |
| **对话形状** | 会话类型 | 两方协作时，「允许的交互顺序」就是协议；违反形状就会卡死、错配或未定义行为，这是通信本身的约束。 |

不先把这三件事说清，讨论「用 HTTP 还是 gRPC」「用 Actor 还是 RPC」都容易停留在表面；**说清了顺序、可见性、对话形状，再去看 Pulsing 的 API 和实现，才能理解我们承诺了什么、没承诺什么。**

---

**分布式里三类常见坑**（先问题，再名词）:

| 你会遇到的坑 | 对应的形式化概念 | 一句话 |
|--------------|------------------|--------|
| 重试导致同一条请求被执行两次、或乱序生效 | **因果序 / happens-before** | 「谁先谁后」有没有保证、重放是否安全 |
| 多副本下「先写后读」读不到、或看到旧值 | **一致性模型** | 「能看到什么状态」——强一致、最终一致、因果一致等 |
| 客户端漏调一步或调错顺序，服务端卡死或协议错配 | **会话类型** | 「对话的形状」——请求/响应/流式是否按约定进行 |

**本 Part 结构**:
1. **因果序**：单 Actor FIFO、跨 Actor 无全局序、若要全局序怎么自己做
2. **一致性**：当前单副本、多副本时的可选模型、Pulsing 的承诺
3. **会话类型**：Ask/Stream/Tell 的协议形状、当前是约定而非类型检查

---

### Slide 29: 因果序（Causal Order / Happens-Before）

**定义（直觉）**: 事件之间的「先后」关系——若 A 在 B 之前发生（或 A 可能影响 B），则 A happens-before B。

**Pulsing 的承诺**:

| 范围 | 保证 | 不保证 |
|------|------|--------|
| **单 Actor 内** | 同一 Actor 的 Mailbox 严格 **FIFO**；同一调用内同步/异步/流式按语义顺序执行 | — |
| **跨 Actor** | 无全局因果序；不同 Actor 上的两件事没有「先后」保证 | 需要跨 Actor 顺序时，自己带序列号、逻辑时钟或向量时钟 |

**典型场景**:
- **单 Actor 顺序**：同一推理 Worker 上，请求 1 处理完再处理请求 2，不会乱序。
- **跨 Actor**：Worker A 先写、Worker B 后读，B 不保证「看到 A 的写」；若要保证，在业务层用版本号/时间戳或单一写者。

**重试与幂等**：因果序不保证「只执行一次」；重试可能导致同一逻辑请求执行多次，业务层需做幂等或幂等键。

---

### Slide 30: 一致性（Consistency）

**定义（直觉）**: 多个副本或多次读时，「能看到什么状态」——强一致、最终一致、因果一致等。

**Pulsing 当前模型**:

| 维度 | 当前状态 | 说明 |
|------|----------|------|
| **副本** | **单副本** | 每个 Actor 实例独占一份状态，无多副本复制 |
| **状态位置** | **进程内内存** | KV Cache、模型权重、Agent 记忆常驻进程，不依赖外部存储做一致性 |
| **多副本一致性** | **不适用** | 若未来做多副本（如同一 name 多实例读多写），需显式选模型（线性化 / 最终一致 / 因果一致等） |
| **持久化** | **未承诺** | Checkpoint/Restore 若做，恢复后的可见性需单独约定 |

**和 K8s/LLM 的对应**:
- 单副本 + 进程内 = 无需 Etcd/Redis 存会话，和「有状态 Pod」心智一致。
- 训练里每个 rank 一份权重、推理里每实例一份 KV Cache，都是「单副本状态」；Pulsing 的 Actor 状态同理。

---

### Slide 31: 会话类型（Session Types）

**定义（直觉）**: 描述「对话的形状」——谁在什么时候发什么类型的消息、收什么类型的响应；若违反形状则协议错误（如 REQ 连发两条不 recv）。

**Pulsing 中的「会话形状」**:

| 抽象 | 形状（直觉） | 实现 |
|------|--------------|------|
| **Ask** | 一发一收：请求 → 响应（或错误） | HTTP/2 上 POST + 等待 body |
| **Stream** | 一发多收：请求 → chunk₁, chunk₂, … , end | 同连接上多帧，带 END/ERROR 标志 |
| **Tell** | 一发零收：消息 → 无响应 | POST 202 Accepted，不等待 body |

**当前边界**:
- **没有用类型系统写死**：没有静态的会话类型检查（如「必须先 login 再 query」），不会在编译期或运行时强制协议顺序。
- **约定 + 文档**：Ask/Stream/Tell 的用法和顺序由 API 约定与文档保证；若客户端漏调或调错顺序，可能得到超时、错误或未定义行为。
- 若需要「协议形状」的强保证，需在业务层或上层框架做状态机/协议校验。

---

### Slide 32: Pulsing 保证什么、不保证什么（速查表）

| 维度 | 我们保证的 | 我们不保证的 / 需自己做的 |
|------|------------|---------------------------|
| **单 Actor 内顺序** | 同一 Actor 上消息 FIFO、同一调用内同步/异步/流式按定义执行 | 跨 Actor 的全局因果序（要的话自己加序列号或逻辑时钟） |
| **状态与副本** | 单副本、进程内状态常驻（KV/模型/Agent 记忆） | 多副本线性化、跨节点一致性、自动故障迁移状态 |
| **发现与组网** | 内建 Gossip，无 Etcd；和 K8s Service 配合即可 | 不替代 K8s：调度、资源、网络策略仍由 K8s 管 |
| **流式与背压** | 流式 + HTTP/2 窗口自然背压，不写限流也能防 OOM | 业务层「谁可以取消流、超时怎么算」需自己约定 |
| **与训练/推理栈** | Ray 调度 + Pulsing 通信可并存；数据面继续用 NCCL/MPI | 不替代 DataLoader/NCCL；训练数据面仍用现有框架 |
| **会话/协议** | Ask/Stream/Tell 有明确语义与实现 | 无静态会话类型检查；协议顺序靠约定与文档 |

**若被问「有没有理论保证」**: 有清晰的行为约定（顺序、失败、背压），理论对应因果序/一致性/会话类型；形式化写成定理是后续工作。

---

**设计延伸：Actor 模型天然允许「按 Actor 差异化」**

因果序、一致性、会话类型这三个本质维度，可以**按 Actor（或按通信对）** 选不同的保证，而不是全系统一刀切：

| 维度 | 按 Actor 差异化的含义 | 示例 |
|------|------------------------|------|
| **因果序** | 有的 Actor 只保证 FIFO；有的可配置为参与因果广播或全局序 | 推理 Worker 严格 FIFO；日志聚合 Actor 可接受乱序或仅最终序 |
| **一致性** | 有的 Actor 单副本强一致；有的多副本最终一致或因果一致 | 配置中心强一致；指标采集最终一致即可 |
| **对话形式** | 有的 Actor 只暴露 Ask；有的只暴露 Tell；有的 Ask+Stream | 计费只 Tell；查询只 Ask；LLM 服务 Ask+Stream |

**为什么 Actor 适合做这种分化**：每个 Actor 是独立的封装单元，其 Mailbox、状态、对外接口都可以单独约定。同一系统里，A 用「FIFO + 单副本 + Ask」、B 用「乱序 + 多副本最终一致 + Tell」，在模型上不冲突，只是当前 Pulsing 实现是**全系统统一**（单 Actor FIFO、单副本、Ask/Stream/Tell 都支持）。若未来要做「按 Actor 或按 channel 配置强弱保证」，形式化三维度正好是配置项，而不是事后补概念。

---

## Part 6: 总结与讨论 (5分钟)

### Slide 33: 核心设计回顾

**通信范式演进**:
> MPI(消除) → ZMQ(暴露) → RPC(掩盖) → Actor(内化)

**Pulsing 通信模型**:
> 同步(<10ms) → 异步(I/O) → 流式(LLM) → tell(日志)

**架构设计**:
> Gossip 组网 + HTTP/2 传输 + 背压传导（Flow Control）+ 有状态编排

**API 取舍**:
> 简洁(装饰器+await) + 类型友好(Typed Proxy) + 生态融合(Ray)

**形式化与边界**（Part 5）:
> 因果序(单 Actor FIFO) + 一致性(单副本进程内) + 会话类型(Ask/Stream/Tell 约定)；保证/不保证 见 Slide 32 速查表

---

### Slide 34: 可能面临的挑战与应对

**挑战**（技术、生态、预期）:

| 挑战 | 说明 | 应对思路 |
|------|------|----------|
| **与 Ray 的定位重叠** | 已有 Ray 做调度+通信，为何再要 Pulsing？ | 明确互补：Ray 强在调度与资源；Pulsing 强在通信语义（因果序/一致性/会话类型可讨论、可配置）。Ray 上可 `pul.mount()` 用 Pulsing 做进程间会话，二者并存。 |
| **生态与心智** | 团队习惯 RPC/微服务，Actor 需要心智切换 | 文档与示例从「问题」出发（顺序、可见性、协议形状），再给 API；受众提示让 K8s/LLM 同学先建立「和我的工作有什么关系」。 |
| **性能预期** | 控制面消息量远小于数据面，但有人会拿 Pulsing 和 NCCL 比吞吐 | 明确边界：Pulsing 不替代 NCCL；张量搬运仍用 MPI/NCCL，Pulsing 管「谁在何时与谁通信」。大 Tensor 用 ZeroCopy 协议减序列化，不碰 AllReduce 数据面。 |
| **可观测与运维** | 分布式 Actor 的链路、故障边界如何与现有监控对接 | 开放问题；当前可打点（日志/指标）与现有 Prometheus/Jaeger 集成；若需「会话级」可观测，后续在形式化边界上做 trace 设计。 |
| **持久化与多副本** | 单副本、进程内状态，故障迁移与多副本线性化未承诺 | 在速查表中写清「不保证」；若业务需要，在业务层做 checkpoint 或选型多副本存储；未来可在「按 Actor 差异化」下做可选一致性模型。 |

**总结**：挑战通过**定位清晰**（控制面 vs 数据面、与 Ray/NCCL 互补）、**文档与形式化**（保证/不保证、为何重要）、以及**分阶段实现**（先单副本与约定，再按需加 RDMA/持久化/可观测）来应对。

---

### Slide 35: 为何不讲 RDMA/NCCL？如何配合？

**为何本文不展开 RDMA / NCCL**

- **分工不同**：RDMA、NCCL 解决的是**数据面**——大块张量/梯度的搬运、AllReduce、集合通信，追求极致带宽与延迟，拓扑与参与者在训练前就固定。Pulsing 解决的是**控制面**——谁在哪个节点、何时发起调用、流式与背压、有状态会话，参与者和拓扑可动态变化。
- **不替代、不重叠**：本文讲的是「控制面通信模型」，所以重点放在 HTTP/2、Gossip、Ask/Stream/Tell、因果序/一致性/会话类型。RDMA/NCCL 在 Part 1 作为「数据面代表」出现，是为了对比代际与定位，而不是要替代它们。
- **若做 RDMA**：开放问题里已列「RDMA 传输层（HPC 场景）」——若未来在 HPC 集群提供 RDMA 传输，也是用于**承载 Actor 消息**（控制面），不是用来做 AllReduce；张量搬运仍交给 NCCL/MPI。

**如何与 RDMA/NCCL 配合**

| 场景 | 用谁 | 说明 |
|------|------|------|
| **训练：梯度/参数同步** | NCCL / MPI | 数据面，Pulsing 不介入。 |
| **训练：谁在哪个 rank、何时开始 step、checkpoint 协调** | Pulsing（或现有编排） | 控制面，可用 Actor 做协调器、状态同步。 |
| **推理：token 流式、路由、背压** | Pulsing | 控制面，Pulsing 主战场。 |
| **推理：单机内 GPU 间大块拷贝** | CUDA/NCCL（若需要） | 数据面，与 Pulsing 正交。 |

一句话：**数据面用 RDMA/NCCL/MPI，控制面用 Pulsing；同一条链路里可以上层走 Pulsing 会话、下层某段用 RDMA 承载，但职责分开。**

---

### Slide 36: 技术讨论

**开放问题**:
1. Actor 监督树的完整实现（Erlang 风格）
2. RDMA 传输层（HPC 场景）
3. 与更多框架的集成（Dify, LlamaIndex）
4. 持久化 Actor 状态（Checkpoint/Restore）

**需要贡献的领域**:
- Rust 核心优化
- Python 示例和文档
- 性能基准测试
- 生产场景验证

---

### Slide 37: 参考资源

**文档**:
- docs/src/guide/communication_patterns.zh.md
- docs/src/design/cluster-communication-evolution.zh.md
- docs/src/design/http2-transport.zh.md
- docs/src/design/node-discovery.zh.md

**代码**:
- crates/pulsing-actor: Rust 核心
- crates/pulsing-py: Python 绑定
- python/pulsing: Python API

**示例**:
- examples/agent: Agent 框架集成
- examples/python: 基础用法

---

## 附录：详细代码示例

### A. 完整 Actor 生命周期示例

```python
from pulsing.actor import Actor, ActorId

class MyActor(Actor):
    def on_start(self, actor_id: ActorId):
        print(f"Started: {actor_id}")

    def on_stop(self):
        print("Stopping...")

    def metadata(self) -> dict[str, str]:
        return {"type": "worker", "version": "1.0"}

    async def receive(self, msg):
        return msg
```

### B. Rust Behavior API

```rust
fn counter(init: i32) -> Behavior<i32> {
    stateful(init, |count, n, _ctx| {
        *count += n;
        BehaviorAction::Same
    })
}

let counter = system.spawn(counter(0)).await?;
```

### C. 重启策略配置

```python
@pul.remote(
    restart_policy="on_failure",  # "never" | "on_failure" | "always"
    max_restarts=3,
    min_backoff=0.1,
    max_backoff=30.0,
)
class ResilientWorker:
    def process(self, data):
        return heavy_computation(data)
```
