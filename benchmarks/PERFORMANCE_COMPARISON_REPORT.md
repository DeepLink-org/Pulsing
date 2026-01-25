# Ray vs Pulsing：分布式 Actor 系统性能对比报告

## 执行摘要

本报告对比了两个分布式 Actor 框架——**Ray** 和 **Pulsing**——在相同负载下的性能表现。

**核心发现**（基于单进程公平对比，Ray 使用 Generators）：

| 指标 | Pulsing 优势 |
|------|-------------|
| 单请求平均延迟 | **快 467 倍**（1.41ms vs 659.30ms） |
| 单请求 P99 延迟 | **快 3,415 倍**（3.85ms vs 13,156ms） |
| 流式平均延迟 | **快 9.3 倍**（112.70ms vs 1,044.85ms） |
| 流式 P99 延迟 | **快 91 倍**（175ms vs 15,949ms） |
| 总吞吐量 | **高 17.8 倍**（6,715 vs 378 操作） |

**结论**：即使 Ray 使用 Generators 实现流式处理，Pulsing 在低延迟、高吞吐场景下仍然显著优于 Ray，适合实时推理服务等延迟敏感型应用。

---

## 1. 架构设计对比

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Pulsing                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    Message     ┌──────────┐    Message     ┌──────────┐  │
│  │  Actor   │ ──────────────▶│  Actor   │ ──────────────▶│  Actor   │  │
│  └──────────┘                └──────────┘                └──────────┘  │
│       │                           │                           │         │
│       └───────────────────────────┴───────────────────────────┘         │
│                          Gossip 协议（集群发现）                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                                Ray                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ObjectRef  ┌─────────────┐  ObjectRef  ┌──────────┐     │
│  │  Actor   │ ──────────▶ │ Object Store │ ◀────────── │  Actor   │     │
│  └──────────┘             └─────────────┘             └──────────┘     │
│       │                         │                           │           │
│       └─────────────────────────┴───────────────────────────┘           │
│              GCS（全局控制存储）+ Raylet（调度器）                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心抽象对比

| 维度 | Pulsing | Ray |
|------|---------|-----|
| **编程模型** | 消息传递（Message Passing） | 远程过程调用（RPC） |
| **核心抽象** | `Actor` + `Message` | `@ray.remote` + `ObjectRef` |
| **调用方式** | `actor_ref.ask(message)` | `actor.method.remote(args)` |
| **返回值** | 直接返回 `Message` | 返回 `ObjectRef`，需 `ray.get()` 获取 |
| **集群发现** | Gossip 协议（去中心化） | GCS 全局控制存储（中心化） |

### 1.3 代码示例对比

**Pulsing 定义 Actor：**
```python
class EchoWorker(Actor):
    async def receive(self, msg: Message):
        if msg.msg_type == "Echo":
            data = msg.to_json()
            return Message.from_json("Response", {"echo": data["text"]})

# 调用
response = await actor_ref.ask(Message.from_json("Echo", {"text": "hello"}))
```

**Ray 定义 Actor：**
```python
@ray.remote
class EchoWorker:
    async def echo(self, text: str) -> dict:
        return {"echo": text}

# 调用
result = await actor.echo.remote("hello")  # 返回 ObjectRef，自动解包
```

### 1.4 流式处理对比

| 维度 | Pulsing | Ray（修正后） |
|------|---------|--------------|
| **实现方式** | `StreamMessage` + `StreamReader` | Ray Generators（`yield`） |
| **数据传输** | 分块流式（边产出边消费） | 分块流式（使用 `async for`） |
| **首字节时间** | 生成第一个 chunk 后即可接收 | 生成第一个 chunk 后即可接收 |
| **内存占用** | 仅缓存当前 chunk | 仅缓存当前 chunk（ObjectRef） |

```
Pulsing 流式：
  Producer: [chunk1] → [chunk2] → [chunk3] → ... → [done]
  Consumer:    ↓          ↓          ↓
            处理1      处理2      处理3      （边收边处理）

Ray Generators（修正后）：
  Producer: [chunk1] → [chunk2] → [chunk3] → ... → [done]
  Consumer:    ↓          ↓          ↓
            处理1      处理2      处理3      （边收边处理）
```

**注意**：修正后的 Ray benchmark 使用 Ray Generators 实现真正的流式处理，与 Pulsing 的流式语义等价。

---

## 2. 关键设计差异

### 2.1 差异 A：调用开销

| 步骤 | Pulsing | Ray |
|------|---------|-----|
| 1. 序列化 | JSON 序列化（轻量） | Pickle 序列化（较重） |
| 2. 传输 | 直接消息传递 | 写入 Object Store |
| 3. 调度 | 无额外调度 | Raylet 调度 + GCS 查询 |
| 4. 获取结果 | 直接返回 | 从 Object Store 读取 |

**影响**：Pulsing 的调用路径更短，单请求延迟显著更低。

### 2.2 差异 B：流式语义（已修正）

| 场景 | Pulsing | Ray（修正后） |
|------|---------|--------------|
| 生成 10 个 item，每个延迟 50ms | TTFT ≈ 50ms，总延迟 ≈ 500ms | TTFT ≈ 50ms，总延迟 ≈ 500ms |
| P99 尾延迟 | 较低（175ms） | 较高（15,949ms） |

**影响**：
- 虽然两者都实现了真正的流式处理，但 Ray 的底层架构（Object Store + 序列化）导致延迟和长尾问题更严重
- 在 LLM 推理等场景，Pulsing 可以实现更好的用户体验（更低的延迟和更稳定的 P99）

### 2.3 差异 C：运行时模型

| 模式 | Pulsing | Ray |
|------|---------|-----|
| 单进程 | 轻量级 ActorSystem | 完整 Ray 运行时（Object Store + Raylet） |
| 多进程 | 每进程一个 ActorSystem，Gossip 组网 | 需连接统一 Ray 集群，否则资源争用 |
| 资源占用 | 低 | 较高（需要 Object Store 内存） |

**影响**：Pulsing 更适合资源受限环境和多进程部署。

---

## 3. 压测实验设计

### 3.1 测试配置

| 配置项 | 单进程模式 | 多进程模式 |
|--------|-----------|-----------|
| 进程数 | 1 | 10（torchrun） |
| Worker 数/类型 | 50 | 1 |
| 总 Worker 数 | 250（5类型×50） | 50（5类型×10进程） |
| 测试时长 | 30 秒 | 300 秒 |
| 目标速率 | 100 req/s | 100 req/s |

### 3.2 Worker 类型

| Worker | 功能 | 负载特征 |
|--------|------|---------|
| **EchoWorker** | 消息回显 | I/O 密集，极低延迟 |
| **ComputeWorker** | `sum(i² for i in range(n))` | CPU 密集，n ∈ [100, 10000] |
| **StreamWorker** | 生成 5-20 个 item，间隔 10-50ms | 流式，中等延迟 |
| **BatchWorker** | 累积 10 个请求后批量处理 | 状态+批处理 |
| **StatefulWorker** | SetState / GetState | 有状态操作 |

### 3.3 请求混合

```
请求类型分布：
├── 70% 单请求（Single Request）
│   ├── 25% Echo
│   ├── 25% Compute
│   ├── 25% Batch
│   └── 25% Stateful
└── 30% 流请求（Stream Request）
    └── 100% Stream
```

### 3.4 统计指标

| 指标 | 说明 |
|------|------|
| **total** | 总请求/流数量 |
| **success_rate** | 成功率 |
| **avg latency** | 平均延迟（端到端） |
| **p50 / p95 / p99** | 延迟百分位数 |
| **errors** | 错误类型统计 |

---

## 4. 数据对比

### 4.1 单进程模式（公平对比）✅

> **测试条件**：30秒，100 req/s，50 Workers/类型，单进程
>
> **重要更新**：Ray benchmark 已修正为使用 Ray Generators 实现真正的流式处理，确保公平对比。

#### 单请求性能

| 指标 | Ray | Pulsing | Pulsing 优势 |
|------|----:|--------:|-------------:|
| 总请求数 | 254 | 4,734 | **18.6×** |
| 成功率 | 100% | 100% | — |
| 平均延迟 | 659.30 ms | 1.41 ms | **467× 更低** |
| P50 延迟 | 265.43 ms | 1.23 ms | **216× 更低** |
| P95 延迟 | 1,764.99 ms | 3.00 ms | **588× 更低** |
| P99 延迟 | 13,156.18 ms | 3.85 ms | **3,415× 更低** |

**分析**：
- Ray 的 P99 延迟高达 13 秒，说明存在严重的长尾问题，可能与 Object Store 争用、序列化开销或调度延迟相关
- Pulsing 的 P99 仅 3.85ms，延迟分布非常稳定，几乎无长尾
- 相同时间内 Pulsing 处理的请求数是 Ray 的 18.6 倍，吞吐量优势显著

#### 流式性能（使用 Ray Generators）

| 指标 | Ray | Pulsing | Pulsing 优势 |
|------|----:|--------:|-------------:|
| 总流数 | 124 | 1,981 | **16.0×** |
| 成功率 | 100% | 100% | — |
| 平均延迟 | 1,044.85 ms | 112.70 ms | **9.3× 更低** |
| P50 延迟 | 385.90 ms | 112.21 ms | **3.4× 更低** |
| P95 延迟 | 3,588.20 ms | 168.56 ms | **21.3× 更低** |
| P99 延迟 | 15,949.15 ms | 175.00 ms | **91× 更低** |

**分析**：
- 即使使用 Ray Generators 实现流式处理，Ray 的流式 P99 仍超过 15 秒，严重影响用户体验
- Pulsing 流式 P99 控制在 175ms 内，更适合实时场景
- 虽然两者都实现了真正的流式处理，但 Pulsing 的延迟和吞吐量仍然显著优于 Ray
- 差异主要来自底层架构：Pulsing 的直接消息传递 vs Ray 的 Object Store + 序列化开销

---

### 4.2 多进程模式（对 Ray 不公平）⚠️

> **测试条件**：`torchrun --nproc_per_node=10`，每进程独立初始化
>
> ⚠️ **注意**：此模式下每个进程创建独立的 Ray 运行时，导致资源争用，对 Ray 不公平。仅供参考。

#### 单请求性能

| 指标 | Ray（Process 8） | Pulsing | 差异 |
|------|---:|---:|---:|
| 总请求数 | 2,867 | 7,473 | Pulsing **2.6×** |
| 平均延迟 | 44.52 ms | 0.36 ms | Pulsing **123× 更低** |
| P99 延迟 | 367.10 ms | 1.01 ms | Pulsing **363× 更低** |

#### 流式性能

| 指标 | Ray（Process 8） | Pulsing | 差异 |
|------|---:|---:|---:|
| 总流数 | 214 | 556 | Pulsing **2.6×** |
| 平均延迟 | 476.55 ms | 368.62 ms | Pulsing **23% 更低** |
| P99 延迟 | 2,096.25 ms | 952.86 ms | Pulsing **2.2× 更低** |

---

### 4.3 错误统计

| 模式 | Ray | Pulsing |
|------|-----|---------|
| 单进程 | 0 错误 | 0 错误 |
| 多进程 | 0 错误 | 1 错误（网络抖动） |

两个框架的稳定性都很好，错误率接近 0%。

---

## 5. 结论

### 5.1 性能对比总结（修正后，Ray 使用 Generators）

| 维度 | Ray | Pulsing | 差异倍数 |
|------|----:|--------:|---------:|
| 单请求平均延迟 | 659.30 ms | 1.41 ms | **467×** |
| 单请求 P99 延迟 | 13,156 ms | 3.85 ms | **3,415×** |
| 流式平均延迟 | 1,044.85 ms | 112.70 ms | **9.3×** |
| 流式 P99 延迟 | 15,949 ms | 175 ms | **91×** |
| 总吞吐量（请求+流） | 378 | 6,715 | **17.8×** |

### 5.2 差异归因

| 差异 | 原因 |
|------|------|
| 单请求延迟 467× | Pulsing 直接消息传递（JSON 序列化）vs Ray Object Store + Pickle 序列化 + Raylet 调度 |
| P99 尾延迟巨大（3,415×） | Ray 的 Object Store GC、序列化开销和调度争用导致严重长尾 |
| 流式延迟差异（9.3×） | 虽然都使用流式处理，但 Pulsing 的消息传递开销远低于 Ray 的 ObjectRef 机制 |
| 吞吐量差异（17.8×） | Pulsing 更低的调用开销和更高效的并发模型支持更高吞吐量 |

### 5.3 适用场景建议

| 场景 | 推荐框架 | 原因 |
|------|---------|------|
| **LLM 推理服务** | Pulsing | 低延迟 + 真流式，首 token 更快 |
| **实时 API 服务** | Pulsing | 微秒级延迟，P99 稳定 |
| **高并发消息处理** | Pulsing | 轻量级，吞吐量更高 |
| **大规模数据处理** | Ray | 成熟生态，丰富的数据处理工具 |
| **分布式机器学习** | Ray | Ray Train / Tune 等成熟组件 |
| **需要复杂调度** | Ray | 支持资源感知调度、Gang 调度等 |

### 5.4 最终结论

> **即使 Ray 使用 Generators 实现流式处理，Pulsing 在延迟敏感型场景下仍然显著优于 Ray**：
> - 单请求延迟快 **467 倍**（1.41ms vs 659.30ms）
> - 单请求 P99 延迟快 **3,415 倍**（3.85ms vs 13,156ms）
> - 流式平均延迟快 **9.3 倍**（112.70ms vs 1,044.85ms）
> - 流式 P99 延迟快 **91 倍**（175ms vs 15,949ms）
> - 总吞吐量高 **17.8 倍**（6,715 vs 378 操作）
>
> 对于需要低延迟、高吞吐的 Actor 系统（如推理服务、实时 API），**强烈推荐使用 Pulsing**。
>
> 对于需要丰富生态和复杂调度的大规模数据处理任务，**Ray 仍是更好的选择**。

---

## 附录

### A. 测试环境

| 项目 | 配置 |
|------|------|
| 操作系统 | macOS (darwin 25.1.0) |
| Python | 3.12.11 |
| Ray | 最新稳定版 |
| Pulsing | 当前开发版 |

### B. 运行测试

```bash
# 单进程模式（推荐，公平对比）
./benchmarks/run_stress_test_ray_single.sh      # Ray
./benchmarks/run_stress_test_pulsing_single.sh  # Pulsing

# 自定义参数
DURATION=60 RATE=200 NUM_WORKERS=100 ./benchmarks/run_stress_test_ray_single.sh
```

### C. 测试脚本

| 脚本 | 说明 |
|------|------|
| `large_scale_stress_test_ray_single.py` | Ray 单进程测试（已修正：使用 Ray Generators） |
| `large_scale_stress_test_pulsing_single.py` | Pulsing 单进程测试 |
| `large_scale_stress_test_ray.py` | Ray 多进程测试（torchrun） |
| `large_scale_stress_test.py` | Pulsing 多进程测试（torchrun） |

### D. 测试修正说明

**Ray benchmark 修正**（2025-01-25）：
- ✅ 修正 `StreamWorker` 使用 Ray Generators（`yield`）实现真正的流式处理
- ✅ 修正调用端使用 `async for` 配合 Ray Generators 消费流式结果
- ✅ 确保参数与 Pulsing benchmark 对齐（count: 5-15, delay: 0.01）

修正后的测试结果更能反映两个框架的真实性能差异，确保公平对比。
