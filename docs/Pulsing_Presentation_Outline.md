# Pulsing 对外技术分享 PPT 大纲

> **目标**: 向开发者介绍 Pulsing 的设计思路、适用场景和当前状态
> **风格**: 技术导向，直接说明问题和解决方案，避免过度包装

---

## Slide 1: 封面

**标题**: Pulsing — 分布式 Actor 运行时

**副标题**: 为 Python 设计的轻量级分布式通信框架

**一句话说明**:
```
用 Rust 实现的 Actor 运行时，提供 Python API。
用于构建分布式 AI 应用，无需外部依赖。
```

---

## Slide 2: 背景 —— 我们在解决什么问题

**观察**: AI 应用正在从单机走向分布式

- 多 Agent 系统需要跨进程/跨机器通信
- LLM 推理服务需要多个 GPU 协同
- 开发环境和生产环境的部署模式差异大

**现有方案的问题**:

| 方案 | 问题 |
|------|------|
| 微服务全家桶 | 太重，需要维护 etcd/Redis 等基础设施 |
| 传统 RPC | 不支持流式，有状态服务管理麻烦 |
| Ray | 缺少原生流式，Actor 发现依赖 GCS |

**我们的切入点**:
- 轻量级：零外部依赖，单一二进制
- 流式优先：为 LLM token 生成场景设计
- 简单易用：本地和远程调用 API 一致

---

## Slide 3: Pulsing 是什么

**定义**: 分布式 Actor 运行时

**核心特性**:

1. **零外部依赖**
   - 纯 Rust + Tokio 实现
   - 不需要 etcd、Redis、NATS
   - `pip install pulsing` 即可使用

2. **Streaming-first**
   - HTTP/2 多路复用
   - 原生支持流式请求/响应
   - 自动背压控制

3. **内置服务发现**
   - Gossip/SWIM 协议
   - 节点自动加入和故障检测
   - 单端口部署

**适用场景**:
- 分布式 LLM 推理服务
- 多 Agent 协作系统
- 需要流式通信的分布式应用

---

## Slide 4: 快速上手

**安装**:
```bash
pip install pulsing
```

**基础示例**:
```python
import pulsing as pul

@pul.remote
class Greeter:
    def greet(self, name: str) -> str:
        return f"Hello, {name}!"

async def main():
    await pul.init()
    greeter = await Greeter.spawn()
    print(await greeter.greet("World"))
    await pul.shutdown()
```

**关键点**:
- `@pul.remote` 将类转为 Actor
- `spawn()` 创建实例
- `await` 直接调用，不需要 `.remote()` 或 `ray.get()`

---

## Slide 5: 从单机到分布式

**代码完全不变，只改启动参数**:

**单机模式**:
```python
await pul.init()
actor = await MyActor.spawn(name="worker")
```

**分布式模式**:
```python
# Node 1
await pul.init(addr="0.0.0.0:8000")
await MyActor.spawn(name="worker")

# Node 2
await pul.init(addr="0.0.0.0:8001", seeds=["node1:8000"])
actor = await MyActor.resolve("worker")  # 跨节点调用
result = await actor.process(data)
```

**位置透明**: 调用代码完全一致，不感知 Actor 位置

---

## Slide 6: 流式通信

**场景**: LLM token 生成，需要逐字返回

**实现**:
```python
@pul.remote
class LLMWorker:
    async def generate(self, prompt: str):
        for token in self.model.stream(prompt):
            yield token  # Python generator 自动转为流式响应

# 消费端
async for token in worker.generate("讲个笑话"):
    print(token, end="", flush=True)
```

**技术细节**:
- HTTP/2 h2c (cleartext) 传输
- 自定义二进制帧协议，避免 JSON 开销
- 流级别背压控制

---

## Slide 7: 为什么选择 Actor 模型

**AI 系统的特点**:
- 有状态：LLM 权重、KV Cache、Agent 记忆
- 长时间运行：推理服务常驻内存
- 异步通信：请求之间不阻塞

**Actor 模型的匹配点**:
```
LLM 推理实例 → Actor (状态: 模型权重 + KV Cache)
Agent         → Actor (状态: 记忆 + 目标)
工具执行器    → Actor (接收请求，返回结果)
```

**与 RPC 的区别**:
- RPC 假设无状态，状态外置到 Redis
- Actor 状态驻留进程内，消息驱动处理

---

## Slide 8: 技术实现要点

### 1. 单一端口设计

```
Port 8000:
  ├── POST /actor/{name}    Actor 消息
  ├── POST /cluster/gossip  集群协议
  └── GET /health           健康检查
```

好处：防火墙配置简单，K8s 部署方便

### 2. HTTP/2 + 二进制帧

- h2c (HTTP/2 over cleartext)：内网无需 TLS
- Prior Knowledge 模式：省掉升级协商
- 帧格式：Length(4B) + Flags(1B) + MsgType + Data

### 3. Gossip + SWIM

```
新节点 ──Join──▶ 种子节点
      ◀─Welcome─  获取成员列表

周期性 Gossip (200ms)：与随机节点同步状态
故障检测：Ping → Suspect → Dead
```

### 4. Rust Core + PyO3

- 核心路径用 Rust：网络、序列化、Gossip
- Python 仅做绑定层：业务逻辑保持 Pythonic

---

## Slide 9: 实际应用场景

### 场景1: 分布式 LLM 推理

```bash
# 启动 Router
pulsing actor pulsing.serving.Router \
  --addr 0.0.0.0:8000 --http_port 8080

# 启动 Worker (自动注册)
pulsing actor pulsing.serving.VllmWorker \
  --model Qwen/Qwen2.5-0.5B \
  --addr 0.0.0.0:8001 \
  --seeds 127.0.0.1:8000
```

- OpenAI 兼容 API
- Worker 自动发现和负载均衡
- 支持流式输出

### 场景2: 多 Agent 协作

```python
@pul.remote
class Researcher:
    async def analyze(self, topic: str) -> str:
        return await llm.invoke(f"分析: {topic}")

# 跨机器部署，调用方式不变
researcher = await Researcher.spawn(name="researcher")
analysis = await researcher.analyze("量子计算")
```

### 场景3: 与 Ray 协作

```python
@ray.remote
class Worker:
    def __init__(self, name):
        pul.mount(self, name=name)  # Ray 调度 + Pulsing 通信
```

---

## Slide 10: 与相关项目的对比

| 项目 | 定位 | 差异 |
|------|------|------|
| **Ray** | 分布式计算框架 | Ray 擅长调度，Pulsing 补全通信层（流式、发现）。可协作使用。 |
| **Temporal** | 工作流编排 | Temporal 专注持久化工作流，Pulsing 专注实时通信。 |
| **Dapr** | 微服务构建块 | Dapr 需要 sidecar，Pulsing 零依赖。 |
| **gRPC** | RPC 框架 | gRPC 无状态，Pulsing 有状态 Actor。 |

**我们的定位**:
- 不做任务调度（用 Ray）
- 不做工作流持久化（用 Temporal）
- 专注：轻量级 Actor 通信 + 流式 + 零依赖

---

## Slide 11: 项目现状和路线图

**当前版本**: v0.1.x

**已实现**:
- ✅ Actor System 核心
- ✅ Gossip/SWIM 集群发现
- ✅ HTTP/2 传输层
- ✅ Python `@pul.remote` API
- ✅ 流式消息
- ✅ AutoGen/LangGraph 集成
- ✅ LLM 推理服务 (Router + Worker)
- ✅ Ray 桥接 (`pul.mount`)

**进行中**:
- 🔧 共享内存零拷贝
- 🔧 Actor 监督树
- 🔧 更多 Agent 框架集成

**规划**:
- 📋 RDMA 传输
- 📋 Kubernetes Operator
- 📋 Web UI 控制台

---

## Slide 12: 参与项目

**为什么参与**:
- 项目早期，技术决策空间大
- 真实的基础设施代码（不是 CRUD）
- Rust + Python 跨语言技术栈

**可以做什么**:

| 方向 | 内容 | 难度 |
|------|------|------|
| Rust 开发 | 核心运行时、传输层优化 | 高 |
| Python 开发 | Agent 集成、示例代码 | 中 |
| 分布式系统 | 一致性哈希、负载均衡策略 | 高 |
| 文档/教程 | 使用文档、技术博客 | 低 |

**联系方式**:
- GitHub: https://github.com/DeepLink-org/pulsing
- Issues: 功能请求、Bug 报告
- Discussions: 技术讨论

---

## Slide 13: 总结

**Pulsing 是什么**:
- 分布式 Actor 运行时
- Rust 核心 + Python API
- 零依赖、流式优先、内置发现

**适用场景**:
- 需要流式通信的分布式应用
- 多 Agent 协作系统
- 补充 Ray 的通信能力

**当前状态**:
- v0.1.x，核心功能可用
- 需要更多真实场景验证
- 欢迎试用和反馈

---

## Slide 14: 资源

**安装**:
```bash
pip install pulsing
```

**代码**:
```bash
git clone https://github.com/DeepLink-org/pulsing
```

**文档**:
- https://deeplink-org.github.io/pulsing

**Demo 建议**:
1. 启动 Router + Worker
2. curl 调用流式 API
3. 展示自动发现过程

---

## 附录: Q&A 准备

**Q: 和 Ray 的关系？**
A: 互补。Ray 管调度，Pulsing 管通信。用 `pul.mount()` 桥接。

**Q: 生产稳定性？**
A: v0.1.x，内部测试通过，但缺乏大规模生产验证。建议从非关键业务开始试用。

**Q: 为什么用 Rust？**
A: 内存安全、零成本抽象、PyO3 绑定成熟。

**Q: 性能数据？**
A: 本地调用 <1ms，跨节点 <5ms，流式吞吐 10K+ msg/s。详细 benchmark 待补充。
