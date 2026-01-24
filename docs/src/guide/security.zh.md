# 安全指南

使用 TLS 加密保护 Pulsing 集群的指南。

## 概述

Pulsing 支持**基于口令的 mTLS（双向 TLS）**实现安全的集群通信。这种创新设计提供：

- **零配置 PKI**：无需手动生成或分发证书
- **口令即准入**：拥有相同口令的节点可以加入集群
- **集群隔离**：不同口令创建完全隔离的集群
- **双向认证**：服务端和客户端互相验证证书

!!! danger "Pickle 序列化风险（可能导致 RCE）"
    Pulsing 目前在 Python-to-Python 的消息载荷上使用 **Pickle**。
    **不要接收不受信的载荷**，也不要把 Pulsing 端口暴露在不受信网络中。

    生产建议：

    - 通过设置 `passphrase` **启用 mTLS**（任何真实部署都应开启）
    - 仍建议做 **网络隔离**（私有网段 + 防火墙）
    - 在默认序列化格式不再是 pickle 之前，把集群当作 **可信边界**

## 启用 TLS

### 开发模式（无 TLS）

默认情况下，Pulsing 使用明文 HTTP/2 (h2c) 便于调试：

```python
import pulsing as pul

# 不设置口令 - 使用明文 HTTP/2
system = await pul.actor_system(addr="0.0.0.0:8000")
```

### 生产模式（mTLS）

要启用 TLS 加密，只需设置口令：

```python
# 设置口令 - 自动启用 mTLS
system = await pul.actor_system(
    addr="0.0.0.0:8000",
    passphrase="my-cluster-secret"
)
```

## 多节点 TLS 集群

集群中的所有节点必须使用**相同口令**才能通信：

```python
# Node 1: 带 TLS 的种子节点
system1 = await pul.actor_system(
    addr="0.0.0.0:8000",
    passphrase="shared-secret"
)

# Node 2: 使用相同口令加入集群
system2 = await pul.actor_system(
    addr="0.0.0.0:8001",
    seeds=["192.168.1.1:8000"],
    passphrase="shared-secret"  # 必须匹配！
)
```

!!! warning "口令不匹配"
    使用不同口令的节点无法通信。TLS 握手将失败。

## 集群隔离

不同口令创建完全隔离的集群：

```python
# 集群 A
system_a = await pul.actor_system(addr="0.0.0.0:8000", passphrase="secret-a")

# 集群 B（不同口令）
system_b = await pul.actor_system(addr="0.0.0.0:9000", passphrase="secret-b")

# system_a 和 system_b 无法通信
```

## 工作原理

Pulsing 使用**确定性 CA 派生**方法：

```
┌─────────────────────────────────────────────────────────────┐
│                       口令 (Passphrase)                      │
│                         "my-secret"                         │
└──────────────────────────┬──────────────────────────────────┘
                           │ HKDF-SHA256
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                确定性 CA 证书                                │
│         (相同口令 → 相同 CA 证书和私钥)                      │
│         算法: Ed25519 | 有效期: 10年                         │
└──────────────────────────┬──────────────────────────────────┘
                           │ 签名
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               节点证书 (每个节点独立)                         │
│         (每个节点独立生成，由 CA 签名)                        │
│         CN: "Pulsing Node <uuid>" | SAN: pulsing.internal   │
└─────────────────────────────────────────────────────────────┘
```

### 核心特性

| 特性 | 说明 |
|------|------|
| **双向认证** | 服务端和客户端都需要提供证书 |
| **口令即准入** | 只有知道口令的节点才能加入 |
| **零配置 PKI** | 无需手动生成/分发证书 |
| **确定性 CA** | 相同口令 → 相同 CA（所有节点信任） |
| **隔离集群** | 不同口令的集群完全隔离 |

## 安全最佳实践

### 1. 使用强口令

!!! tip "口令强度"
    生产环境使用高熵随机字符串：
    ```python
    # 好：高熵值
    passphrase = "aX9#mK2$nL5@pQ8&"

    # 差：弱/可预测
    passphrase = "password123"
    ```

### 2. 使用环境变量

将口令存储在环境变量中，而不是代码中：

```python
import os
import pulsing as pul

passphrase = os.environ.get("PULSING_PASSPHRASE")

# 创建系统，可选启用 TLS
system = await pul.actor_system(
    addr="0.0.0.0:8000",
    passphrase=passphrase  # None = 无 TLS（开发模式）
)
```

### 3. 口令轮换

轮换口令的步骤：

1. 使用新口令部署新节点
2. 逐步将 Actor 迁移到新节点
3. 下线旧节点

!!! note "滚动更新"
    使用不同口令的节点无法通信。请规划短暂的过渡期。

### 4. 网络隔离

即使启用 TLS，也应使用网络级安全措施：

- 部署在私有 VPC/子网中
- 使用防火墙限制访问
- 跨数据中心通信考虑使用 VPN

## 与其他框架对比

| 方面 | Pulsing | Ray | 传统 mTLS |
|------|---------|-----|-----------|
| **配置复杂度** | 1 行代码 | 多个配置文件 | 需要 PKI 基础设施 |
| **证书管理** | 无需管理 | 需要分发证书 | 需要 CA + 证书轮换 |
| **新节点加入** | 知道口令即可 | 需要预配置证书 | 需要签发新证书 |
| **集群隔离** | 不同口令 | 不同证书体系 | 不同 CA |
| **加密算法** | Ed25519 + mTLS | TLS 1.2/1.3 | 取决于配置 |

## 当前限制

!!! warning "当前限制"
    - **无授权机制**：任何 Actor 可以调用任何 Actor（仅认证，无授权）
    - **Pickle 序列化**：消息载荷仍使用 Pickle（计划替换为 msgpack）
    - **无证书轮换**：更换口令需要重启集群

## 示例：安全分布式计数器

```python
import os
import pulsing as pul

# 从环境变量获取口令
PASSPHRASE = os.environ.get("PULSING_SECRET", None)

@pul.remote
class SecureCounter:
    def __init__(self, init_value: int = 0):
        self.value = init_value

    def get(self) -> int:
        return self.value

    def increment(self, n: int = 1) -> int:
        self.value += n
        return self.value

async def main():
    # 创建系统，可选启用 TLS
    system = await pul.actor_system(
        addr="0.0.0.0:8000",
        passphrase=PASSPHRASE
    )

    # 生成安全计数器
    counter = await SecureCounter.local(system, init_value=0)

    print("安全计数器运行中...")
    print(f"TLS 已启用: {PASSPHRASE is not None}")

    await system.shutdown()
```

## 下一步

- 了解[远程 Actor](remote_actors.zh.md) 的集群通信
- 查看 [HTTP2 传输](../design/http2-transport.zh.md) 了解传输细节
- 阅读[语义与保证](semantics.zh.md) 了解消息传递保证
