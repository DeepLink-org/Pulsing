# AutoGen + Pulsing 示例

使用 Pulsing 作为 AutoGen 运行时，**单机/分布式同一套 API**。

## 快速开始

```bash
# 单机模式
python simple.py

# 分布式模式 (torchrun)
./run_distributed.sh

# 分布式模式 (手动)
./run_distributed.sh --manual
```

## 示例说明

| 文件 | 说明 |
|------|------|
| `simple.py` | 单机模式，对比 AutoGen 原生运行时 |
| `distributed.py` | 分布式模式，Writer/Editor/Manager 多进程协作 |
| `run_distributed.sh` | 启动脚本，支持 torchrun 和手动模式 |

## 使用方式

```python
from pulsing.integrations.autogen import PulsingRuntime

# 单机
runtime = PulsingRuntime()

# 分布式 - 种子节点
runtime = PulsingRuntime(addr="0.0.0.0:8001")

# 分布式 - 加入集群
runtime = PulsingRuntime(addr="0.0.0.0:8002", seeds=["node1:8001"])

# API 与 AutoGen 完全一致
await runtime.start()
await runtime.register_factory("agent", MyAgent)
response = await runtime.send_message(msg, recipient=AgentId("agent", "default"))
```

## 对比优势

| 特性 | AutoGen gRPC | Pulsing |
|------|-------------|---------|
| 中央协调器 | 必须 | 不需要 |
| 自动发现 | ❌ | ✅ Gossip |
| 故障转移 | ❌ | ✅ |
| K8s 部署 | 复杂 | 简单 |
