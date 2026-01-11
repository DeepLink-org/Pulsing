# Actor 列表（CLI）

`pulsing actor list` 是一个轻量的 **观察者（observer）** 工具：通过 HTTP 端点查询 actor 列表，**无需加入 gossip 集群**。

## 适用场景

- 验证节点是否可达
- 查看某个节点上有哪些 actors
- 快速了解集群概况（从 seeds 发现并逐个查询）

## 单节点查询

```bash
pulsing actor list --endpoint 127.0.0.1:8000
```

### 显示系统/内部 actors

```bash
pulsing actor list --endpoint 127.0.0.1:8000 --all_actors True
```

### JSON 输出

```bash
pulsing actor list --endpoint 127.0.0.1:8000 --json True
```

## 集群查询（通过 seeds）

```bash
pulsing actor list --seeds 127.0.0.1:8000,127.0.0.1:8001
```

## 说明

- 该命令通过 HTTP/2（h2c）请求实现，不需要加入集群。
- 如果节点未暴露相关 HTTP 端点，会显示为不可连接。

