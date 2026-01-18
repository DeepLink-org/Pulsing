# Inspect CLI 示例

这个示例展示如何启动一个多节点的 Pulsing 服务，然后使用 `pulsing inspect` 命令查看集群状态、actors 分布和指标。

## 快速开始

### 方式一：一键完整演示（推荐）

运行完整的演示脚本，会自动启动服务并展示所有 inspect 命令：

```bash
./examples/inspect/demo.sh
```

这个脚本会：
1. ✅ 自动启动 3 个节点（后台运行）
2. ✅ 依次运行所有 inspect 子命令并展示输出
3. ✅ 演示 cluster、actors、metrics、watch 等功能
4. ✅ 最后保持运行，你可以手动尝试更多命令
5. ✅ 按 Ctrl+C 自动清理所有节点

**输出示例：**
- 集群状态（成员列表、状态统计）
- Actors 分布（全部、top 5、过滤结果）
- Metrics 摘要（关键指标）
- Watch 模式（实时监控演示）

### 方式二：仅启动服务（用于手动测试）

如果你只想启动服务，然后手动运行 inspect 命令：

```bash
./examples/inspect/start_demo.sh
```

这个脚本会：
- 启动 3 个节点（后台运行）
- 显示节点 PID 和日志位置
- 保持运行，等待你手动执行 inspect 命令
- 按 Ctrl+C 停止所有节点

### 方式三：完全手动（用于学习）

**1. 启动服务（3 个终端）**

**终端 1 - 节点 1（种子节点）：**
```bash
python examples/inspect/demo_service.py --port 8000
```

**终端 2 - 节点 2：**
```bash
python examples/inspect/demo_service.py --port 8001 --seed 127.0.0.1:8000
```

**终端 3 - 节点 3：**
```bash
python examples/inspect/demo_service.py --port 8002 --seed 127.0.0.1:8000
```

**2. 使用 Inspect 命令（新终端）**

#### 查看集群状态

```bash
# 查看集群成员和状态
pulsing inspect cluster --seeds 127.0.0.1:8000
```

输出示例：
```
Connecting to cluster via seeds: ['127.0.0.1:8000']...

Cluster Status: 3 total nodes (3 alive)
================================================================================

Status Summary:
  Alive: 3

Node ID               Address                         Status
--------------------------------------------------------------------------------
1                     127.0.0.1:8000                 Alive
2                     127.0.0.1:8001                 Alive
3                     127.0.0.1:8002                 Alive

================================================================================
```

#### 查看 Actors 分布

```bash
# 查看所有 actors 的分布
pulsing inspect actors --seeds 127.0.0.1:8000

# 只看 top 5
pulsing inspect actors --seeds 127.0.0.1:8000 --top 5

# 过滤特定类型的 actors
pulsing inspect actors --seeds 127.0.0.1:8000 --filter worker
```

输出示例：
```
Connecting to cluster via seeds: ['127.0.0.1:8000']...
Found 3 alive nodes

Actor Distribution (5 unique actors):
================================================================================
Actor Name                              Instances    Nodes
--------------------------------------------------------------------------------
workers/worker-1                       1            1
workers/worker-2                       1            1
workers/worker-3                       1            2
workers/worker-4                       1            2
services/router                        1            1
services/cache                         1            3

================================================================================
Total: 5 unique actors, 6 instances
Across 3 nodes
```

#### 查看 Metrics

```bash
# 查看原始 Prometheus 指标
pulsing inspect metrics --seeds 127.0.0.1:8000

# 只看关键指标摘要
pulsing inspect metrics --seeds 127.0.0.1:8000 --raw False
```

#### 实时监控（Watch 模式）

```bash
# 监控集群变化
pulsing inspect watch --seeds 127.0.0.1:8000 --kind cluster --interval 2.0

# 监控 actors 变化
pulsing inspect watch --seeds 127.0.0.1:8000 --kind actors --interval 1.0

# 监控所有变化
pulsing inspect watch --seeds 127.0.0.1:8000 --kind all --interval 1.0
```

输出示例：
```
Watching cluster via seeds: ['127.0.0.1:8000']...
Refresh interval: 1.0s, Watching: cluster
Press Ctrl+C to stop

[14:30:15] Initial state: 3/3 nodes alive

[14:30:16] No changes

[14:30:17] Changes detected:
  • Node 2: Alive -> Suspect

[14:30:18] Changes detected:
  • Node 2: Suspect -> Alive
```

## 服务架构

这个示例服务包含：

- **Node 1 (8000)**：
  - `services/router` - 路由服务
  - `workers/worker-1` - 工作节点 1
  - `workers/worker-2` - 工作节点 2

- **Node 2 (8001)**：
  - `workers/worker-3` - 工作节点 3
  - `workers/worker-4` - 工作节点 4

- **Node 3 (8002)**：
  - `services/cache` - 缓存服务

## Inspect 命令参考

### 通用选项

所有子命令都支持：

- `--seeds <addresses>` - 逗号分隔的种子节点地址（必需）
- `--timeout <seconds>` - 请求超时（默认：10.0）
- `--best_effort True` - 即使部分节点失败也继续（默认：False）

### Cluster 子命令

```bash
pulsing inspect cluster --seeds 127.0.0.1:8000
```

显示：
- 集群成员总数和存活数
- 按状态分组的统计
- 每个节点的详细信息（ID、地址、状态）

### Actors 子命令

```bash
pulsing inspect actors --seeds 127.0.0.1:8000 [options]
```

选项：
- `--top N` - 只显示前 N 个 actors（按实例数排序）
- `--filter STR` - 按名称子串过滤
- `--all_actors True` - 包含内部/系统 actors

显示：
- Actor 名称
- 实例数量
- 分布节点列表

### Metrics 子命令

```bash
pulsing inspect metrics --seeds 127.0.0.1:8000 [options]
```

选项：
- `--raw True` - 输出原始 Prometheus 格式（默认）
- `--raw False` - 只显示关键指标摘要

### Watch 子命令

```bash
pulsing inspect watch --seeds 127.0.0.1:8000 [options]
```

选项：
- `--interval <seconds>` - 刷新间隔（默认：1.0）
- `--kind <type>` - 监控类型：`cluster`、`actors`、`metrics`、`all`（默认：`all`）
- `--max_rounds N` - 最大刷新轮数（None = 无限）

## 演示脚本说明

### `demo.sh` - 完整演示脚本

自动完成以下步骤：

1. **启动服务**：在后台启动 3 个节点
2. **Inspect Cluster**：展示集群成员和状态
3. **Inspect Actors**：展示 actors 分布（全部、top 5、过滤）
4. **Inspect Metrics**：展示关键指标摘要
5. **Watch Mode**：演示实时监控（3 轮）

脚本会在最后保持运行，你可以：
- 手动运行更多 inspect 命令
- 查看日志文件：`/tmp/pulsing_node*.log`
- 按 Ctrl+C 自动停止所有节点并退出

**提示**：如果你想跳过自动演示，直接手动测试，可以在演示结束后（脚本保持运行时）直接运行你想要的 inspect 命令。

## 实验建议

1. **运行完整演示**
   ```bash
   ./examples/inspect/demo.sh
   ```

2. **手动尝试不同的 inspect 命令**
   ```bash
   pulsing inspect cluster --seeds 127.0.0.1:8000
   pulsing inspect actors --seeds 127.0.0.1:8000 --top 3
   pulsing inspect metrics --seeds 127.0.0.1:8000 --raw False
   ```

3. **使用 watch 模式观察变化**
   - 启动 watch 后，停止一个节点（Ctrl+C），观察状态变化
   - 重新启动节点，观察恢复过程

4. **测试过滤和排序**
   ```bash
   # 只看 worker 类型的 actors
   pulsing inspect actors --seeds 127.0.0.1:8000 --filter worker

   # 只看服务类型的 actors
   pulsing inspect actors --seeds 127.0.0.1:8000 --filter router
   ```

5. **测试容错**
   ```bash
   # 使用 best_effort，即使部分节点失败也继续
   pulsing inspect actors --seeds 127.0.0.1:8000,127.0.0.1:9999 --best_effort True
   ```

## 故障排查

### 无法连接到集群

- 确保至少一个种子节点正在运行
- 检查端口是否正确（默认 8000）
- 尝试使用 `--timeout` 增加超时时间

### 看不到某些 actors

- 使用 `--all_actors True` 查看所有 actors（包括系统 actors）
- 检查 actor 是否真的创建成功（查看服务日志）

### Watch 模式没有变化

- 确保 `--interval` 设置合理（不要太短，避免频繁请求）
- 尝试不同的 `--kind` 选项

## 下一步

- 查看 [CLI Operations 文档](../../docs/src/guide/operations.md) 了解更多 CLI 命令
- 尝试其他示例：[examples/README.md](../README.md)
- 学习如何创建自己的 actors：[examples/python/](../python/)
