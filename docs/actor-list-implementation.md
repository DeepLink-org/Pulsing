# Pulsing Actor List 完整实现总结

!!! note "文档迁移"
    本页偏实现细节，面向用户的最新版已迁移到文档站点的 **Guide**：
    - `docs/src/guide/actor_list.zh.md`
    - `docs/src/guide/actor_list.md`
    - `docs/src/guide/operations.zh.md`
    - `docs/src/guide/operations.md`

## ✅ 已完成功能

### 1. 本地查询模式
在运行 actor system 的进程内查询 actors：

```bash
# 在应用代码中
from pulsing.cli.actor_list import list_actors_impl
await list_actors_impl()

# 或在同一进程中作为 Python API
from pulsing.actor import get_system
names = get_system().local_actor_names()
```

**功能：**
- ✅ 列出用户 actors（默认）
- ✅ 列出所有 actors（`--all_actors True`）
- ✅ 显示 Python 类名（如 `__main__.Counter`）
- ✅ 显示代码路径（如 `/path/to/file.py`）
- ✅ 表格和 JSON 输出格式

### 2. 远程查询模式
从外部连接到远程集群并查询 actors：

```bash
# 查询整个集群
pulsing actor list --list_seeds "127.0.0.1:8000"

# 查询特定节点
pulsing actor list --list_seeds "127.0.0.1:8000" --node_id 12345

# JSON 输出
pulsing actor list --list_seeds "127.0.0.1:8000" --json True
```

**功能：**
- ✅ 通过 seeds 连接远程集群
- ✅ 自动发现集群中的所有节点
- ✅ 查询每个节点的 actors
- ✅ 显示节点状态和响应性
- ✅ 支持查询特定节点（`--node_id`）

## 实现架构

### 组件层次

```
┌─────────────────────────────────────────┐
│  CLI: pulsing actor list                │
│  (python/pulsing/cli/__main__.py)       │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  list_actors_command()                  │
│  (python/pulsing/cli/actor_list.py)    │
│  - 解析参数                              │
│  - 选择本地/远程模式                     │
└────────────┬────────────────────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
┌───────────┐  ┌──────────────────┐
│ 本地模式   │  │ 远程模式          │
│           │  │                  │
│ get_      │  │ create_actor_    │
│ system()  │  │ system(seeds)    │
│           │  │                  │
│ local_    │  │ all_named_       │
│ actor_    │  │ actors()         │
│ names()   │  │                  │
└───────────┘  └──────────────────┘
      │              │
      ▼              ▼
┌─────────────────────────────────────────┐
│  Metadata Registry                      │
│  (_actor_metadata_registry)             │
│  - Python class name                    │
│  - Source file path                     │
│  - Module name                          │
└─────────────────────────────────────────┘
```

### 关键代码位置

1. **Rust 侧元信息提取** (`crates/pulsing-py/src/actor.rs`)
   ```rust
   impl Actor for PythonActorWrapper {
       fn metadata(&self) -> HashMap<String, String> {
           // 自动提取 __class__, __module__, __file__
       }
   }
   ```

2. **Python 侧元信息注册** (`python/pulsing/actor/remote.py`)
   ```python
   def _register_actor_metadata(name: str, cls: type):
       """在创建 actor 时注册类型信息"""

   def get_actor_metadata(name: str) -> dict[str, str] | None:
       """查询 actor 的元信息"""
   ```

3. **CLI 实现** (`python/pulsing/cli/actor_list.py`)
   - `list_actors_impl()`: 核心查询逻辑
   - `_list_remote_node_actors()`: 远程节点查询
   - `_print_actors_output()`: 格式化输出

## 输出示例

### 本地查询（表格格式）
```
Name                           Type            Class                               Code Path
----------------------------------------------------------------------------------------------------------------------------------
counter-1                      user            __main__.Counter                    /tmp/demo.py
counter-2                      user            __main__.Counter                    /tmp/demo.py
calculator                     user            __main__.Calculator                 /tmp/demo.py

Total: 3 actor(s)
```

### 远程查询（多节点）
```
Connecting to cluster via seeds: ['127.0.0.1:9001']...
Found 2 nodes in cluster

================================================================================
Node 12345 (127.0.0.1:9001) - Status: Alive
================================================================================
  Node is responsive (ping: 1234567890)
  Name                           Type            Class                               Code Path
  ----------------------------------------------------------------------------------------------------------------------------------
  service-a-1                    user            -                                   -
  service-a-2                    user            -                                   -

  Total: 2 actor(s)

================================================================================
Node 67890 (127.0.0.1:9002) - Status: Alive
================================================================================
  Node is responsive (ping: 1234567891)
  Name                           Type            Class                               Code Path
  ----------------------------------------------------------------------------------------------------------------------------------
  service-b-1                    user            -                                   -
  service-b-2                    user            -                                   -
  service-b-3                    user            -                                   -

  Total: 3 actor(s)
```

## 使用场景

### 场景 1: 开发调试
在应用内部快速查看创建了哪些 actors：

```python
from pulsing.actor import init, remote, get_system
from pulsing.cli.actor_list import list_actors_impl

await init()
# ... 创建 actors ...

# 查看当前 actors
await list_actors_impl()
```

### 场景 2: 运维监控
从外部查看生产集群的 actors 分布：

```bash
# 查看整个集群
pulsing actor list --list_seeds "prod-node-1:8000"

# 查看特定节点
pulsing actor list --list_seeds "prod-node-1:8000" --node_id 12345

# 导出为 JSON 供监控系统使用
pulsing actor list --list_seeds "prod-node-1:8000" --json True > actors.json
```

### 场景 3: 集群诊断
结合 `pulsing inspect` 使用，全面了解集群状态：

```bash
# 先查看集群拓扑
pulsing inspect --seeds "127.0.0.1:8000"

# 再查看详细的 actor 列表
pulsing actor list --list_seeds "127.0.0.1:8000" --all_actors True
```

## 局限性和未来改进

### 当前局限

1. **远程元信息缺失**：查询远程节点时，无法获取 Python 类名和代码路径
   - 原因：metadata 存储在本地进程内存中
   - 影响：远程查询只能看到 actor 名字

2. **Uptime 精度**：当前显示的是系统 uptime，不是单个 actor 的创建时间
   - 原因：ActorRegistry 存储创建时间，但 local_actor_names() 不返回

3. **性能**：查询大集群时需要逐个 ping 节点
   - 可能的优化：并发查询、缓存结果

### 建议改进（优先级从高到低）

- [ ] **P1**: 在 Rust 的 ActorRegistry 中存储并返回 metadata
  - 让远程查询也能看到类型信息

- [ ] **P2**: 添加每个 actor 的精确 uptime
  - 修改 `local_actor_names()` 返回更详细信息

- [ ] **P2**: 添加消息统计（处理量、错误率等）
  - 从 metrics 系统获取

- [ ] **P3**: 支持过滤和搜索
  - 按名称、类型、节点等过滤

- [ ] **P3**: 交互式模式（实时刷新）
  - 类似 `top` 命令的体验

## 测试

```bash
# 运行测试
cd /Users/reiase/workspace/Pulsing
PYTHONPATH=python pyenv exec python -m pytest tests/python/test_actor_list.py -v

# 运行演示
bash examples/bash/demo_actor_list.sh
bash examples/bash/demo_actor_list_remote.sh
```

## 相关文件

- `python/pulsing/cli/actor_list.py` - 核心实现
- `python/pulsing/cli/__main__.py` - CLI 集成
- `python/pulsing/actor/remote.py` - 元信息注册
- `crates/pulsing-py/src/actor.rs` - Rust 元信息提取
- `tests/python/test_actor_list.py` - 测试用例
- `examples/bash/demo_actor_list.sh` - 本地演示
- `examples/bash/demo_actor_list_remote.sh` - 远程演示
- `docs/actor-list-guide.md` - 用户文档
