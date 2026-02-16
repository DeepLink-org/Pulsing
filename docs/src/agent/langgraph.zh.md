# LangGraph 集成

Pulsing 提供 `with_pulsing()`，一行代码实现分布式 LangGraph。

## 特性

- **一行集成** - 包装任意编译后的图
- **节点映射** - 将特定节点路由到远程 Worker
- **API 兼容** - 相同的 `ainvoke()`、`invoke()`、`astream()` 方法
- **状态序列化** - 自动在节点间传递状态

## 快速开始

```python
from pulsing.integrations.langgraph import with_pulsing
from langgraph.graph import StateGraph

# 像往常一样构建图
graph = StateGraph(MyState)
graph.add_node("llm", llm_fn)
graph.add_node("tool", tool_fn)
app = graph.compile()

# ✨ 一行代码实现分布式
distributed_app = with_pulsing(
    app,
    node_mapping={
        "llm": "langgraph_node_llm",
        "tool": "langgraph_node_tool",
    },
    seeds=["gpu-server:8001"],
)

# 使用方式完全相同
result = await distributed_app.ainvoke({"messages": [...]})
```

## 节点映射

`node_mapping` 参数定义哪些节点远程执行：

- **Key**: LangGraph 节点名称
- **Value**: 远程 Worker 上的 Pulsing Actor 名称
- **未映射的节点**: 在本地执行

```python
node_mapping={
    "llm": "langgraph_node_llm",    # LLM → GPU 服务器
    "tool": "langgraph_node_tool",  # Tool → CPU 服务器
}
```

## 启动 Worker

```python
from pulsing.integrations.langgraph import start_worker

# GPU 服务器
await start_worker("llm", llm_node, addr="0.0.0.0:8001")

# CPU 服务器 (加入集群)
await start_worker("tool", tool_node, addr="0.0.0.0:8002", seeds=["gpu:8001"])
```

## 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `node_mapping` | 节点到 Actor 的映射 | `{}` (全部本地) |
| `addr` | 本节点地址 | `None` (自动) |
| `seeds` | 种子节点地址 | `[]` |

## 示例

```bash
cd examples/agent/langgraph
./run_distributed.sh
```

## API 参考

### with_pulsing

```python
def with_pulsing(
    compiled_graph,
    *,
    node_mapping: dict[str, str] | None = None,
    addr: str | None = None,
    seeds: list[str] | None = None,
) -> PulsingGraphWrapper
```

### start_worker

```python
async def start_worker(
    node_name: str,
    node_func: Callable,
    *,
    addr: str,
    seeds: list[str] | None = None,
    actor_name: str | None = None,
) -> None
```

### PulsingGraphWrapper

```python
class PulsingGraphWrapper:
    async def ainvoke(input: dict, config: dict | None = None, **kwargs) -> dict
    def invoke(input: dict, config: dict | None = None, **kwargs) -> dict
    async def astream(input: dict, config: dict | None = None, **kwargs) -> AsyncIterator[dict]
    async def close() -> None
```
