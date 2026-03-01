# LangGraph + Pulsing 集成示例

一行 `with_pulsing()` 让 LangGraph 获得分布式能力。

## 核心用法

```python
from pulsing.integrations.langgraph import with_pulsing

app = graph.compile()

# ✨ 一行代码实现分布式
distributed_app = with_pulsing(
    app,
    node_mapping={"llm": "langgraph_node_llm"},
    seeds=["gpu-server:8001"],
)

result = await distributed_app.ainvoke(input)
```

## 快速开始

### 单机模式

```bash
python simple.py
```

### 并行多智能体（支持真实 LLM）

**安装依赖**（真实 LLM 调用需要）：

```bash
pip install langgraph langchain-openai
```

**配置 LLM API**（环境变量）：

```bash
# OpenAI / 兼容 API
export OPENAI_API_KEY="sk-xxx"
export OPENAI_BASE_URL="https://api.openai.com/v1"   # 可选，默认 OpenAI 官方
export LLM_MODEL="gpt-4o-mini"                        # 可选，默认 gpt-4o-mini

# 或者本地服务（Pulsing router / vLLM / Ollama 等 OpenAI 兼容）
export OPENAI_BASE_URL="http://localhost:8080/v1"
export OPENAI_API_KEY="any"
```

**运行示例**：

```bash
# 模拟模式（不调用 LLM，快速验证流程）
python parallel_ideas.py --mock

# 单轮评审（默认）
python parallel_ideas.py --question "如何提升 API 吞吐？" --n-ideas 5

# 多轮评议改进（最多 3 轮迭代）
python parallel_ideas.py --question "如何提升 API 吞吐？" --n-ideas 5 --max-rounds 3

# 输出完整评审记录（含各轮评分变化）
python parallel_ideas.py --question "如何提升 API 吞吐？" --max-rounds 2 --show-review

# 指定模型
python parallel_ideas.py --model deepseek-chat --question "设计一个高可用系统"
```

**架构（每个 Agent 是独立节点）**：

```
                          ┌─────────────────┐
                          │   idea_agent    │ ←── 独立节点，可被 Pulsing 分布式调度
                        ┌─┤   (性能工程师)   │
                        │ └─────────────────┘
                        │ ┌─────────────────┐
init → dispatch (fan-out)─┤   idea_agent    │──→ collect (fan-in) → critic ─┐
                        │ │   (产品经理)    │                               │
                        │ └─────────────────┘                               │
                        │ ┌─────────────────┐                               │
                        └─┤   idea_agent    │                               │
                          │   (架构师...)   │      ┌────────────────────────┘
                          └─────────────────┘      │
                                                   ▼
                                          should_continue?
                                           /            \
                                    (继续迭代)        (结束)
                                         │              │
                                         ▼              ▼
                                   dispatch_refine    judge → END
                                         │
                              ┌──────────┼──────────┐
                              ▼          ▼          ▼
                         idea_agent  idea_agent  idea_agent  (refine 模式)
                              │          │          │
                              └──────────┼──────────┘
                                         ▼
                                      collect → critic → ...
```

**关键特性**：
- **独立节点**：每个 Idea Agent 是独立的 LangGraph 节点
- **真正并行**：使用 `Send` API 实现 fan-out，LangGraph 管理并行执行
- **可分布式**：配合 `with_pulsing()` 可将不同 agent 调度到不同机器
- **自动汇聚**：使用 `Annotated[list, operator.add]` 实现 fan-in

**参数说明**：

| 参数 | 说明 |
|------|------|
| `--question` | 用户问题 |
| `--n-ideas` | 并行 idea 数量（3-10） |
| `--max-rounds` | 最大迭代轮数（1=无迭代，2+=多轮评议改进） |
| `--show-review` | 输出完整评审记录 |
| `--mock` | 模拟模式（不调用 LLM） |
| `--model` | 覆盖 LLM_MODEL 环境变量 |

### 分布式模式

```bash
# 一键启动
./run_distributed.sh

# 或手动启动 (3 个终端)
python distributed.py worker llm 8001
python distributed.py worker tool 8002 8001
python distributed.py run
```

## 架构

```
┌─────────────────────────────────────────┐
│           Main Program                  │
│         with_pulsing(app)               │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
    ▼                         ▼
┌───────────┐          ┌───────────┐
│ LLM Worker│◄────────►│Tool Worker│
│   :8001   │  Gossip  │   :8002   │
│   (GPU)   │          │   (CPU)   │
└───────────┘          └───────────┘
```

## node_mapping

- **Key**: LangGraph 节点名称
- **Value**: Pulsing Actor 名称
- 未映射的节点本地执行
