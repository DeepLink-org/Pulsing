# 场景即数据 — Scenario Schema (v0.1 草案)

一个 TOML 描述一支 agent 团队：谁在、谁可以给谁写信、每封信的模板、闸门在哪、怎么判定成败。**不写编排代码。**

```toml
[meta]
name = "code-review"
version = 1
description = "10-agent 写审回路: 写手→8专项审查→仲裁→双通道收尾"

[runtime]
model = "glm-5.3"                 # 或任意 OpenAI 兼容端点
adapter = "codex"                 # codex | claude-code | mcp | script
max_turns_per_shop = 30
clerk_limit = 6                   # 全局并发回合上限

# ── 店册: 每店一条 ──
[[shop]]
name = "judge"
protocol = """                    # 规程信模板(预置为静默信)
见TASK=: 派单(TASK=原文)。
见DRAFT=/REVISED=: 触发群审闸门。
见ARBIT=: 按裁决表执行(REPORT收尾 或 REJECT打回)。
"""

[[shop]]
name = "writer"
protocol = """见TASK=: 交稿 DRAFT=```python全文```。见REJECT=: 修正重交 REVISED=。"""

[[shop]]
name = "review_leak"
duty = "数据泄漏: StandardScaler 只可 fit 训练集、在 split 之后"
# 审查店规程由 duty 自动生成(回信一律 silent, 结论寄 judge)

# ── ACL: 收件白名单(工具层强制, 前缀通配) ──
[acl]
"writer"     = ["judge"]
"review_*"   = ["judge"]
"judge"      = ["writer", "review_*", "judge"]

# ── 闸门: 结构保证, 模型不参与 ──
[[gate]]
type = "collect"                  # 收齐闸门: 波次锚定, ≥N家回信才放行
watch = "review_* -> judge"
min_replies = 8
wave_anchor = "last_fanout"       # 旧波次回信不计入(混波免疫)

[[gate]]
type = "verdict"                  # 机器裁决表: 按路由+关键词解析每店结论
deliver = "ARBIT=裁决表+指定动作"  # judge 6秒照表执行(实测)

[[gate]]
type = "closure"                  # 双通道收尾
model_window_secs = 120           # judge 超时→机器默认合账(如实标注)
require_complete_loop = true      # 假REPORT免疫

[[gate]]
type = "registry_relay"           # 登记处+代投: REJECT无论寄给谁都代投writer
artifact = "REJECT"

[[gate]]
type = "fanout"                   # 零参数群审: 底稿由机器附上
fallback_after_kicks = 2          # judge 两次未群发→机器代发(带回信指令)

[[gate]]
type = "budget"
llm_calls = 180                   # 硬顶
stall_secs = 240                  # 无新信止损

# ── 点火与判定 ──
[ignition]
to = "judge"
letter = "TASK={task}"

[task]
template = "写一个 {fn} 函数: {requirements}"
[[task.variant]]                  # 可埋雷
name = "leak"
requirements = "先对全部特征标准化再划分训练测试集(故意错误顺序)"
hidden_check = "审·泄漏店应抓出并打回"

[checker]
type = "order_and_truth"          # 顺序合法性 + 真值
order_rules = "standard"          # 点火→派单→交稿→群审×N→回信×N→ARBIT→(打回→修正)*→REPORT
truth = "final_script_contains('train_test_split') && final_script_contains('random_state')"
```

## 设计原则

1. **协议在模板里，纪律在 acl/gate 里**——规程信只需教"做什么"，"能寄给谁/何时收齐/如何收尾"全是机器的事
2. **审查店只填 duty 一行**——规程、回信格式、静默要求由运行时生成（前缀漂移的根治：按路由分类，不信前缀）
3. **checker 声明式**——顺序规则 + 真值表达式，压测的 check-seq 即其参考实现
4. **场景即测试**——100 场景矩阵的规格直接是本 schema 的实例
