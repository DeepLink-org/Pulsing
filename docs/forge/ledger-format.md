# 账本格式规范 v0.1 — Agent 团队的飞行记录仪

> 谁定义格式，谁有生态。这是 Forge 最核心的标准化资产。

## 文件布局

```
runs/<scenario>-<unix_ts>/
├── events-<shop>.jsonl    # 每店一个文件（并行追加无锁）
├── report.txt             # 人读终态报告
└── sequence.html          # 时序图（可选）
```

## 事件格式（JSONL, 每行一条）

```json
{"ts_ms": 1793343015000, "from": "writer", "to": "judge",
 "body": "DRAFT=```python\n...\n```", "ev": "reply_out"}
```

### 事件类型

| ev | 含义 | 写入时机 |
|---|---|---|
| `reply_out` | 店发出一封信 | send 成功后（先记账后投递） |
| `silent_inbound` | 静默信入箱 | silent=true 到达 |
| `inbound` | 非静默信入箱（唤醒） | silent=false 到达 |
| `drain` | 店读走全部信 | drain 执行后 |
| `turn_end` | 回合结束 | 店完成一次店员调用 |
| `codex_err` | 潮汐层店员 stderr | 非空时 |

### 不变量（与实现绑定，规范级）

1. **Exactly-once**: 同一 (from, seq) 序列号至多出现一次
2. **先记账后投递**: 信件送达（触发回合）前事件必须已落盘
3. **账本即登记处**: 信箱 drain 后信件消失，但 events 中的记录永存——一切回看走账本
4. **顺序**: 同一店的事件按 ts_ms 严格递增

## 账本的消费方式

| 消费者 | 用途 | 实现参考 |
|---|---|---|
| `replay()` | 重建任意时刻的状态 | forge-runtime Journal |
| `sequence.html` | 泳道时序图 | 8765 大屏 |
| `check-seq` | 通信顺序合法性 | 压测巡检 |
| `audit` | 通信审计 | 压测 summary |

这些消费者在 Forge 中都以独立工具存在——账本是唯一共享的真相。
