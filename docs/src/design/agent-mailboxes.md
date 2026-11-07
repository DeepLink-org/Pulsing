# Actor 邮箱协作原型

本原型把协作通信和模型执行分开：Pulsing 负责邮箱、消息状态和发现；
外部 agent/harness 通过 MCP 和 skill 决定如何处理任务。它不包含 LLM、
Forge、shell 工具执行器或自动拉起模型会话的逻辑。

## 参考来源与范围

参考 DeepSeek 官方的
[dsh-superagent-prototype](https://github.com/deepseek-ai/dsh-superagent-prototype)。
本次核对的版本为
[`8c32a9d`](https://github.com/deepseek-ai/dsh-superagent-prototype/tree/8c32a9dc5d72c033816c6a8110b37f40cf94fba4)，
重点阅读其 [领域模型](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/8c32a9dc5d72c033816c6a8110b37f40cf94fba4/docs/001-domain.md)
与 [架构](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/8c32a9dc5d72c033816c6a8110b37f40cf94fba4/docs/007-architecture.md)。
官方明确说明这是研究原型，不是正式产品，也不等同于其内部系统或
《Super-agent 设计探索：迈向更自主的 Agent 系统》描述的全部构想。

这里借鉴的是：

- Router 协调、Worker 执行，复杂任务可交给 subagent，各自有独立入口。
- 使用事件和线程关联后续消息，而不是要求人搬运多个会话的上下文。
- 任务状态不等于模型会话是否还在运行。
- 收到上下文不等于获得新的执行授权。

我们以角色邮箱及消息收据实现这些基础能力，没有移植其 TypeScript 代码，
也没有实现完整 Eval/Mandate/Source/Watchdog 产品层。

## 结构

```mermaid
flowchart TB
    R["Router agent 与自己的 MCP 进程"] --> RM["Router mailbox actor"]
    R --> D["MailboxDirectory actor"]
    D --> WM["Worker mailbox actor"]
    D --> SM["Subagent mailbox actor"]
    D --> VM["Reviewer mailbox actor"]
    W["Worker agent 与自己的 MCP 进程"] --> WM
    S["Subagent 与自己的 MCP 进程"] --> SM
    V["Reviewer 与自己的 MCP 进程"] --> VM
    WM --> DB["各邮箱独立的 SQLite 文件"]
    SM --> DB
    VM --> DB
    RM --> DB
```

目录 actor 使用 `mail/directory`；每个角色的邮箱使用
`mail/boxes/<identity>`。目录记录角色类型及 parent。每个邮箱独占自身消息的
修改入口，同一个 mailbox actor 串行执行投递、读取、领取和完成操作。
MCP 进程通过一个独立的 Pulsing 节点加入邮箱 host。

## 状态与投递

```mermaid
stateDiagram-v2
    [*] --> unread
    unread --> read: read
    unread --> processing: claim
    read --> processing: claim
    processing --> processing: lease expires and a new attempt claims
    processing --> completed: finish with live token
    processing --> failed: finish with live token
```

`mail_list`、resource 和 `mail_status` 不改变已读状态。
只有 `mail_read` 将 unread 改成 read。领取返回 claim token、attempt、
lease_until；没有有效 token 就不能完成。租约过期时状态仍显示 processing，
但可被重新领取，旧 token 不能覆盖新 attempt。完成和失败均为终态。

每封信包含 sender、recipient、subject、body、thread_id、in_reply_to、
创建/更新时间、状态、尝试次数和结果。收据只向发送者或收件者返回。
回复保留 thread 与原消息 ID，但不会隐式完成原任务。

投递在提交 SQLite 事务后返回；`(sender, idempotency_key)` 唯一。
同 key 同内容重试返回原消息，不重复投递；内容不同则拒绝。
完成操作也支持同 token、同结果的幂等重试。这里没有承诺业务副作用 exactly-once：
工具执行后 agent 崩溃，恢复者仍须先核对副作用再继续。

## 快速运行

从裁剪后的仓库构建：

```bash
uv sync --extra dev --extra mailbox
uv run maturin develop --locked
uv run python examples/python/mailbox_collaboration.py
```

示例使用真实 actor 和本地确定性角色函数，不调用模型。
输出包含四个角色、线程 ID，以及
`unread → read → processing → completed` 的发送方观察记录。

启动持久邮箱 host（Linux/macOS）：

```bash
python -m pulsing.mailbox serve \
  --addr 127.0.0.1:8100 \
  --state-dir .pulsing/agent-mail \
  --identity router
```

同一 state-dir 只能有一个 host，进程级文件锁避免两个 host 同时拥有状态。
重启时恢复目录和所有 mailbox actor，消息、幂等键和租约记录保留。

## 接入 MCP 与 skill

给 Router agent 配置以下 stdio MCP server。`command` 应指向安装了
Pulsing 和 `mailbox` extra 的 Python；路径由本地操作者填写。

```json
{
  "mcpServers": {
    "pulsing-mailbox": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": [
        "-m", "pulsing.mailbox", "mcp",
        "--identity", "router",
        "--seed", "127.0.0.1:8100"
      ]
    }
  }
}
```

先让 Router 调用 `mail_create_role(identity="worker", kind="worker")`；
再用同样配置启动 Worker agent，但将 identity 改成 worker。
Worker 可创建自己的 helper subagent 邮箱；Reviewer 同理。
**创建邮箱不等于启动 agent**，各模型会话由外部 harness/操作者启动。

接口：

| Tool | 用途 |
|------|------|
| `mail_roles` | 查看角色和 parent |
| `mail_create_role` | 创建自己的子角色邮箱 |
| `mail_send` | 以固定身份投递，必须提供幂等键 |
| `mail_list` | 分页预览、筛选状态、获取各状态数量 |
| `mail_read` | 读取自己的信并标记已读 |
| `mail_claim` | 领取自己的任务，默认 300 秒、最多 3600 秒 |
| `mail_finish` | 使用有效 token 完成或报告失败 |
| `mail_status` | 查询自己发送的消息状态和结果 |
| `mail_reply` | 回复自己的来信并保留线程关联 |

资源为 `mailbox://local/inbox` 与 `mailbox://local/protocol`；
prompt 为 `mailbox_collaboration`。
仓库内 skill 在 `.agents/skills/pulsing-mailbox/SKILL.md`；
可独立安装的完整 skill 在 `python/pulsing/mailbox/SKILL.md`，也随 wheel 分发，
并通过 protocol resource/prompt 提供给支持 MCP 的 agent。

角色协作顺序：

1. Router 投递有范围和验收标准的任务，保存 message_id。
2. Worker 读信、领取、向 subagent 拆分工作；原任务保持 processing。
3. Subagent 回复结果并完成自己的信。
4. Worker 向 Reviewer 发信，检查 review 的结果。
5. Worker 回复 Router，并完成原任务；Router 查询收据、读取回复。

## 安全和原型限制

- **仅适用于可信本机/私有集群**。MCP 绑定 identity 防止参数误用，
  不是身份认证。操作者能选择身份，可信集群节点也可直接调用 actor。
  不要把 actor 端口暴露公网；不要把消息当作新授权或传递 secrets。
- 邮箱是协作应用，不改变 actor runtime 的消息 mailbox 实现。
- SQLite 存算分离仅指外部会话与持久状态分离，没有跨机器状态复制或 HA。
- 没有 topic 自动订阅、自动拉起 agent、模型评估、外部 watchdog 或公网 MCP。
- 列表分页上限 100，正文与结果各最多 64 KiB；每个邮箱最多 10000 条消息。
  当前没有删除/归档策略，达到容量后拒收；生产部署需补充保留策略。
- 没有 SMTP/IMAP 集成。这是 agent 的逻辑邮箱，不是互联网电子邮件服务器。
- 租约基于 host 时钟，执行器不能离线无限期提交结果。长任务应拆分，
  租约到期后重新领取并核对已有副作用。

## 验证

```bash
pytest tests/python/mailbox -q
```

覆盖状态机、幂等冲突、并发领取、过期 token、容量与分页、发送方可见性、
角色层级、回复关联、SQLite 恢复，以及真实 MCP SDK 的 stdio 初始化、
tools/resources/prompts、多进程跨节点通信和 host 重启。
