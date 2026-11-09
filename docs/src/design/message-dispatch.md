# Pi 消息驱动调度 v1

这一层将邮件交给原生 Pi harness，而不是重新实现模型循环。
参考 Rivet 的提交/排队/取消分离方式，但不提供持久任务、故障接管或副作用重放。
自动模型运行是显式选择，默认关闭。

## 使用

```bash
pip install 'pulsing[mailbox]'
pulsing --dispatch --dispatch-max-turns 8 -- pi
```

独立安装的插件也可选择启用：

```bash
python -m pulsing.mailbox pi-plugin --output ./pulsing-pi \
  --dispatch --dispatch-max-turns 8
pi install /absolute/path/to/pulsing-pi
pi
```

或在运行时设置 `PULSING_DISPATCH=1`、`PULSING_DISPATCH_MAX_TURNS=8`。
`--dispatch` 当前只适用于 Pi；DSH/Codex/通用包装器收到此选项会明确拒绝，
不会悄悄退回脚本模拟。Pi 已验证版本为 1.5.0。

操作员命令：

- `/pulsing status`：查看 waiting/queued/running/paused/stopped、预算及待处理数量。
- `/pulsing pause`：停止新的自动启动，不强行取消已运行的工作。
- `/pulsing resume`：在 harness 空闲后显式恢复，重置自动尝试预算，
  重新扫描未领取的 unread/read 邮件。默认关闭时，这也是本会话的显式启用方式。

用户取消当前 Pi run 后会暂停自动调度；不会因为未读邮件还在而立即重新运行。
失败或预算耗尽也需操作员检查后恢复。关闭会话会停止通知订阅。
reload 保留同一会话的内存游标、预算和暂停状态；新建会话使用新身份。

## 路径

```text
发送者 → 目标 mailbox actor 接受邮件
                    ↓
             本机 delivery edge
                    ↓
        接收者 MCP mail_wait 请求返回
                    ↓
             Pi extension 调度器
                    ↓
      原生 Pi custom message / follow-up
                    ↓
        原生模型、工具和权限/审批流程
                    ↓
          mail_read / claim / finish
```

没有中央 broker，也不监听或解析日志文件。远程发送最终在接收者所属节点投递，
所以接收者本机的通知同样覆盖跨节点邮件。

## 忙碌、合并与取消

- Pi 忙碌、存在待处理原生输入或正在等待用户交互时不主动启动新 turn。
- 一次通知最多包含 20 个 message ID，不把邮件正文提升为系统指令。
- 通过 Pi 的 custom message 和 follow-up 路径启动，不插队 steering，
  不直接调用工具，不改变原生权限或审批。
- 每会话最多预取 128 个 ID；达到上限后停止取更多，剩余邮件保留在 mailbox。
- 每会话默认最多 8 次自动启动尝试，可配置 1–64。尝试是保守预算：
  审计/提交失败或人工运行抢先进入也可能消耗一次，不代表成功完成一次任务。
- 同一 delivery cursor 不会因邮件仍 unread 而重复触发；幂等投递重试不产生新边。
- 用户取消、harness 错误、通知连接故障、邮箱生命周期变化、预算耗尽都会暂停，
  不自动重试模型或外部副作用。
- 新任务和新回复仍可能形成反馈循环；自动尝试预算是兜底，不是 token 或费用保证。

通知只表示“有邮件可检查”，不表示任务被接纳、领取、完成或验收。
用户显式启用也不意味着邮件能扩大操作员原有授权；模型仍须检查范围，
危险操作和人工审批仍由原生 harness 执行。

取消后，处理中邮件不伪造 finish，也不自动抢占租约。
操作员需核对副作用和租约，再决定是否手动重新领取。
`/pulsing resume` 只重新考虑 unread/read，不能恢复任意中断中的工作。

## 为什么不会堵塞 actor

`Mailbox.changes` 只做有界快照，立即返回。长等待发生在 MCP 请求的协程中，
不是串行 mailbox actor 的处理函数。

每个 mailbox incarnation 有独立 epoch，delivery cursor 随首次投递递增。
快照后的通知订阅与 cursor 比较在同一线程锁下完成，避免“快照与订阅之间刚好
收到邮件”导致丢唤醒。跨线程只通过 `call_soon_threadsafe` 解析 future，
同一 waiter 的突发通知合并成一次唤醒。

长轮询最长 20 秒，客户端单请求超时 30 秒。超时只做本地心跳，不触发模型。
这不是完全没有 HTTP 请求，而是避免让模型反复空查。
每邮箱最多 32 个 waiter，本进程最多 1024 个；取消、超时和正常返回均释放名额。

## API 与日志

伴生 MCP 增加两个身份绑定接口：

- `mail_wait(epoch, after, limit, timeout_seconds)`：只观察自己的新投递，
  返回 epoch/cursor/items/more，不修改已读状态，不领取任务。
- `mail_dispatch_event(phase, dispatch_id, message_ids, reason)`：记录本机 harness
  的观察事件，不执行任务，也不授予权限。

Pi 模型仍只注册原有九个邮箱工具，调度器在后台使用上述接口。
Codex 等 MCP 客户端也能显式调用 `mail_wait`，但不因此获得后台模型唤醒。

actor 日志中的 `dispatch.observed` 关联 requested/submitted/started/settled、
paused/resumed/stopped 与 dispatch ID、邮件 ID。这些是适配器观察，
不是事务型运行账本；原生事件与提交确认可能异步交错。
停止时的最后日志也不构成进程崩溃后的持久保证。
离线 replay 的 `dispatches` 字段及 Markdown 中可查看这些事件。

## 验证及范围

```bash
pytest tests/python/mailbox/test_dispatch.py -q
node --test tests/python/mailbox/dispatch.test.mjs
```

可选原生 SDK 检查沿用[Pi/DSH 插件安装文档](native-agent-plugins.md)中的
`PULSING_NATIVE_SDK_ROOT`：

```bash
PULSING_NATIVE_SDK_ROOT="$PWD/.pulsing/harness-dev" \
  pytest tests/python/mailbox/test_native_plugins.py -q
```

真实 Pi harness 使用本地内存模型事件完成读取、领取、回执，并验证取消后暂停、
操作员恢复；没有外部模型请求、真实 API key 或付费推理。
另覆盖突发合并、背压、预算、审计期间取消、重载游标、epoch 变化、
长轮询并发投递、超时和 waiter 清理。

v1 不包含持久化恢复、完整 Task/Run/Acceptance 台账、deadline/token 预算、
公平调度、文件交接、自动租约续期或 hostile-node 认证。
这些不能仅凭“消息驱动”一词推断为已经支持。
