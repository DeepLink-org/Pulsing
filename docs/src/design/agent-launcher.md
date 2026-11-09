# 用 `pulsing --` 启动 coding agent

推荐入口：

```bash
pip install 'pulsing[mailbox]'
pulsing -- codex
# Codex 的参数原样放在后面：
pulsing -- codex exec --json "检查当前仓库"
```

Pi 与 DeepSeek Harness 也支持自动注入：`pulsing -- pi`、
`pulsing -- dsh web`；原生插件的独立安装和身份机制见[接入文档](native-agent-plugins.md)。
Pi 可用 `pulsing --dispatch -- pi` 显式选择[消息驱动调度](message-dispatch.md)。

无需单独启动邮箱服务，也不需要先安装 marketplace 插件。这里的 `pulsing`
进程就是 Codex 的伴生节点，启动并监督一个 Codex 子进程；它同时承载 Pulsing
runtime、本实例邮箱目录及仅监听 loopback 的 MCP HTTP 接口。
不内置模型，不替代 Codex 的登录、执行权限或沙箱。

Codex 适配器使用仅限本次调用的 `-c mcp_servers...` 注入连接配置，
并用 `--no-daemon` 避免把这个受监督进程变成其他会话共享的 daemon 客户端。
协作协议通过 MCP 初始化 instructions、protocol resource 和 prompt 提供，
不会改写 `~/.codex/config.toml`、仓库规则、用户提示词或全局插件安装状态。
独立的 `pulsing` marketplace 插件是另一种接入方式，不是 wrapper 的前置条件，
也不是此处自动安装的组件。

## 每个 Codex 一个伴生进程，每个 thread 一个邮箱

```text
pulsing A
├── 本地 mailbox actors：agent / subagent 各一个
├── 本地 MCP HTTP 接口
└── Codex A
    ├── 主 agent
    └── subagents

pulsing B
├── 本地 mailbox actors：agent / subagent 各一个
├── 本地 MCP HTTP 接口
└── Codex B
```

每个实例的目录 actor 为 `mail/instances/<instance>`。各邮箱 actor 只由所属
实例创建并持有在有界内存中；跨节点投递通过 Pulsing 命名发现直接找到收件箱。
Seed 是组网入口，不是集中持有所有邮箱的服务。

Codex 每次 MCP tool call 提供 `_meta.threadId`。伴生进程使用
`<instance>-<threadId 的稳定哈希>` 作为邮箱地址，不接受模型填写的 sender。
同一 thread 重连仍是同一个邮箱，subagent 与主 agent 共享网络而不共享身份。
缺少 metadata 的工具调用会失败，不能静默退回主 agent 的邮箱。

首次工具调用时创建邮箱。让主 agent 和每个 subagent 开始工作时先调用
`mail_identity`，再用 `mail_roles` 查找对方的地址及原始 thread ID。
不是在 Codex 的 spawn 事件发生时预建邮箱，也没有 background 模型唤醒。
该模式不提供 `mail_create_role`，子 agent 的创建仍使用 Codex 自身的工具。

## 多机组网

机器 A：

```bash
SEED=10.0.0.10:8100 pulsing --instance machine-a -- codex
```

机器 B：

```bash
SEED=10.0.0.10:8100 pulsing --instance machine-b -- codex
```

两台机器使用相同的 SEED 即可，无需指定 PULSING_ADDR：

- A 拥有 `10.0.0.10`，且端口可用：自动监听 `10.0.0.10:8100`，自身成为 seed，
  不等待另一个节点先启动，也不把自身地址当作远程入口。
- B 不拥有这个 IP：向操作系统查询到该 seed 的路由，选择相应的本机接口 IP，
  自动分配空闲端口，然后加入网络。多网卡、VPN 场景使用的是路由接口，
  不是猜测 hostname 对应地址或固定绑定 localhost。
- 如果 seed 地址在本机且已有节点监听，后续本机实例也会选择空闲端口加入。
  并发争用 seed 端口时，未抢到端口的自动实例会重新判断一次，再加入已有节点，
  这个重试发生在启动目标 agent 之前，不会重复执行 agent。

也可用等价的统一参数，在 A、B 上都运行：

```bash
pulsing --seed 10.0.0.10:8100 -- codex
```

`--` 前是 Pulsing 参数，之后是目标程序及其参数，不经过 shell 重新解释。
显式 seed/addr 参数优先于环境变量。`--seed` 可重复，`SEED` 可用逗号分隔多个地址；
seed 要使用实际拥有的 IP 和固定端口（支持 IPv4 和 `[IPv6]:port`），不能是
`0.0.0.0`、`:0`、hostname 或只存在于外部 NAT 映射上的地址。
地址选择只使用本地 bind/路由查询，不发送应用探测包。
无 seed 时形成独立网络，默认监听 `127.0.0.1:0`。
加入远程 seed 时，在节点初始化完成后最多再等待 15 秒发现另一实例的目录，
发现后才启动目标程序；超时就失败，不回退创建隔离网络。建议先启动 seed 所在
机器的实例，机器之间需要双向私网连接。
如果配置地址上的监听者不是 Pulsing companion，也会因为发现失败而报错，
不会将它误认为可用集群。

高级配置仍支持 `--addr` / PULSING_ADDR 指定监听地址，优先于自动选择。
显式指定一个与本机 seed 相同的地址时，该进程作为 seed；
指定其他地址时，按该地址加入网络，不会被自动改写。
`SEED=1` 仍兼容强制 seed 模式，但不再是多机使用的推荐配置，也不能与其他地址混用。

只用于可信机器上的私网或 VPN。MCP 接口由随机 bearer token 保护，仅绑定
loopback；token 只通过子进程环境传递，不写配置文件。但 actor 集群仍不是对抗
恶意参与者的认证边界，不能把 actor 端口公开到互联网。

## 生命周期与内存边界

- 节点、空内存目录和 MCP 就绪后才启动目标程序。
- 保留目标程序的 stdin/stdout/stderr、工作目录和退出码。
- 转发终止信号，等待退出，超时后终止子进程；目标程序结束后关闭 MCP 和 actor runtime。
- 不自动重启 Codex 或重做任务；租约恢复和副作用核对仍按邮箱协议进行。
- 默认实例 ID 随每次启动生成，进程元数据和日志在
  `$XDG_CACHE_HOME/pulsing/mail-runtime/agents/<instance>`（默认 `~/.cache/...`）。
- 使用稳定的 `--instance` / `PULSING_INSTANCE_ID` 可保留地址映射，但不能保留消息。
  实例 ID 必须在集群内唯一，最多 24 个 ASCII 字母、数字、`_`、`-`。
- `--runtime-dir` / `PULSING_RUNTIME_DIR` 指定进程发现和日志目录；
  `--state-dir` / `PULSING_STATE_DIR` 只作兼容别名。锁防止进程元数据被并发持有。
- 每个邮箱默认最多 10000 条消息、8 MiB 逻辑负载；满载拒收，不隐式丢信。

退出一个实例后，它的角色、信件、幂等键和租约消失，其他实例的邮箱不受影响。
重启不会恢复信件；不能把旧的 message ID/token 当作仍有效的任务继续提交。
没有跨机器复制、HA 或存活 MCP 会话的无缝重连。

## Actor 日志与离线回放

运行目录默认包含 `mail-events.jsonl`，由目录及邮箱 actor 自己记录：
生命周期、角色/thread 绑定、完整投递正文、状态变更、读取、领取、完成、
幂等重试、收据查询和拒绝。每个 actor 生命周期有独立 stream_id 和严格序号。
不包含 claim token；新进程从不读取日志作为状态库。

```bash
python -m pulsing.mailbox replay node-a/mail-events.jsonl node-b/mail-events.jsonl \
  --format markdown --output discussion.md
```

回放只重建离线历史，不向集群投递，不恢复 live mailbox，也不调用模型。
缺失序号、损坏行或冲突重复会报错，不能悄悄跳过并声称记录完整。

## 其他 harness

`pulsing -- <任意程序> ...` 支持通用进程监督，并注入：

- `PULSING_INSTANCE_ID`：实例 ID
- `PULSING_ADDR`：本进程的实际 actor 地址
- `PULSING_MCP_URL`、`PULSING_MCP_TOKEN`：本地 MCP 连接信息
- `PULSING_PROTOCOL`：协作协议文件路径
- `PULSING_PYTHON`：包装器当前使用的 Python，供原生插件启动伴生进程时使用

名为 `codex`、`pi`、`dsh` 的程序（以及 `.exe` 名称）会自动使用对应适配器。
Codex 注入 CLI MCP 配置，Pi 注入 extension，DSH 注入本次调用的 Cordis overlay。
重命名的程序可用 `--adapter codex|pi|dsh`；`--adapter generic` 禁用自动配置。
其他 harness 需要自行消费这些环境变量，并为每个 agent 提供可靠的
`_meta.threadId`；不能宣称任意程序都已自动接入 MCP。

## 可选通信审计

```bash
pulsing --audit-log .pulsing/mcp-audit.jsonl -- codex
```

也可设置 PULSING_AUDIT_LOG。每个实例建议使用独立日志文件。
审计记录经过认证的 MCP POST 请求及其响应，包括时间、实例、PID、调用关联 ID、
threadId、工具名、完整业务参数和结果、HTTP 状态、耗时与异常。
可据此追踪 send/read/claim/reply/finish 的请求、收据及处理状态。
凭据和 claim token 以 SHA-256 指纹代替，Authorization 头不写入日志。

这是额外的 MCP 边界审计，不是邮箱状态库，也不包含模型未公开的内部推理。
它不替代 Codex 的可见会话记录。日志写入失败会显露为错误；不要在磁盘不足或
日志不完整时声称已经获得完整审计。

## 验证

```bash
pytest tests/python/mailbox tests/python/cli -q
```

测试使用确定性 harness 和真实 MCP HTTP SDK，不调用模型。覆盖同进程主/子 agent
身份隔离、缺失 metadata 拒绝、跨伴生进程投递、内存所有权、重启清空、日志回放、
退出码及 SIGTERM。相同 SEED、无 PULSING_ADDR 的真实双进程测试覆盖自动引导与加入；
远程接口选择、多入口和 IPv6 通过确定性路由测试验证，不替代真实多机网络验收。
Codex CLI 0.164.0 的 `mcp list --json` 可用于不调用模型的
配置检查。完整的真实 Codex 模型/subagent 执行和多台物理机器网络仍需单独验收。

Codex thread metadata 接入依据：
[upstream MCP thread metadata](https://github.com/openai/codex/pull/18093)。
