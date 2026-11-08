# Codex 多机邮箱插件

推荐使用 [`pulsing -- codex` 伴生启动器](agent-launcher.md)，实现一个 Codex
对应一个 Pulsing 进程，并为其各 agent/subagent 绑定独立邮箱。下文保留独立插件
模式：每个 MCP 进程固定一个身份，不等同于伴生模式的多 thread 身份绑定。

插件将现有 Pulsing MCP server 和邮箱协作 skill 打包为 Codex 可安装的
本地插件。每个 Codex 进程绑定一个身份，不需要额外的 LLM SDK，也不修改邮箱协议。
使用 Codex 插件机制的兼容格式 `.codex-plugin/plugin.json`，附带 `.mcp.json`、
`skills/` 及本地 marketplace。插件生成过程不修改 `~/.codex`，不启动模型。

**邮箱基础设施是隐式的**：Codex 加载插件的 MCP 时自动启动或复用本机节点，
或按配置加入已有网络，同时自动注册自身身份。不需要单独运行邮箱服务，
也不需要先让 Router 创建 Worker 的邮箱。

## 准备环境

每台参与机器安装当前仓库版本的 Pulsing，以及 `mailbox` extra。
从源码构建（Linux/macOS）：

```bash
uv sync --extra dev --extra mailbox
uv run maturin develop --locked
```

下文 `python` 指安装了此版本 Pulsing 的 Python；使用上述环境时，可替换为
`uv run python`。Codex 自身的安装、登录和模型权限由操作者管理。
CLI 安装步骤已在 `codex-cli 0.164.0` 验证；旧版本若没有 `codex plugin`
子命令，需要先升级。

## 1. 本地开发：安装后直接启动 Codex

```bash
uv run python -m pulsing.mailbox codex-plugin \
  --output .pulsing/codex-router \
  --identity router \
  --kind router \
  --seed 1 \
  --addr 127.0.0.1:8100

codex plugin marketplace add "$PWD/.pulsing/codex-router"
codex plugin add pulsing@pulsing-local
codex -C "$PWD"
```

`--seed 1` 等价于 MCP 环境中的 `SEED=1`：Codex 加载插件时，**当前 MCP
进程本身就是 seed**，直接监听 `127.0.0.1:8100` 并创建空的内存邮箱目录，
不会再拉起另一个 host 进程。其他本机 agent 使用 `--seed 127.0.0.1:8100` 加入。

默认运行目录为 `$XDG_CACHE_HOME/pulsing/mail-runtime`，未设置时为
`~/.cache/pulsing/mail-runtime`。生成器的 `--runtime-dir /absolute/path`
隔离进程元数据和 actor 日志；这不是邮箱数据库或恢复目录。

起始提示词只需要：

```text
使用 pulsing 插件，确认身份并列出可以协作的 agent。
```

## 2. 多机：首个插件引导网络，其他插件自动加入

假设 A、B、C 的私网 IP 分别为 `10.0.0.10`、`10.0.0.11`、`10.0.0.12`。
请替换为实际地址，不要直接照用示例 IP。
在 A 上生成 seed 插件，指定 `--seed 1`：

```bash
python -m pulsing.mailbox codex-plugin \
  --output .pulsing/codex-router \
  --identity router \
  --kind router \
  --seed 1 \
  --runtime-dir "$PWD/.pulsing/codex-network" \
  --addr 10.0.0.10:8100

codex plugin marketplace add "$PWD/.pulsing/codex-router"
codex plugin add pulsing@pulsing-local
codex plugin list --marketplace pulsing-local --json
codex -C "$PWD"
```

Codex 加载插件时，当前 MCP 进程自动成为 seed，进程的 actor 监听地址
`10.0.0.10:8100` 就是其他进程使用的 seed 地址，无需单独启动服务。
同一机器的其他 agent 也按普通加入节点连接这个地址。
各机器仍须安装并启动自己的 Codex 会话；插件不远程启动模型。

生成器默认把当前 Python 的绝对路径写入 MCP 配置。若 Codex 使用另一环境，
提供 `--python /absolute/path/to/venv/bin/python`，该环境也必须安装此版本
`pulsing[mailbox]`。生成结果不依赖 Codex 的工作目录，但不能将 A 的 Python 路径
原样搬到不同路径的 B；应在每台机器上单独生成。

输出目录必须不存在，生成器不会覆盖已有目录或更改已有 Codex 设置。
`--seed` 可以重复，但 `1` 不能与其他地址或另一个 `1` 混用。
生成配置固定 identity，不依赖 shell 的变量替换。
`--kind` 默认为 worker。身份不存在时自动注册；已经由父角色创建过的身份保留
原有 kind、parent，不会被插件启动配置覆盖。

如果 Codex 沙箱/网络策略阻止 MCP 入网，请按本机策略批准所需连接，不要通过
关闭全部审批或沙箱绕过限制。

## 3. 在 B、C 安装独立身份

在 B 上：

```bash
python -m pulsing.mailbox codex-plugin \
  --output .pulsing/codex-worker-b \
  --identity worker-b \
  --kind worker \
  --seed 10.0.0.10:8100 \
  --addr 10.0.0.11:8201

codex plugin marketplace add "$PWD/.pulsing/codex-worker-b"
codex plugin add pulsing@pulsing-local
codex -C /path/to/repository
```

在 C 上重复以上步骤，identity 改成 `reviewer-c`，本机地址改为
`10.0.0.12:8201`，kind 改为 reviewer，output 使用新的目录。
插件自动加入 A 的网络并注册身份，无需预先创建邮箱。
指定 seed 但连接或目录发现失败时，初始化会报错，不会退回创建独立网络。

组网仍需要一个可达的入口地址；这里的“自动”是启动、连接、注册自动完成，
不是在任意网络中无配置发现其他机器。网络必须允许参与节点双向通信；
使用本机可绑定、其他节点可达的私网 IP，多机部署不能使用 `127.0.0.1`。
首个节点需要稳定端口；加入节点可以使用 `:0`，严格防火墙下则需预留端口。

此原型没有对抗恶意参与者的身份认证。仅使用可信私网或 VPN，并以防火墙限制
集群访问，**不要将 actor 端口暴露到公网**。

每个生成的 marketplace 都命名为 `pulsing-local`。一个 Codex 配置目录只安装
一个身份；若同一机器需要多个角色，应为各进程分别设置独立的 `CODEX_HOME`
（并分别完成登录/模型配置及插件安装）。不要将多个角色的插件装进同一个
Codex 配置目录，也不要通过修改身份读取别人的邮箱。

## 节点生命周期与本地状态

`SEED=1` 的节点与其 MCP 进程同生命周期：关闭或重启该 MCP 会暂时中断邮箱服务；
再次加载同一配置会得到空邮箱，不恢复消息或租约。应保持 seed 会话运行，当前没有跨机器 HA
或存活会话的无缝重连。其他节点不会因连接失败就私自创建另一个网络。

actor 日志、`endpoint.json` 和锁文件保存在 runtime-dir，没有 SQLite。
只有一个进程可以拥有 runtime-dir。`mailbox://local/inbox` 返回当前进程的
`pid`、`addr` 和 `seed`，其中 seed 的 `pid`、`addr` 与 endpoint 记录一致。
端口设为 `:0` 时，可从该资源读取实际分配的端口；多机稳定部署建议指定固定端口。

兼容的本机便利模式仍可用：既没有 seed 参数也没有 SEED 环境变量时，插件自动
启动或复用一个独立的运行中 sidecar，其他无 seed 的本机插件共享它。该节点内存
独立于 MCP 客户端会话保留，但 sidecar 重启即清空；进程日志在 runtime-dir 的
`host.log`。需要“当前进程就是 seed”时应显式指定 `SEED=1`，不要依赖此便利模式。
不要在节点仍持有锁时删除运行目录。`mail-events.jsonl` 可用
`python -m pulsing.mailbox replay` 离线查看，永远不会被加载成运行状态。

## 使用 SEED 环境变量

除了生成器的 `--seed` 参数，MCP 进程也支持以下环境配置：

| 配置 | 行为 |
|------|------|
| `SEED=1` | 当前 MCP 进程作为 seed，自动创建内存目录 |
| `SEED=10.0.0.10:8100` | 加入这个 seed |
| `SEED=10.0.0.10:8100,10.0.0.20:8100` | 使用多个入口地址加入网络 |
| `PULSING_ADDR=10.0.0.10:8100` | 当前进程的 actor 监听地址 |

显式 `--seed` 参数优先于 SEED，显式 `--addr` 优先于 PULSING_ADDR。
生成器未指定网络参数时，不会把生成时 shell 的网络变量冻结到插件中，而是在
MCP 启动时读取环境。未配置监听地址时默认 `127.0.0.1:0`。

在插件 `.mcp.json` 的 `mcpServers.pulsing-mailbox` 中，可配置：

```json
{
  "env": {
    "SEED": "1",
    "PULSING_ADDR": "10.0.0.10:8100"
  }
}
```

上面是该 server 对象的字段片段；保留原有 command、args，并移除需要由环境决定的
`--seed`、`--addr` 参数。务必让 Codex 将这些环境变量传给 MCP，不能假设宿主
shell 中的任意变量都会被客户端转发。使用生成器的 `--seed 1 --addr ...`
可以直接固化等价配置，不依赖环境转发。

## 4. 分派、执行与验收

Router 提示词示例：

```text
通过邮箱把当前任务分派给 worker-b。写清仓库地址、基线 revision、
允许修改的路径、任务边界和验收命令，使用稳定的幂等键。
让 worker-b 完成后通过邮箱向 reviewer-c 请求审查，并把结果回复给 router。
不要因为对方邮箱存在就假定它在线；没有结果时报告等待状态。
未经我批准不要 push 或合并。
```

Worker 提示词：

```text
使用 pulsing 插件，确认身份 worker-b，检查来信。
只领取当前授权范围内的任务；领取成功后按租约执行。
回复修改摘要、可获取代码的位置和实际验证结果，再完成原任务。
没有任务就结束本轮，不要无限轮询。
```

Reviewer 提示词：

```text
使用 pulsing 插件，确认身份 reviewer-c，检查审查任务。
成功领取后核对指定代码和验收标准，回复问题与验证结果，再完成任务。
没有任务就结束本轮。
```

操作者在投递任务后让对应会话处理邮箱，在回复后让 Router 检查收据。
本插件没有后台唤醒或常驻 worker loop；挂起的 Codex 不会因为来信自动执行。
需要无人值守调度时，应由另行授权的外部 harness 驱动 Codex 轮次。

邮箱只传任务与结果，不同步代码、Git objects、工作目录或凭证。
各机器需要相同基线以及操作者批准的代码传递渠道，例如共享仓库、获准发布的
分支或 artifact。领取租约不提供业务副作用 exactly-once，重试前必须核对既有修改。
审查成功和 `completed` 收据也不代表代码已合并。

## 验证与排查

```bash
pytest tests/python/mailbox -q
```

测试通过真实 MCP SDK 启动生成配置中的命令，覆盖独立 Router/Worker/Reviewer
进程、自动启动、并发复用、自动身份注册、seed 加入、退出后的节点保留、
重启清空、日志回放、投递、幂等、领取、回复和完成；不调用付费模型。
它们在单机通过 TCP 连接不同进程，不替代真实私网、防火墙和 Codex 模型端到端验收。

- 无插件命令：检查 Codex 版本。
- 找不到 Python/Pulsing：检查生成的 `.mcp.json` 中绝对路径及该环境安装状态。
- 启动失败：检查 MCP 错误日志、目录权限及监听端口占用；
  本机 sidecar 模式还可检查 runtime-dir 的 `host.log`。
- 无法发现 actor：检查 seed、双向路由、本机绑定 IP 和端口策略；
  首个机器需要先启动 Codex，让插件引导网络。
- 同身份任务已领取：另一个消费者可能还持有租约，不要绕过 token。
- 改了连接信息：生成新的输出目录，更新本地 marketplace，再按 Codex 版本支持的
  refresh/reinstall 流程刷新插件并重启会话；不要假设缓存会实时读取源文件。

插件打包格式参考：
[OpenAI plugin packaging](https://developers.openai.com/plugins/build/plugins)。
