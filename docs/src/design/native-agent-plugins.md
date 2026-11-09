# Pi 与 DeepSeek Harness 的 Pulsing 插件

三个 harness 共用同一套内存邮箱、Pulsing 发现和 MCP 服务：
Codex 通过原生 MCP 配置接入，Pi 通过 extension 注册工具，
DeepSeek Harness（DSH）通过 Cordis 插件注册工具。插件名均为 `pulsing`。
不包含模型客户端，不改变 harness 的模型配置、权限或审批策略。

这里的 Pi 是 Pi coding agent，不是 Pi Durable；
DSH 是 DeepSeek Harness，不是将整个 dsh-superagent 研究原型移植到 Pulsing。

## 推荐：启动时自动注入

```bash
pip install 'pulsing[mailbox]'

pulsing -- pi
pulsing -- dsh web
pulsing -- dsh headless "检查当前仓库的测试说明"
pulsing -- codex
```

Pi 和 DSH 必须已安装。验证版本：

- Pi：`@earendil-works/pi-coding-agent` 1.5.0，Node.js 22.19+。
- DSH：`@deepseek-ai/dsh` 0.2.2-rc.1，该版本要求 Node.js 24.9+。
- Python：安装了 `pulsing[mailbox]` 的 Python 3.10+。

包装器为本次调用启动一个伴生 Pulsing 节点；Pi 用 `--extension` 加载插件，
DSH 用临时 `--patch` overlay 加载插件，不安装全局 npm 包、不修改用户 profile。
原有 profile、prompt 和应用参数保留。DSH 的简写 `dsh web` 会规范为
`dsh --profile web`，避免插入 launcher 参数后破坏简写解析。

如果可执行文件改了名称，可显式指定适配器：

```bash
pulsing --adapter pi -- /path/to/custom-pi
pulsing --adapter dsh -- /path/to/custom-dsh --profile web
```

Pi 的 install/auth/config 等管理子命令、DSH 的 plugin 管理子命令不注入会话插件。
包装器不是替代它们的包管理器。

## 组网

机器 A 拥有 seed IP，机器 B 使用相同的 SEED；无需分别指定 PULSING_ADDR：

```bash
# 机器 A
SEED=10.0.0.10:8100 pulsing -- pi

# 机器 B
SEED=10.0.0.10:8100 pulsing -- dsh web
```

Codex 也可以加入同一网络。seed IP 所在机器自动绑定 seed，其他机器选择路由接口
和可用端口。应使用互通的私网地址，确保防火墙允许节点之间的 actor 通信。
当前插件互通集成测试使用同机独立进程，不能替代物理多机网络验证。

## 独立安装 Pi 插件

生成包时不会自动安装依赖，也不会修改 Pi 配置：

```bash
python -m pulsing.mailbox pi-plugin --output ./pulsing-pi
pi install /absolute/path/to/pulsing-pi

SEED=10.0.0.10:8100 pi
```

本地开发也可直接加载生成的 extension：

```bash
pi --extension /absolute/path/to/pulsing-pi/pi.mjs
```

这是本地生成的包，尚未发布 npm；不要用 `npm:pulsing` 猜测注册表安装地址。
`pi-plugin --python /absolute/path/to/python` 可以指定已安装 Pulsing 的解释器。
还可显式设置 `--seed`、`--addr`、`--runtime-dir` 和 `--instance`。
默认网络设置在运行时读取，不捕获生成插件时的 SEED 或凭据。
已有输出目录会被拒绝，升级时生成到新目录。

## 独立安装 DSH 插件

```bash
python -m pulsing.mailbox dsh-plugin --output ./pulsing-dsh
dsh plugin --profile web add "link:/absolute/path/to/pulsing-dsh"

SEED=10.0.0.10:8100 dsh web
```

生成的是 Cordis bundle，包含 `package.json` 和 `cordis.patch.yml`，
不要求编译 TypeScript 或安装第三方运行时依赖。
安装命令由操作者执行，会改变指定 DSH profile；Python 生成器自身不做该操作。
插件使用 DSH 的公开 agents/tools/systemPrompt 服务，预览版本接口可能继续变化。

dsh-superagent 等上层应用可能通过工具 allowlist 隐藏 `mail_*`，
或以完整 system prompt 替换通用协议段。此时需在上层应用的授权范围内配置；
插件不绕过这些限制，不修改其工作票据、GitHub/飞书接入或权限。

## 伴生进程与身份

两种加载方式行为相同：

1. 包装器已提供连接时，插件复用它，不再启动第二个节点。
2. 独立安装模式首次启动会话时，插件使用配置的 Python 自动启动伴生进程。
   端点和 bearer token 只通过匿名管道返回，不写入包、profile 或 rendezvous 文件。
3. 工具 schema 从现有 MCP 服务发现；插件不维护另一份 Python 邮箱协议。
4. Pi 每次工具调用读取 `ctx.sessionManager.getSessionId()`；
   DSH 每次读取工具执行上下文的 `execution.agent.id`。
   分别以 `pi:`、`dsh:` 命名空间发送 `_meta.threadId`。
5. 缺少原生身份时拒绝执行，不能退回父 agent 的邮箱；
   DSH 的继承工具也不捕获父身份。模型参数不能选择发送者。
6. Pi 新建/恢复/分叉会话随原生 ID 切换邮箱；reload 不重置伴生节点。
   SDK 自行创建的会话也必须加载该 extension，不能假设第三方 subagent 插件会自动继承。
7. 退出时释放插件自建的伴生进程；包装器创建的进程仍由包装器监督。
   移除 standalone Pi extension 后若当前进程仍运行，重启 Pi 以释放进程级伴生资源。

注册的工具为 `mail_identity`、`mail_roles`、`mail_send`、`mail_list`、
`mail_read`、`mail_claim`、`mail_finish`、`mail_reply`、`mail_status`。
消息正文不是执行授权，收到邮件不自动触发模型运行。
Pi 可以显式启用[有预算的消息驱动调度](message-dispatch.md)，DSH/Codex 暂不自动唤醒。

## 日志、限制与验证

邮箱状态、角色、领取和去重键仅存在内存。重启后丢失，日志不恢复运行邮箱。
Actor JSONL 记录处理过程，`--audit-log` 可额外记录 MCP 边界审计。
回放方式与 Codex 相同，见[邮箱文档](agent-mailboxes.md)。
这不是 hostile-node 认证、沙箱、持久任务或自动唤醒系统。

确定性回归不调用模型：

```bash
pytest tests/python/mailbox -q
node --test tests/python/mailbox/native_bridge.test.mjs
```

可选真实 SDK 兼容检查需要先安装指定版本，再指向包含 node_modules 的目录：

```bash
npm install --prefix .pulsing/harness-dev --ignore-scripts --no-audit --no-fund \
  @earendil-works/pi-coding-agent@1.5.0 @deepseek-ai/dsh@0.2.2-rc.1

PULSING_NATIVE_SDK_ROOT="$PWD/.pulsing/harness-dev" \
  pytest tests/python/mailbox/test_native_plugins.py -q
```

检查包含真实 Pi SDK 的工具注册、reload 和新会话切换，真实 DSH Cordis 注册器的
工具注册/卸载，以及通过不同伴生节点进行 Pi/DSH 邮件交接。
DSH 注册器检查不模拟完整模型推理；没有执行付费模型或长期自主任务。

接口参考：[Pi extensions](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md)、
[DSH](https://www.deepseek.com/en/harness/)、
[dsh-superagent 的公开集成示例](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/main/src/app/sessions.ts)。
