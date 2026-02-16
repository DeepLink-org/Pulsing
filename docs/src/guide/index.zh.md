# 用户指南

本指南介绍**如何使用** Pulsing 构建应用。设计原理请参阅 [架构与设计](../design/architecture.zh.md)。

**建议阅读顺序：** 先看 [Actor 基础](actors.zh.md)（什么是 Actor、`spawn`、方法），再看 [通信范式](communication_patterns.zh.md)（ask / tell / streaming），最后看 [远程 Actor](remote_actors.zh.md)（集群与 resolve）。

---

## 主题

<div class="grid cards" markdown>

-   :material-account-group:{ .lg .middle } **Actor 基础**

    ---

    什么是 Actor、`@remote`、`spawn` 以及构建健壮分布式应用的核心模式

    [:octicons-arrow-right-24: Actor 指南](actors.zh.md)

-   :material-message-text:{ .lg .middle } **通信范式**

    ---

    何时使用同步、异步、流式和发送即忘（ask / tell）

    [:octicons-arrow-right-24: 通信范式](communication_patterns.zh.md)

-   :material-cloud-sync:{ .lg .middle } **远程 Actor**

    ---

    集群搭建与 resolve：命名 Actor、多节点

    [:octicons-arrow-right-24: 远程 Actor](remote_actors.zh.md)

-   :material-tools:{ .lg .middle } **CLI 运维**

    ---

    运行、检查和基准测试集群

    [:octicons-arrow-right-24: 运维操作](operations.zh.md)

</div>

---

## 快速链接

| 目标 | 链接 |
|------|------|
| 刚接触 Pulsing？ | [快速开始](../quickstart/index.zh.md) |
| 先学 Actor 基础 | [Actor 指南](actors.zh.md) |
| 选择通信范式 | [通信范式](communication_patterns.zh.md) |
| 可靠性模式 | [可靠性](reliability.zh.md) |
| 保护集群安全 | [安全](security.zh.md) |
| 运行 LLM 推理 | [LLM 推理](../quickstart/llm_inference.zh.md) |
| API 详情 | [API 概述](../api/overview.zh.md) |
| 完整 API 契约 | [完整参考](../api_reference.zh.md) |
