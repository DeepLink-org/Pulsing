# Security Policy

## 支持的版本

| 版本 | 支持状态 |
| --- | --- |
| 0.1.x | ✅ |

## 报告漏洞

如果你发现了安全漏洞，请**不要**在公开的 Issue 中报告。

### 报告方式

1. **GitHub Private Vulnerability Reporting**（推荐）
   - 访问 [Security Advisories](https://github.com/DeepLink-org/Pulsing/security/advisories)
   - 点击 "Report a vulnerability"

2. **邮件**
   - 发送邮件至：reiase@gmail.com
   - 邮件标题请包含 `[SECURITY]`

### 报告内容

请在报告中包含以下信息：

- 漏洞类型（如：远程代码执行、拒绝服务等）
- 受影响的组件和版本
- 复现步骤
- 概念验证代码（如果有）
- 潜在影响评估

### 响应时间

- 我们会在 **48 小时内** 确认收到你的报告
- 我们会在 **7 天内** 提供初步评估
- 我们会在修复发布后公开致谢（除非你希望保持匿名）

## 安全最佳实践

使用 Pulsing 时，请注意：

1. **网络隔离**：生产环境中，Actor System 应运行在受保护的网络中
2. **消息验证**：对不受信任来源的消息进行验证
3. **资源限制**：配置适当的 Mailbox 大小和超时时间
4. **日志脱敏**：避免在日志中记录敏感信息

## 已知安全限制

- Actor 间消息目前未加密（计划在未来版本支持 TLS）
- Gossip 协议假设网络环境可信

感谢你帮助保持 Pulsing 的安全！
