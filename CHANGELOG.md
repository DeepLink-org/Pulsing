# Changelog

本项目的所有重要更改都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added
- 初始版本的 Pulsing Actor System
- 基于 Gossip 协议的集群发现（SWIM）
- HTTP/2 传输层
- PyO3 Python 绑定
- 流式消息支持（StreamMessage）
- CLI 工具（`pulsing` 命令）

### Features
- **Actor System**
  - 位置透明的 ActorRef
  - Ask/Tell 消息模式
  - Actor 生命周期管理（on_start/on_stop）
  - 流式请求和响应
- **Cluster**
  - 零外部依赖的 Gossip 协议
  - 自动故障检测
  - Actor 位置注册和发现
- **Python API**
  - 完整的异步 Python API
  - 与 asyncio 原生集成
  - LLM serving actors（Router、Worker）

## [0.1.0] - 2024-12-24

### Added
- 项目初始化
- Actor System 核心实现
- Python 绑定
- 基础文档和示例

[Unreleased]: https://github.com/DeepLink-org/Pulsing/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DeepLink-org/Pulsing/releases/tag/v0.1.0
