# Bash 示例脚本

这个目录包含用于测试和演示 Pulsing 功能的 Bash 脚本。

## 当前状态

此目录暂无活跃的演示脚本。

如需查看/管理 actors，请使用 `pulsing inspect` 命令（观察者模式，不加入 gossip 集群）：

```bash
# 查询单个节点的 actors
pulsing inspect actors --endpoint 127.0.0.1:8000

# 查询整个集群的 actors
pulsing inspect actors --seeds 127.0.0.1:8000

# 查看集群状态
pulsing inspect cluster --seeds 127.0.0.1:8000

# 实时监视
pulsing inspect watch --seeds 127.0.0.1:8000
```

更多 CLI 用法参见 [CLI 命令文档](../../docs/src/guide/operations.zh.md)。

## 环境要求

- Python 3.11+
- pyenv（脚本会自动使用 pyenv 的 Python）
- 已安装 Pulsing 包或设置正确的 PYTHONPATH
- Bash shell

## 开发新脚本

创建新的演示脚本时，请遵循以下规范：

1. **文件命名**：`demo_<feature>.sh`
2. **添加注释**：在脚本开头说明功能
3. **错误处理**：使用 `set -e` 和适当的错误检查
4. **使用 pyenv**：通过 `pyenv exec python` 调用 Python
5. **可执行权限**：`chmod +x <script>.sh`
6. **清理临时文件**：脚本结束时清理创建的临时文件
7. **更新此 README**：添加新脚本的说明
