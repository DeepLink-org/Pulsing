# Bash 示例脚本

这个目录包含用于测试和演示 Pulsing 功能的 Bash 脚本。

## 脚本列表

### `demo_actor_list.sh`

演示如何在应用中使用 actor list 功能查看当前运行的 actors。

**功能演示：**
- 在应用启动后查看 actor 列表
- 默认模式：只显示用户创建的 actors
- `--all_actors` 模式：显示所有 actors（包括系统内部 actors）
- JSON 输出格式
- 底层 API 使用（`system.local_actor_names()`）

**使用方法：**

```bash
cd examples/bash
./demo_actor_list.sh
```

或从项目根目录：

```bash
bash examples/bash/demo_actor_list.sh
```

**输出示例：**

```
======================================================================
  Pulsing Actor List 演示
======================================================================

Python: Python 3.12.11

运行演示...

================================================================================
演示：在应用中使用 pulsing actor list
================================================================================

1. 初始化 actor system...
   ✓ 系统启动: 0.0.0.0:49724

2. 创建业务 actors...
   ✓ 创建了 3 个 actors

3. 使用 Python API 查看 actors:
   ----------------------------------------------------------------------------
   本地 actors: calculator, counter-1, counter-2

4. 使用 CLI 格式化输出（只显示用户 actors）:
   ----------------------------------------------------------------------------
Name                           Type            Uptime       Code Path
-----------------------------------------------------------------------------------------------------------
counter-1                      user            0s           -
counter-2                      user            0s           -
calculator                     user            0s           -

Total: 3 actor(s)

...
```

**重要说明：**

`pulsing actor list` 是设计用于在**运行中的应用进程内**调用的管理功能，而不是独立的命令行工具。这是因为：

1. Actor system 是进程本地的，需要在同一进程中才能访问
2. 这种设计更适合集成到应用的管理接口中
3. 对于外部查看远程集群，应使用 `pulsing inspect --seeds <address>`

**在应用中集成：**

```python
from pulsing.actor import init, get_system
from pulsing.cli.actor_list import list_actors_impl

await init()
# ... 创建 actors ...

# 在管理端点或 REPL 中调用
await list_actors_impl(all_actors=False, output_format='table')
```

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
