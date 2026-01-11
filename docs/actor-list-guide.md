# Actor List 命令使用指南

`pulsing actor list` 命令用于列出当前 Actor 系统中的 actors。

## 基本用法

### 列出用户 actors（默认）

```bash
pulsing actor list
```

默认情况下，只显示用户创建的命名 actors，不包括以 `_` 开头的系统内部 actors。

输出示例：

```
Name                           Type            Uptime       Code Path
---------------------------------------------------------------------------------------------------
counter-1                      user            5m 23s       -
counter-2                      user            5m 23s       -
calculator                     user            5m 23s       -

Total: 3 actor(s)
```

### 列出所有 actors（包括系统 actors）

```bash
pulsing actor list --all_actors True
```

包括系统内部的 actors：

```
Name                           Type            Uptime       Code Path
---------------------------------------------------------------------------------------------------
counter-1                      user            5m 23s       -
_system_internal               system          5m 30s       -
_python_actor_service          system          5m 30s       -

Total: 5 actor(s)
```

### JSON 输出格式

```bash
pulsing actor list --json True
```

以 JSON 格式输出，方便脚本处理：

```json
[
  {
    "name": "counter-1",
    "type": "user",
    "code_path": null,
    "uptime": "5m 23s"
  },
  {
    "name": "counter-2",
    "type": "user",
    "code_path": null,
    "uptime": "5m 23s"
  }
]
```

## 在 Python 代码中使用

`pulsing actor list` CLI 命令需要在运行 actor system 的进程内调用。更常见的用法是直接在 Python 代码中使用：

```python
import asyncio
from pulsing.actor import init, remote, get_system
from pulsing.cli.actor_list import list_actors_impl


@remote
class Counter:
    def __init__(self):
        self.count = 0


async def main():
    # 初始化系统
    await init()
    system = get_system()

    # 创建一些 actors
    await Counter.remote(system, name="counter-1")
    await Counter.remote(system, name="counter-2")

    # 列出 actors
    await list_actors_impl(all_actors=False, output_format="table")

    # 或者直接使用底层 API
    actor_names = system.local_actor_names()
    user_actors = [n for n in actor_names if not n.startswith("_")]
    print(f"User actors: {user_actors}")


if __name__ == "__main__":
    asyncio.run(main())
```

## 字段说明

- **Name**: Actor 的名称
- **Type**: Actor 类型
  - `user`: 用户创建的 actors
  - `system`: 系统内部 actors
- **Uptime**: Actor 运行时间（当前为系统启动时间的近似值）
- **Code Path**: Python 类的代码路径（当前版本暂未实现，显示为 `-`）

## 未来改进

- [ ] 显示每个 actor 的精确创建时间/运行时间
- [ ] 显示 Python actor 的类型（类名）和代码路径
- [ ] 显示 actor 的消息处理统计（处理数量、错误数等）
- [ ] 支持通过 `--seeds` 参数查询远程集群的 actors
- [ ] 支持过滤和搜索（按名称、类型等）
