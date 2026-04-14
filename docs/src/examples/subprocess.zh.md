# 子进程

通过与标准库兼容的 API 执行命令。

`pulsing.subprocess` 对齐 Python 的 `subprocess` 接口，并在需要声明资源时，
提供一个可选的 Pulsing 后端执行路径。

## 运行

```bash
python examples/python/subprocess_example.py
python examples/python/subprocess_example.py --resources
USE_POLSING_SUBPROCESS=1 python examples/python/subprocess_example.py --resources
```

## 后端选择

- 不传 `resources` 时，调用直接走 Python 原生 `subprocess`。
- 传了 `resources=...` 但没有设置 `USE_POLSING_SUBPROCESS=1` 时，示例仍然使用原生后端。
- 只有同时满足“传入 `resources`”和“设置 `USE_POLSING_SUBPROCESS=1`”两个条件时，才会切到 Pulsing 后端。

## 代码

```python
import pulsing.subprocess as subprocess

PIPE = subprocess.PIPE
extra = {"resources": {"num_cpus": 2}}

result = subprocess.run(
    ["echo", "hello from run()"],
    capture_output=True,
    text=True,
    check=True,
    **extra,
)

proc = subprocess.Popen(["cat"], stdin=PIPE, stdout=PIPE, text=True, **extra)
stdout, _ = proc.communicate(input="pipe test")
```

完整示例还覆盖了：

- `check_output()` 的单次命令输出采集
- 带 stdin/stdout/stderr 的 `Popen(...).communicate()`
- `TimeoutExpired` 超时处理
- 基于持久 `/bin/sh` 的多轮 shell 会话

## 要点

- 建议使用 `import pulsing.subprocess as subprocess`，保持和标准库一致的调用方式。
- 资源调度路径是显式开启的，已有 `subprocess` 风格代码可以渐进迁移。
- 在资源后端模式下，模块会懒初始化 Pulsing，无需调用方提前 `await pul.init()`。
- 这些同步包装器不能在当前活跃的 Pulsing event loop 线程里直接调用。
  如果在 async 代码里需要它们，请用 `asyncio.to_thread(...)`，或者直接改用
  async API。
