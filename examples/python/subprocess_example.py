#!/usr/bin/env python3
"""
pulsing.subprocess — 平替 subprocess 示例

只需切换 import，进程就从本地变成远程 actor，业务代码一行不变：

    import subprocess                        # 本地运行
    import pulsing.subprocess as subprocess  # 远程 actor 运行

用法:
    python subprocess_example.py           # 标准库 subprocess（本地）
    python subprocess_example.py --remote  # pulsing.subprocess（远程 actor）
"""

import sys
import asyncio

if "--remote" in sys.argv:
    import pulsing as pul
    import pulsing.subprocess as subprocess
else:
    import subprocess

    pul = None

PIPE = subprocess.PIPE


def run_demos():
    resource_kwargs = {"resources": {"num_cpus": 2}} if pul else {}

    # run() — 等价于 subprocess.run()
    result = subprocess.run(["echo", "hello"], capture_output=True, **resource_kwargs)
    print("run()        →", result.stdout.decode().strip())

    # check_output() — 等价于 subprocess.check_output()
    hostname = subprocess.check_output(["hostname"])
    print("check_output →", hostname.decode().strip())

    # Popen + communicate() — stdin 管道传数据
    proc = subprocess.Popen(["cat"], stdin=PIPE, stdout=PIPE, **resource_kwargs)
    stdout, _ = proc.communicate(input=b"pipe test")
    print("Popen        →", stdout.decode().strip())

    # 示例1：只读取输出
    proc = subprocess.Popen(
        ["ls", "-la"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = proc.communicate()
    print(stdout.decode())

    # 示例2：发送输入并读取输出
    proc = subprocess.Popen(
        ["python3", "-c", "print(input())"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate(input=b"Hello World")
    print(stdout.decode())  # Hello World

    # 示例3：设置超时
    try:
        stdout, stderr = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()


async def main():
    if pul:
        await pul.init()

    if pul:
        # pulsing.subprocess 是同步阻塞调用，需在工作线程里执行
        # 与 async 代码里调用任何阻塞 I/O 的标准做法相同
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, run_demos)
    else:
        run_demos()

    if pul:
        await pul.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
