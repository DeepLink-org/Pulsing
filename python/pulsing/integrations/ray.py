"""
pulsing.ray - 在 Ray 集群中初始化 Pulsing

每个 Ray worker 进程调用 init_in_ray() 即可启动 Pulsing 并自动加入集群。
通过 Ray 的 internal KV store 协调 seed 节点发现。

推荐用法:
    import ray
    from pulsing.integrations.ray import init_in_ray

    ray.init(runtime_env={"worker_process_setup_hook": init_in_ray})
    init_in_ray()  # driver 进程也需要初始化
"""

try:
    import ray
    from ray.experimental.internal_kv import (
        _internal_kv_get,
        _internal_kv_put,
        _internal_kv_del,
    )
except ImportError:
    raise ImportError(
        "pulsing.integrations.ray requires Ray. Install with: pip install 'ray[default]'"
    )

import asyncio
import threading

_SEED_KEY = "pulsing:seed_addr"

# 后台事件循环（供 sync init 使用）
_loop = None
_thread = None


def _get_node_ip():
    """获取当前 Ray 节点 IP"""
    ctx = ray.get_runtime_context()
    node_id = ctx.get_node_id()
    for node in ray.nodes():
        if node["NodeID"] == node_id and node["Alive"]:
            return node["NodeManagerAddress"]
    raise RuntimeError("无法获取当前 Ray 节点 IP")


def _start_background_loop():
    """启动后台事件循环线程"""
    global _loop, _thread
    if _thread is not None:
        return

    ready = threading.Event()

    def _run():
        global _loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _loop = loop
        ready.set()
        loop.run_forever()

    _thread = threading.Thread(target=_run, daemon=True, name="pulsing-event-loop")
    _thread.start()
    ready.wait()


def _run_sync(coro):
    """在后台事件循环中同步执行协程"""
    fut = asyncio.run_coroutine_threadsafe(coro, _loop)
    return fut.result(timeout=30)


async def _do_init(addr, seeds=None):
    from pulsing.core import init

    return await init(addr=addr, seeds=seeds)


async def _do_shutdown():
    from pulsing.core import shutdown

    await shutdown()


def _get_seed():
    """从 Ray KV store 获取 seed 地址"""
    data = _internal_kv_get(_SEED_KEY)
    return data.decode() if data else None


def _try_set_seed(addr):
    """原子写入 seed 地址，返回 True 表示写入成功（我是 seed）。

    _internal_kv_put(overwrite=False) 返回值语义：
        False = key 不存在，已写入（成功）
        True  = key 已存在，未覆盖（失败）
    """
    already_exists = _internal_kv_put(_SEED_KEY, addr.encode(), overwrite=False)
    return not already_exists


def init_in_ray():
    """在当前进程初始化 Pulsing 并加入集群。

    可直接调用，也可作为 Ray worker_process_setup_hook:

        ray.init(runtime_env={"worker_process_setup_hook": init_in_ray})
        init_in_ray()  # driver 也需要
    """
    if not ray.is_initialized():
        raise RuntimeError("Ray 未初始化，请先调用 ray.init()")

    node_ip = _get_node_ip()
    _start_background_loop()

    # 已有 seed → 直接加入
    seed_addr = _get_seed()
    if seed_addr is not None:
        return _run_sync(_do_init(f"{node_ip}:0", seeds=[seed_addr]))

    # 启动为潜在 seed
    system = _run_sync(_do_init(f"{node_ip}:0"))
    my_addr = str(system.addr)

    if _try_set_seed(my_addr):
        return system  # 写入成功，我是 seed

    # 竞争失败（极罕见），重新加入实际 seed
    _run_sync(_do_shutdown())
    return _run_sync(_do_init(f"{node_ip}:0", seeds=[_get_seed()]))


async def async_init_in_ray():
    """在当前进程初始化 Pulsing 并加入集群（异步版本）。

    适用于 async Ray actor。
    """
    if not ray.is_initialized():
        raise RuntimeError("Ray 未初始化，请先调用 ray.init()")

    node_ip = _get_node_ip()

    seed_addr = _get_seed()
    if seed_addr is not None:
        return await _do_init(f"{node_ip}:0", seeds=[seed_addr])

    system = await _do_init(f"{node_ip}:0")
    my_addr = str(system.addr)

    if _try_set_seed(my_addr):
        return system

    await _do_shutdown()
    return await _do_init(f"{node_ip}:0", seeds=[_get_seed()])


def cleanup():
    """清理 Pulsing 在 Ray KV store 中的状态"""
    _internal_kv_del(_SEED_KEY)


__all__ = ["init_in_ray", "async_init_in_ray", "cleanup", "_get_seed", "_loop"]
