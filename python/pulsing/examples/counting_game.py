"""Counting Game — Pulsing 分布式报数游戏

20 个节点依次报数并广播，演示 Pulsing 的 actor 消息通信能力。
Ray 仅用于启动多进程，报数逻辑完全由 Pulsing actor 完成。

运行:
    python -m pulsing.examples.counting_game
    python -m pulsing.examples.counting_game --num-workers 10
"""

import os
import time

import ray

import pulsing as pul


# ── 报数 Actor ───────────────────────────────────────────


@ray.remote
class Counter:
    """每个节点持有名字、有序节点列表、报数日志。"""

    def __init__(self, name, peers):
        self.name = name
        self.peers = sorted(peers)
        self.log = []
        pul.mount(self, name=name)  # 一行接入 Pulsing 网络

    async def yield_number(self):
        """报数：广播自己的编号给所有节点"""
        num = self.peers.index(self.name) + 1
        for peer in self.peers:
            proxy = (await pul.resolve(peer, timeout=30)).as_type(Counter)
            await proxy.on_number(num, self.name)

    async def on_number(self, num, from_who):
        """收到报数：记录，前序节点报完则接力"""
        self.log.append({"number": num, "from": from_who})
        idx = self.peers.index(self.name)
        if idx > 0 and from_who == self.peers[idx - 1]:
            await self.yield_number()

    def get_pid(self):
        return os.getpid()

    def get_log(self):
        return list(self.log)


# ── 运行 ─────────────────────────────────────────────────


def run(num_workers=20):
    """运行报数游戏（需要 Ray 已初始化）。返回各节点日志，失败抛异常。"""
    names = [f"node_{i:02d}" for i in range(num_workers)]
    t0 = time.time()

    # 1) 创建 Ray actor（__init__ 中自动 pul.mount 接入 Pulsing）
    print(f"[counting_game] 启动 {num_workers} 个节点 ...")
    actors = [Counter.remote(name, names) for name in names]
    pids = ray.get([a.get_pid.remote() for a in actors])
    assert len(set(pids)) == num_workers, "worker 进程数不足"
    print(f"[counting_game] {num_workers} 节点就绪 ({time.time()-t0:.1f}s)")

    # 2) node_00 报数 → 自动接力至 node_19
    print("[counting_game] node_00 开始报数 ...")
    ray.get(actors[0].yield_number.remote())

    # 3) 等待所有节点收齐日志
    deadline = time.time() + 30
    while time.time() < deadline:
        logs = ray.get([a.get_log.remote() for a in actors])
        done = sum(1 for lg in logs if len(lg) == num_workers)
        print(f"\r[counting_game] 收集日志 {done}/{num_workers}", end="", flush=True)
        if done == num_workers:
            break
        time.sleep(0.5)
    else:
        raise TimeoutError("报数超时")
    print()

    # 4) 验证：每条日志的 from 应与报数序号对应
    for entries in logs:
        for e in entries:
            assert e["from"] == f"node_{e['number']-1:02d}"

    # 5) 打印结果
    order = " → ".join(f"{i+1}:{names[i]}" for i in range(min(5, num_workers)))
    if num_workers > 5:
        order += f" → ... → {num_workers}:{names[-1]}"
    elapsed = time.time() - t0
    print(f"[counting_game] 报数顺序: {order}")
    print(
        f"[counting_game] 通过! {num_workers}x{num_workers}={num_workers**2} 条消息, {elapsed:.1f}s"
    )
    pul.cleanup_ray()
    return logs


# ── CLI ──────────────────────────────────────────────────


def main():
    import argparse

    p = argparse.ArgumentParser(description="Pulsing 分布式报数游戏")
    p.add_argument("--num-workers", type=int, default=20)
    args = p.parse_args()

    ray.init(num_cpus=args.num_workers + 1)
    try:
        run(args.num_workers)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
