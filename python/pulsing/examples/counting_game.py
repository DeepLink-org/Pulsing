"""Counting Game — Pulsing distributed counting game

20 nodes count in sequence and broadcast, demonstrating Pulsing's actor messaging capability.
Ray is only used for multiprocessing; counting logic is entirely handled by Pulsing actors.

Usage:
    python -m pulsing.examples.counting_game
    python -m pulsing.examples.counting_game --num-workers 10
"""

import os
import time

import ray

import pulsing as pul


# ── Counting Actor ───────────────────────────────────────────


@ray.remote
class Counter:
    """Each node holds its name, ordered peer list, and counting log."""

    def __init__(self, name, peers):
        self.name = name
        self.peers = sorted(peers)
        self.log = []
        pul.mount(self, name=name)  # One line to join Pulsing network

    async def yield_number(self):
        """Yield number: broadcast own number to all nodes"""
        num = self.peers.index(self.name) + 1
        for peer in self.peers:
            proxy = await pul.resolve(peer, cls=Counter, timeout=30)
            await proxy.on_number(num, self.name)

    async def on_number(self, num, from_who):
        """Receive number: log it, relay if previous node finished"""
        self.log.append({"number": num, "from": from_who})
        idx = self.peers.index(self.name)
        if idx > 0 and from_who == self.peers[idx - 1]:
            await self.yield_number()

    def get_pid(self):
        return os.getpid()

    def get_log(self):
        return list(self.log)


# ── Run ─────────────────────────────────────────────────


def run(num_workers=20):
    """Run counting game (requires Ray initialized). Returns logs from all nodes, raises on failure."""
    names = [f"node_{i:02d}" for i in range(num_workers)]
    t0 = time.time()

    # 1) Create Ray actors (auto pul.mount in __init__ to join Pulsing)
    print(f"[counting_game] Starting {num_workers} nodes ...")
    actors = [Counter.remote(name, names) for name in names]
    pids = ray.get([a.get_pid.remote() for a in actors])
    assert len(set(pids)) == num_workers, "Not enough worker processes"
    print(f"[counting_game] {num_workers} nodes ready ({time.time()-t0:.1f}s)")

    # 2) node_00 yields -> auto relays to node_19
    print("[counting_game] node_00 starting count ...")
    ray.get(actors[0].yield_number.remote())

    # 3) Wait for all nodes to collect complete logs
    deadline = time.time() + 30
    while time.time() < deadline:
        logs = ray.get([a.get_log.remote() for a in actors])
        done = sum(1 for lg in logs if len(lg) == num_workers)
        print(
            f"\r[counting_game] Collecting logs {done}/{num_workers}",
            end="",
            flush=True,
        )
        if done == num_workers:
            break
        time.sleep(0.5)
    else:
        raise TimeoutError("Counting timeout")
    print()

    # 4) Verify: each log entry's 'from' should match the number
    for entries in logs:
        for e in entries:
            assert e["from"] == f"node_{e['number']-1:02d}"

    # 5) Print results
    order = " → ".join(f"{i+1}:{names[i]}" for i in range(min(5, num_workers)))
    if num_workers > 5:
        order += f" → ... → {num_workers}:{names[-1]}"
    elapsed = time.time() - t0
    print(f"[counting_game] Counting order: {order}")
    print(
        f"[counting_game] Passed! {num_workers}x{num_workers}={num_workers**2} messages, {elapsed:.1f}s"
    )
    pul.cleanup_ray()
    return logs


# ── CLI ──────────────────────────────────────────────────


def main():
    import argparse

    p = argparse.ArgumentParser(description="Pulsing distributed counting game")
    p.add_argument("--num-workers", type=int, default=20)
    args = p.parse_args()

    ray.init(num_cpus=args.num_workers + 1)
    try:
        run(args.num_workers)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
