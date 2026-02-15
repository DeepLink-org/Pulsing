"""
🛡️ Chaos-proof - Actor 崩溃自动重启，任务不丢失
"""

import asyncio
import random
import pulsing as pul


@pul.remote(restart_policy="on_failure", max_restarts=50)
class FlakyWorker:
    def __init__(self):
        self.call_count = 0

    def work(self, x: int) -> int:
        self.call_count += 1
        if random.random() < 0.3:  # 30% 概率崩溃
            raise RuntimeError(f"boom at call {self.call_count}")
        return x + 1


async def main():
    await pul.init()
    try:
        w = await FlakyWorker.spawn(name="flaky")

        results, retries = [], 0
        for i in range(50):
            for attempt in range(10):  # 最多重试 10 次
                try:
                    results.append(await w.work(i))
                    break
                except Exception:
                    retries += 1
                    await asyncio.sleep(0.01)
            else:
                results.append(None)  # 真的失败了

        ok = sum(1 for r in results if r is not None)
        print("\n" + "=" * 50)
        print("🛡️  Chaos-proof Result")
        print("=" * 50)
        print("   Total tasks:   50")
        print(f"   Succeeded:     {ok}")
        print(f"   Retries:       {retries}")
        print("   Crash rate:    30%")
        print("=" * 50)
        if ok == 50:
            print("✅ All succeeded! Actor auto-restarted on crash.")
        else:
            print(f"⚠️  {50 - ok} tasks failed")
        print("=" * 50 + "\n")
    finally:
        await pul.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
