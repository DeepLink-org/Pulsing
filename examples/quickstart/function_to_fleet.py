import asyncio
import os
import time
from pulsing.actor import remote
from pulsing.agent import runtime


@remote
class Worker:
    async def run(self, x: int) -> int:
        await asyncio.sleep(0.02)  # simulate I/O
        return x * x


async def main():
    n = int(os.getenv("WORKERS", "8"))
    m = int(os.getenv("ITEMS", "200"))
    async with runtime():
        ws = [await Worker.spawn(name=f"w{i}") for i in range(n)]
        t0 = time.perf_counter()
        res = await asyncio.gather(*(ws[i % n].run(i) for i in range(m)))
        dt = time.perf_counter() - t0
        print("\n" + "=" * 50)
        print("⚡ Function → Fleet Result")
        print("=" * 50)
        print(f"   Workers:     {n}")
        print(f"   Tasks:       {m}")
        print(f"   Duration:    {dt:.2f}s")
        print(f"   Throughput:  {m / dt:.1f} qps")
        print("=" * 50)
        print("✅ Same code, more workers = higher throughput")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
