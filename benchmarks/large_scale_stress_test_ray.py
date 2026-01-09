#!/usr/bin/env python3
"""
大规模压测脚本 - Ray版本
等价于 large_scale_stress_test.py，用于性能对比

使用方法:
    torchrun --nproc_per_node=10 benchmarks/large_scale_stress_test_ray.py \
        --duration 300 \
        --rate 100
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

try:
    import ray  # type: ignore
except ImportError:
    print("Error: Ray is not installed. Please install it with: pip install ray")
    sys.exit(1)


# ============================================================================
# 压测统计
# ============================================================================


@dataclass
class StressTestStats:
    """压测统计信息"""

    total_requests: int = 0
    total_streams: int = 0
    successful_requests: int = 0
    successful_streams: int = 0
    failed_requests: int = 0
    failed_streams: int = 0
    total_latency_ms: float = 0.0
    total_stream_latency_ms: float = 0.0
    request_latencies: list[float] = field(default_factory=list)
    stream_latencies: list[float] = field(default_factory=list)
    errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    health_warnings: list[str] = field(default_factory=list)

    def add_health_warning(self, warning: str):
        """添加健康警告"""
        self.health_warnings.append(f"{time.time()}: {warning}")

    def add_request(self, success: bool, latency_ms: float, error: str | None = None):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            self.total_latency_ms += latency_ms
            self.request_latencies.append(latency_ms)
        else:
            self.failed_requests += 1
            if error:
                self.errors[error] += 1

    def add_stream(self, success: bool, latency_ms: float, error: str | None = None):
        self.total_streams += 1
        if success:
            self.successful_streams += 1
            self.total_stream_latency_ms += latency_ms
            self.stream_latencies.append(latency_ms)
        else:
            self.failed_streams += 1
            if error:
                self.errors[error] += 1

    def get_summary(self) -> dict:
        """获取统计摘要"""
        avg_latency = (
            self.total_latency_ms / self.successful_requests
            if self.successful_requests > 0
            else 0.0
        )
        avg_stream_latency = (
            self.total_stream_latency_ms / self.successful_streams
            if self.successful_streams > 0
            else 0.0
        )

        request_latencies_sorted = sorted(self.request_latencies)
        stream_latencies_sorted = sorted(self.stream_latencies)

        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]

        return {
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "success_rate": (
                    self.successful_requests / self.total_requests * 100
                    if self.total_requests > 0
                    else 0.0
                ),
                "avg_latency_ms": avg_latency,
                "p50_latency_ms": percentile(request_latencies_sorted, 50),
                "p95_latency_ms": percentile(request_latencies_sorted, 95),
                "p99_latency_ms": percentile(request_latencies_sorted, 99),
            },
            "streams": {
                "total": self.total_streams,
                "successful": self.successful_streams,
                "failed": self.failed_streams,
                "success_rate": (
                    self.successful_streams / self.total_streams * 100
                    if self.total_streams > 0
                    else 0.0
                ),
                "avg_latency_ms": avg_stream_latency,
                "p50_latency_ms": percentile(stream_latencies_sorted, 50),
                "p95_latency_ms": percentile(stream_latencies_sorted, 95),
                "p99_latency_ms": percentile(stream_latencies_sorted, 99),
            },
            "errors": dict(self.errors),
            "health_warnings": self.health_warnings[-10:],  # 只保留最后10个警告
        }


# ============================================================================
# 5种不同的Worker Actor (Ray版本)
# ============================================================================


@ray.remote
class EchoWorker:
    """Echo Worker - 简单回显"""

    async def echo(self, text: str) -> dict:
        """异步方法，与Pulsing版本等价"""
        return {"echo": text}


@ray.remote
class ComputeWorker:
    """Compute Worker - 计算密集型"""

    async def compute(self, n: int) -> dict:
        """异步方法，与Pulsing版本等价"""
        # 模拟计算
        result = sum(i * i for i in range(n))
        return {"result": result}


@ray.remote
class StreamWorker:
    """Stream Worker - 流式响应"""

    async def generate_stream(self, count: int, delay: float) -> list[dict]:
        """生成流式数据（使用异步，与Pulsing版本等价）

        注意：Ray支持async def，可以使用asyncio.sleep实现真正的异步延迟。
        这比time.sleep更高效，因为不会阻塞线程。
        """
        result = []
        for i in range(count):
            result.append(
                {
                    "index": i,
                    "value": f"item_{i}",
                    "timestamp": time.time(),
                }
            )
            # 使用asyncio.sleep，与Pulsing版本完全等价
            await asyncio.sleep(delay)
        return result


@ray.remote
class BatchWorker:
    """Batch Worker - 批量处理"""

    def __init__(self):
        self.batch = []
        self.batch_size = 10

    async def batch_add(self, item: int) -> dict:
        """异步方法，与Pulsing版本等价"""
        self.batch.append(item)

        if len(self.batch) >= self.batch_size:
            result = sum(self.batch)
            self.batch = []
            return {"sum": result}
        return {"count": len(self.batch)}


@ray.remote
class StatefulWorker:
    """Stateful Worker - 有状态处理"""

    def __init__(self):
        self.state = {}
        self.counter = 0

    async def set_state(self, key: str, value: int) -> dict:
        """异步方法，与Pulsing版本等价"""
        self.state[key] = value
        self.counter += 1
        return {"counter": self.counter}

    async def get_state(self, key: str) -> dict:
        """异步方法，与Pulsing版本等价"""
        value = self.state.get(key)
        return {"key": key, "value": value}


# Worker类型映射
WORKER_TYPES = {
    "echo": EchoWorker,
    "compute": ComputeWorker,
    "stream": StreamWorker,
    "batch": BatchWorker,
    "stateful": StatefulWorker,
}


# ============================================================================
# 压测客户端
# ============================================================================


class StressTestClient:
    """压测客户端"""

    def __init__(
        self,
        worker_refs: dict[str, ray.actor.ActorHandle],
        stats: StressTestStats,
        rate: float = 100.0,
        expected_nodes: int = 10,
        expected_worker_types: list[str] = None,
    ):
        self.worker_refs = worker_refs
        self.stats = stats
        self.rate = rate  # 每秒请求数
        self.interval = 1.0 / rate if rate > 0 else 0.0
        self.running = True
        self.expected_nodes = expected_nodes
        self.expected_worker_types = expected_worker_types or [
            "echo",
            "compute",
            "stream",
            "batch",
            "stateful",
        ]
        self.health_check_interval = 30.0  # 每30秒检查一次健康状态
        self.remote_requests = 0  # 远程请求计数
        self.local_requests = 0  # 本地请求计数

    def _extract_base_worker_type(self, worker_type: str) -> str:
        """从worker_type中提取基础类型"""
        if "_remote_" in worker_type:
            return worker_type.split("_remote_")[0]

        if worker_type in ["echo", "compute", "stream", "batch", "stateful"]:
            return worker_type

        parts = worker_type.split("_")
        if len(parts) > 0 and parts[0] in [
            "echo",
            "compute",
            "stream",
            "batch",
            "stateful",
        ]:
            return parts[0]

        return worker_type

    async def send_single_request(self, worker_type: str) -> bool:
        """发送单个请求（异步，与Pulsing版本等价）"""
        if worker_type not in self.worker_refs:
            return False

        worker_ref = self.worker_refs[worker_type]
        start_time = time.time()

        try:
            base_type = self._extract_base_worker_type(worker_type)

            if base_type not in ["echo", "compute", "batch", "stateful"]:
                return False

            # 生成请求参数
            # Ray的remote()返回ObjectRef，对于async方法，可以直接await ObjectRef
            if base_type == "echo":
                payload = {"text": f"echo_{random.randint(1, 1000)}"}
                result = await worker_ref.echo.remote(payload["text"])
            elif base_type == "compute":
                payload = {"n": random.randint(100, 10000)}
                result = await worker_ref.compute.remote(payload["n"])
            elif base_type == "batch":
                payload = {"item": random.randint(1, 100)}
                result = await worker_ref.batch_add.remote(payload["item"])
            elif base_type == "stateful":
                msg_type = random.choice(["SetState", "GetState"])
                if msg_type == "SetState":
                    payload = {
                        "key": f"key_{random.randint(1, 100)}",
                        "value": random.randint(1, 1000),
                    }
                    result = await worker_ref.set_state.remote(
                        payload["key"], payload["value"]
                    )
                else:
                    payload = {"key": f"key_{random.randint(1, 100)}"}
                    result = await worker_ref.get_state.remote(payload["key"])

            if result is None:
                raise ValueError("Empty response from actor")

            latency_ms = (time.time() - start_time) * 1000
            self.stats.add_request(True, latency_ms)
            return True

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = str(e)[:100]
            self.stats.add_request(False, latency_ms, error_msg)
            return False

    async def send_stream_request(self, worker_type: str) -> bool:
        """发送流式请求（异步，与Pulsing版本等价）"""
        base_type = self._extract_base_worker_type(worker_type)
        if base_type != "stream":
            return False

        stream_worker_key = None
        if worker_type in self.worker_refs and base_type == "stream":
            stream_worker_key = worker_type
        elif "stream" in self.worker_refs:
            stream_worker_key = "stream"
        else:
            for key in self.worker_refs.keys():
                if self._extract_base_worker_type(key) == "stream":
                    stream_worker_key = key
                    break

        if stream_worker_key is None or stream_worker_key not in self.worker_refs:
            return False

        worker_ref = self.worker_refs[stream_worker_key]
        start_time = time.time()

        try:
            count = random.randint(5, 20)
            delay = random.uniform(0.01, 0.05)

            # Ray的流式处理：使用异步方法，与Pulsing版本等价
            # Ray的remote()返回ObjectRef，可以直接await获取结果
            stream_items = await worker_ref.generate_stream.remote(count, delay)

            chunk_count = 0
            for _chunk in stream_items:
                chunk_count += 1

            latency_ms = (time.time() - start_time) * 1000
            self.stats.add_stream(True, latency_ms)
            return True

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = str(e)[:100]
            self.stats.add_stream(False, latency_ms, error_msg)
            return False

    async def check_cluster_health(self) -> tuple[bool, str]:
        """检查集群健康状态"""
        try:
            # Ray集群信息
            cluster_resources = ray.cluster_resources()
            _available_resources = ray.available_resources()  # noqa: F841

            # 检查节点数量（通过资源信息推断）
            num_nodes = cluster_resources.get("node:__internal__:__head_node__", 0)
            if num_nodes == 0:
                # 尝试其他方式获取节点数
                num_nodes = len(
                    [k for k in cluster_resources.keys() if k.startswith("node:")]
                )

            # 简化健康检查：检查worker是否可用
            missing_workers = []
            for worker_type in self.expected_worker_types:
                if worker_type not in self.worker_refs:
                    missing_workers.append(worker_type)

            if missing_workers:
                return False, f"Missing workers: {missing_workers}"

            return True, f"Cluster healthy with {len(self.worker_refs)} workers"

        except Exception as e:
            return False, f"Health check error: {e}"

    async def run_stress_test(self, duration: float):
        """运行压测"""
        end_time = time.time() + duration
        last_health_check = time.time()
        report_interval = 10.0

        print(f"[StressTest] Starting stress test for {duration}s at {self.rate} req/s")
        print(
            f"[StressTest] Expected: {self.expected_nodes} nodes, {len(self.expected_worker_types)} worker types per node"
        )

        # 初始健康检查
        is_healthy, health_msg = await self.check_cluster_health()
        if not is_healthy:
            print(f"[StressTest] ⚠️  WARNING: Initial health check failed: {health_msg}")
        else:
            print(f"[StressTest] ✓ Initial health check passed: {health_msg}")

        async def worker_loop():
            """工作循环（异步版本，与Pulsing版本等价）"""
            local_worker_types = ["echo", "compute", "stream", "batch", "stateful"]
            local_workers = [wt for wt in local_worker_types if wt in self.worker_refs]
            remote_workers = [wt for wt in self.worker_refs.keys() if "_remote_" in wt]

            if not local_workers and not remote_workers:
                print("[StressTest] Warning: No workers available!")
                return

            use_remote_probability = 0.7

            while self.running and time.time() < end_time:
                # 选择worker
                if remote_workers and (
                    not local_workers or random.random() < use_remote_probability
                ):
                    worker_type = random.choice(remote_workers)
                    self.remote_requests += 1
                elif local_workers:
                    worker_type = random.choice(local_workers)
                    self.local_requests += 1
                else:
                    worker_type = random.choice(remote_workers)
                    self.remote_requests += 1

                # 随机选择single或stream
                if random.random() < 0.7:
                    await self.send_single_request(worker_type)
                else:
                    base_type = self._extract_base_worker_type(worker_type)
                    if base_type == "stream":
                        await self.send_stream_request(worker_type)
                    else:
                        await self.send_single_request(worker_type)

                # 控制速率（使用asyncio.sleep，与Pulsing版本等价）
                if self.interval > 0:
                    await asyncio.sleep(self.interval)

        # 启动多个并发worker（使用asyncio协程，与Pulsing版本等价）
        num_workers = max(1, int(self.rate / 10))
        tasks = [asyncio.create_task(worker_loop()) for _ in range(num_workers)]

        # 定期报告和健康检查
        async def report_loop():
            nonlocal last_health_check
            while self.running and time.time() < end_time:
                await asyncio.sleep(report_interval)
                summary = self.stats.get_summary()
                print("\n[StressTest] Progress Report:")
                print(
                    f"  Requests: {summary['requests']['total']} "
                    f"(success: {summary['requests']['successful']}, "
                    f"failed: {summary['requests']['failed']})"
                )
                print(
                    f"  Streams: {summary['streams']['total']} "
                    f"(success: {summary['streams']['successful']}, "
                    f"failed: {summary['streams']['failed']})"
                )
                if summary["requests"]["successful"] > 0:
                    print(
                        f"  Avg Latency: {summary['requests']['avg_latency_ms']:.2f}ms"
                    )

                total_worker_requests = self.remote_requests + self.local_requests
                if total_worker_requests > 0:
                    remote_pct = self.remote_requests / total_worker_requests * 100
                    local_pct = self.local_requests / total_worker_requests * 100
                    print(
                        f"  Worker Selection: {remote_pct:.1f}% remote, {local_pct:.1f}% local"
                    )

                if time.time() - last_health_check >= self.health_check_interval:
                    is_healthy, health_msg = await self.check_cluster_health()
                    if is_healthy:
                        print(f"  ✓ Cluster Health: {health_msg}")
                    else:
                        print(f"  ⚠️  Cluster Health Warning: {health_msg}")
                        self.stats.add_health_warning(health_msg)
                    last_health_check = time.time()

        report_task = asyncio.create_task(report_loop())

        # 等待所有任务完成（使用asyncio，与Pulsing版本等价）
        await asyncio.gather(*tasks, report_task)

        self.running = False
        print("\n[StressTest] Stress test completed")


# ============================================================================
# 主函数
# ============================================================================


async def main():
    parser = argparse.ArgumentParser(description="大规模压测脚本 - Ray版本")
    parser.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="压测持续时间（秒）",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=100.0,
        help="每秒请求数（0表示无限制）",
    )
    parser.add_argument(
        "--address",
        type=str,
        default=None,
        help="Ray集群地址（如: ray://head-node:10001）",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="本地rank（torchrun自动设置）",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="总进程数（torchrun自动设置）",
    )
    parser.add_argument(
        "--stabilize-timeout",
        type=float,
        default=10.0,
        help="集群稳定等待时间（秒），默认10秒",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="benchmark_logs",
        help="日志文件目录，默认benchmark_logs",
    )

    args = parser.parse_args()

    # 从环境变量获取torchrun信息
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    rank = int(os.environ.get("RANK", local_rank))

    # 设置日志文件
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"stress_test_ray_rank_{rank}.log")

    # 创建文件日志处理器
    class TeeOutput:
        """同时输出到控制台和文件"""

        def __init__(self, file_path, original_stdout):
            self.file = open(file_path, "w", encoding="utf-8")
            self.original_stdout = original_stdout

        def write(self, text):
            try:
                self.file.write(text)
                self.file.flush()
            except Exception:
                pass
            try:
                self.original_stdout.write(text)
                self.original_stdout.flush()
            except Exception:
                pass

        def flush(self):
            try:
                self.file.flush()
            except Exception:
                pass
            try:
                self.original_stdout.flush()
            except Exception:
                pass

        def fileno(self):
            """返回文件描述符（Ray需要）"""
            return self.original_stdout.fileno()

        def isatty(self):
            """检查是否是终端"""
            return self.original_stdout.isatty()

        def readable(self):
            """检查是否可读"""
            return False

        def writable(self):
            """检查是否可写"""
            return True

        def seekable(self):
            """检查是否可seek"""
            return False

        def close(self):
            try:
                self.file.close()
            except Exception:
                pass

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    tee = TeeOutput(log_file, original_stdout)
    sys.stdout = tee
    sys.stderr = tee

    print(f"\n{'=' * 60}")
    print(f"Stress Test Process {rank}/{world_size} (Ray Version)")
    print(f"Local Rank: {local_rank}, World Size: {world_size}")
    print(f"Log file: {log_file}")
    print(f"{'=' * 60}\n")

    # 初始化Ray
    # ⚠️ 重要：Ray在torchrun多进程环境下，每个进程都是独立的
    # 每个进程都会创建自己的Ray实例，导致资源消耗巨大
    # 解决方案：严格限制每个进程的资源使用
    if not ray.is_initialized():
        if args.address:
            # 连接到指定的Ray集群（推荐方式）
            ray.init(address=args.address, ignore_reinit_error=True)
            print(f"[Process {rank}] Connected to Ray cluster at {args.address}")
        else:
            # 本地模式：每个进程创建独立的Ray实例，但严格限制资源
            # 关键：限制CPU、内存、禁用不必要的功能
            # 每个进程最多使用1个CPU核心，避免资源耗尽
            num_cpus = 1
            # 限制对象存储内存为100MB（非常保守）
            object_store_memory = 100_000_000

            ray.init(
                num_cpus=num_cpus,
                num_gpus=0,  # 不使用GPU
                object_store_memory=object_store_memory,
                ignore_reinit_error=True,
                include_dashboard=False,  # 禁用dashboard，减少资源消耗
                _temp_dir=f"/tmp/ray_rank_{rank}",  # 每个进程独立的临时目录
            )
            print(
                f"[Process {rank}] Ray initialized: {num_cpus} CPU, "
                f"{object_store_memory / 1e6:.0f}MB object store"
            )

    print(f"[Process {rank}] Ray initialized")
    print(f"[Process {rank}] Ray cluster resources: {ray.cluster_resources()}")

    # 等待集群稳定
    stabilize_timeout = args.stabilize_timeout
    print(
        f"[Process {rank}] Waiting for cluster to stabilize ({stabilize_timeout}s)..."
    )
    await asyncio.sleep(stabilize_timeout)

    # 创建5种worker（每个进程都创建）
    worker_refs = {}
    worker_names = ["echo", "compute", "stream", "batch", "stateful"]

    print(f"\n[Process {rank}] Spawning workers...")
    for worker_name in worker_names:
        try:
            worker_class = WORKER_TYPES[worker_name]
            worker_ref = worker_class.options(name=f"{worker_name}_{rank}").remote()
            worker_refs[worker_name] = worker_ref
            print(f"  ✓ Spawned {worker_name} worker")
        except Exception as e:
            print(f"  ✗ Failed to spawn {worker_name} worker: {e}")

    # 等待所有进程启动worker
    print(
        f"[Process {rank}] Waiting for all workers to register ({stabilize_timeout}s)..."
    )
    await asyncio.sleep(stabilize_timeout)

    # 尝试解析其他进程的worker（用于跨进程通信）
    print(f"\n[Process {rank}] Resolving remote workers...")
    resolved_count = 0
    for other_rank in range(world_size):
        if other_rank == rank:
            continue

        for worker_name in worker_names:
            try:
                # Ray通过名称获取actor
                remote_ref = ray.get_actor(f"{worker_name}_{other_rank}")
                worker_refs[f"{worker_name}_remote_{other_rank}"] = remote_ref
                resolved_count += 1
                if resolved_count <= 5:
                    print(f"  ✓ Resolved {worker_name}_{other_rank}")
            except Exception as e:
                if resolved_count < 5:
                    print(f"  ✗ Failed to resolve {worker_name}_{other_rank}: {e}")

    if resolved_count > 5:
        print(f"  ... and {resolved_count - 5} more workers resolved")

    total_expected = (world_size - 1) * len(worker_names)
    if resolved_count < total_expected:
        print(
            f"  ⚠️  Warning: Only resolved {resolved_count}/{total_expected} remote workers"
        )

    # 创建压测客户端
    stats = StressTestStats()
    client = StressTestClient(
        worker_refs,
        stats,
        rate=args.rate,
        expected_nodes=world_size,
        expected_worker_types=["echo", "compute", "stream", "batch", "stateful"],
    )

    # 运行压测
    try:
        await client.run_stress_test(args.duration)
    except KeyboardInterrupt:
        print(f"\n[Process {rank}] Interrupted by user")
        client.running = False
    except Exception as e:
        print(f"\n[Process {rank}] Error during stress test: {e}")
        import traceback

        traceback.print_exc()
        raise

    # 打印最终统计
    print(f"\n{'=' * 60}")
    print(f"Final Statistics - Process {rank} (Ray Version)")
    print(f"{'=' * 60}")
    summary = stats.get_summary()
    print(json.dumps(summary, indent=2))

    # 保存统计到文件
    stats_dir = args.log_dir
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, f"stress_test_stats_ray_rank_{rank}.json")
    with open(stats_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Process {rank}] Statistics saved to {stats_file}")

    # 关闭Ray（可选，因为其他进程可能还在使用）
    # ray.shutdown()
    print(f"[Process {rank}] Shutdown complete")

    # 恢复stdout/stderr并关闭日志文件
    if hasattr(sys.stdout, "close"):
        try:
            sys.stdout.close()
        except Exception:
            pass
    sys.stdout = original_stdout
    sys.stderr = original_stderr


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback

        print(f"\n[FATAL ERROR] Process failed: {e}")
        print(traceback.format_exc())
        sys.exit(1)
