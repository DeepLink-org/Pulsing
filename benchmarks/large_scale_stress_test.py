#!/usr/bin/env python3
"""
大规模压测脚本 - 10个进程，5种worker，随机single和stream通信

使用方法:
    torchrun --nproc_per_node=10 benchmarks/large_scale_stress_test.py \
        --duration 300 \
        --rate 100 \
        --seed-nodes "127.0.0.1:8000"
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

from pulsing.actor import (
    Actor,
    ActorRef,
    ActorSystem,
    Message,
    StreamMessage,
    SystemConfig,
    create_actor_system,
)

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
# 5种不同的Worker Actor
# ============================================================================


class EchoWorker(Actor):
    """Echo Worker - 简单回显"""

    async def receive(self, msg: Message):
        if msg.msg_type == "Echo":
            data = msg.to_json()
            return Message.from_json("EchoResponse", {"echo": data.get("text", "")})
        return Message.empty()


class ComputeWorker(Actor):
    """Compute Worker - 计算密集型"""

    async def receive(self, msg: Message):
        if msg.msg_type == "Compute":
            data = msg.to_json()
            n = data.get("n", 1000)
            # 模拟计算
            result = sum(i * i for i in range(n))
            return Message.from_json("ComputeResponse", {"result": result})
        return Message.empty()


class StreamWorker(Actor):
    """Stream Worker - 流式响应"""

    async def receive(self, msg: Message):
        if msg.msg_type == "GenerateStream":
            data = msg.to_json()
            count = data.get("count", 10)
            delay = data.get("delay", 0.01)

            stream_msg, writer = StreamMessage.create("StreamItem")

            async def produce():
                try:
                    for i in range(count):
                        await writer.write_json(
                            {
                                "index": i,
                                "value": f"item_{i}",
                                "timestamp": time.time(),
                            }
                        )
                        await asyncio.sleep(delay)
                    await writer.close()
                except Exception as e:
                    await writer.error(str(e))

            asyncio.create_task(produce())
            return stream_msg
        return Message.empty()


class BatchWorker(Actor):
    """Batch Worker - 批量处理"""

    def __init__(self):
        self.batch = []
        self.batch_size = 10

    async def receive(self, msg: Message):
        if msg.msg_type == "BatchAdd":
            data = msg.to_json()
            self.batch.append(data.get("item"))

            if len(self.batch) >= self.batch_size:
                result = sum(self.batch)
                self.batch = []
                return Message.from_json("BatchResult", {"sum": result})
            return Message.from_json("BatchAck", {"count": len(self.batch)})
        return Message.empty()


class StatefulWorker(Actor):
    """Stateful Worker - 有状态处理"""

    def __init__(self):
        self.state = {}
        self.counter = 0

    async def receive(self, msg: Message):
        if msg.msg_type == "SetState":
            data = msg.to_json()
            key = data.get("key")
            value = data.get("value")
            self.state[key] = value
            self.counter += 1
            return Message.from_json("StateSet", {"counter": self.counter})

        elif msg.msg_type == "GetState":
            data = msg.to_json()
            key = data.get("key")
            value = self.state.get(key)
            return Message.from_json("StateValue", {"key": key, "value": value})

        return Message.empty()


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
        system: ActorSystem,
        worker_refs: dict[str, ActorRef],
        stats: StressTestStats,
        rate: float = 100.0,
        expected_nodes: int = 10,
        expected_worker_types: list[str] = None,
    ):
        self.system = system
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
        """从worker_type中提取基础类型

        处理格式：
        - "echo" -> "echo"
        - "echo_remote_0" -> "echo"
        - "stream_remote_5" -> "stream"
        """
        # 处理远程worker格式: "echo_remote_0"
        if "_remote_" in worker_type:
            return worker_type.split("_remote_")[0]

        # 处理本地worker格式: "echo", "compute", "stream", "batch", "stateful"
        # 这些已经是基础类型，直接返回
        if worker_type in ["echo", "compute", "stream", "batch", "stateful"]:
            return worker_type

        # 处理其他可能的格式（如"echo_0"）
        parts = worker_type.split("_")
        if len(parts) > 0 and parts[0] in [
            "echo",
            "compute",
            "stream",
            "batch",
            "stateful",
        ]:
            return parts[0]

        # 默认返回原值
        return worker_type

    async def send_single_request(self, worker_type: str) -> bool:
        """发送单个请求"""
        if worker_type not in self.worker_refs:
            return False

        worker_ref = self.worker_refs[worker_type]
        start_time = time.time()

        try:
            # 提取基础worker类型
            base_type = self._extract_base_worker_type(worker_type)

            # 只处理支持single请求的worker类型
            if base_type not in ["echo", "compute", "batch", "stateful"]:
                return False

            # 随机选择消息类型
            msg_types = {
                "echo": "Echo",
                "compute": "Compute",
                "batch": "BatchAdd",
                "stateful": random.choice(["SetState", "GetState"]),
            }

            msg_type = msg_types.get(base_type, "Echo")
            payload = self._generate_payload(base_type, msg_type)

            msg = Message.from_json(msg_type, payload)
            resp = (await worker_ref.ask(msg)).to_json()

            # 验证响应不为空
            if resp is None:
                raise ValueError("Empty response from actor")

            latency_ms = (time.time() - start_time) * 1000

            self.stats.add_request(True, latency_ms)
            return True

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = str(e)[:100]  # 增加错误信息长度以便调试
            self.stats.add_request(False, latency_ms, error_msg)
            return False

    async def send_stream_request(self, worker_type: str) -> bool:
        """发送流式请求"""
        # 提取基础类型，检查是否为stream worker
        base_type = self._extract_base_worker_type(worker_type)
        if base_type != "stream":
            return False

        # 查找stream worker（可能是本地的"stream"或远程的"stream_remote_X"）
        stream_worker_key = None
        if worker_type in self.worker_refs and base_type == "stream":
            stream_worker_key = worker_type
        elif "stream" in self.worker_refs:
            stream_worker_key = "stream"
        else:
            # 尝试查找远程stream worker
            for key in self.worker_refs.keys():
                if self._extract_base_worker_type(key) == "stream":
                    stream_worker_key = key
                    break

        if stream_worker_key is None or stream_worker_key not in self.worker_refs:
            return False

        worker_ref = self.worker_refs[stream_worker_key]
        start_time = time.time()

        try:
            payload = {
                "count": random.randint(5, 20),
                "delay": random.uniform(0.01, 0.05),
            }

            msg = Message.from_json("GenerateStream", payload)
            response = await worker_ref.ask(msg)
            reader = response.stream_reader()

            chunk_count = 0
            async for chunk_bytes in reader:
                _ = json.loads(chunk_bytes)  # Parse to validate JSON
                chunk_count += 1

            latency_ms = (time.time() - start_time) * 1000
            self.stats.add_stream(True, latency_ms)
            return True

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = str(e)[:100]  # 增加错误信息长度
            self.stats.add_stream(False, latency_ms, error_msg)
            return False

    def _generate_payload(self, worker_type: str, msg_type: str) -> dict:
        """生成随机负载"""
        if msg_type == "Echo":
            return {"text": f"echo_{random.randint(1, 1000)}"}
        elif msg_type == "Compute":
            return {"n": random.randint(100, 10000)}
        elif msg_type == "BatchAdd":
            return {"item": random.randint(1, 100)}
        elif msg_type == "SetState":
            return {
                "key": f"key_{random.randint(1, 100)}",
                "value": random.randint(1, 1000),
            }
        elif msg_type == "GetState":
            return {"key": f"key_{random.randint(1, 100)}"}
        return {}

    async def check_cluster_health(self) -> tuple[bool, str]:
        """检查集群健康状态

        Returns:
            (is_healthy, error_message)
        """
        try:
            # 检查节点数量
            members = await self.system.members()
            alive_members = [m for m in members if m.get("status") == "Alive"]

            if len(alive_members) < self.expected_nodes:
                return (
                    False,
                    f"Only {len(alive_members)}/{self.expected_nodes} nodes alive",
                )

            # 检查每个节点的worker类型
            missing_workers = []
            for member in alive_members:
                node_id = str(member.get("node_id"))
                node_workers = []

                # 检查本地worker
                for worker_type in self.expected_worker_types:
                    if worker_type in self.worker_refs:
                        node_workers.append(worker_type)

                # 检查远程worker（通过named actors）
                for worker_type in self.expected_worker_types:
                    try:
                        # 尝试查找该节点上的worker
                        instances = await self.system.get_named_instances(
                            f"{worker_type}_0"
                        )
                        for inst in instances:
                            if str(inst.get("node_id")) == node_id:
                                if worker_type not in node_workers:
                                    node_workers.append(worker_type)
                                break
                    except Exception:
                        pass

                # 检查是否缺少worker类型
                missing = [
                    wt for wt in self.expected_worker_types if wt not in node_workers
                ]
                if missing:
                    missing_workers.append(f"Node {node_id}: missing {missing}")

            if missing_workers:
                return False, f"Missing workers: {'; '.join(missing_workers)}"

            return (
                True,
                f"All {len(alive_members)} nodes healthy with {len(self.expected_worker_types)} worker types each",
            )

        except Exception as e:
            return False, f"Health check error: {e}"

    async def run_stress_test(self, duration: float):
        """运行压测"""
        end_time = time.time() + duration
        last_health_check = time.time()
        report_interval = 10.0  # 每10秒报告一次

        print(f"[StressTest] Starting stress test for {duration}s at {self.rate} req/s")
        print(
            f"[StressTest] Expected: {self.expected_nodes} nodes, {len(self.expected_worker_types)} worker types per node"
        )

        # 初始健康检查
        is_healthy, health_msg = await self.check_cluster_health()
        if not is_healthy:
            print(f"[StressTest] ⚠️  WARNING: Initial health check failed: {health_msg}")
            print("[StressTest] Continuing anyway, but monitoring cluster health...")
        else:
            print(f"[StressTest] ✓ Initial health check passed: {health_msg}")

        async def worker_loop():
            """工作循环"""
            # 使用所有可用的worker（包括本地和远程）
            local_worker_types = ["echo", "compute", "stream", "batch", "stateful"]
            local_workers = [wt for wt in local_worker_types if wt in self.worker_refs]
            remote_workers = [wt for wt in self.worker_refs.keys() if "_remote_" in wt]

            if not local_workers and not remote_workers:
                print("[StressTest] Warning: No workers available!")
                return

            # 70%使用远程worker，30%使用本地worker（优先测试远程通信）
            use_remote_probability = 0.7

            while self.running and time.time() < end_time:
                # 根据概率选择本地或远程worker
                # 70%概率选择远程worker，30%概率选择本地worker
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

                # 随机选择single或stream（70% single, 30% stream）
                # 注意：stream请求只能发送给stream worker
                if random.random() < 0.7:
                    await self.send_single_request(worker_type)
                else:
                    # 对于stream请求，确保选择的是stream worker
                    base_type = self._extract_base_worker_type(worker_type)
                    if base_type == "stream":
                        await self.send_stream_request(worker_type)
                    else:
                        # 如果不是stream worker，发送single请求
                        await self.send_single_request(worker_type)

                # 控制速率
                if self.interval > 0:
                    await asyncio.sleep(self.interval)

        # 启动多个并发worker
        num_workers = max(1, int(self.rate / 10))  # 每个worker处理约10 req/s
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

                # 显示远程/本地请求比例
                total_worker_requests = self.remote_requests + self.local_requests
                if total_worker_requests > 0:
                    remote_pct = self.remote_requests / total_worker_requests * 100
                    local_pct = self.local_requests / total_worker_requests * 100
                    print(
                        f"  Worker Selection: {remote_pct:.1f}% remote, {local_pct:.1f}% local"
                    )

                # 定期健康检查
                if time.time() - last_health_check >= self.health_check_interval:
                    is_healthy, health_msg = await self.check_cluster_health()
                    if is_healthy:
                        print(f"  ✓ Cluster Health: {health_msg}")
                    else:
                        print(f"  ⚠️  Cluster Health Warning: {health_msg}")
                        self.stats.add_health_warning(health_msg)
                    last_health_check = time.time()

        report_task = asyncio.create_task(report_loop())

        # 等待所有任务完成
        await asyncio.gather(*tasks, report_task)

        self.running = False
        print("\n[StressTest] Stress test completed")


# ============================================================================
# 主函数
# ============================================================================


async def main():
    parser = argparse.ArgumentParser(description="大规模压测脚本")
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
        "--seed-nodes",
        type=str,
        nargs="+",
        default=[],
        help="Seed节点地址列表（如: 127.0.0.1:8000）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="本地绑定端口（0表示自动分配）",
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

    # 设置日志文件 - 每个进程写入独立的日志文件
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"stress_test_rank_{rank}.log")

    # 创建文件日志处理器
    class TeeOutput:
        """同时输出到控制台和文件"""

        def __init__(self, file_path, original_stdout):
            self.file = open(file_path, "w", encoding="utf-8")
            self.original_stdout = original_stdout

        def write(self, text):
            # 写入文件
            try:
                self.file.write(text)
                self.file.flush()
            except Exception:
                pass
            # 写入原始stdout（控制台）
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

        def close(self):
            try:
                self.file.close()
            except Exception:
                pass

    # 保存原始stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # 重定向stdout和stderr到日志文件（同时输出到控制台）
    tee = TeeOutput(log_file, original_stdout)
    sys.stdout = tee
    sys.stderr = tee

    print(f"\n{'='*60}")
    print(f"Stress Test Process {rank}/{world_size}")
    print(f"Local Rank: {local_rank}, World Size: {world_size}")
    print(f"Log file: {log_file}")
    print(f"{'='*60}\n")

    # 配置系统
    if args.port == 0:
        # 自动分配端口，避免冲突
        base_port = 8000
        port = base_port + rank
    else:
        port = args.port + rank

    config = SystemConfig.with_addr(f"0.0.0.0:{port}")

    # 添加seed节点（除了自己）
    if args.seed_nodes:
        seeds = []
        for seed in args.seed_nodes:
            # 解析seed地址
            if ":" in seed:
                seeds.append(seed)
            else:
                seeds.append(f"{seed}:{8000 + int(seed.split('.')[-1]) % 10}")
        config = config.with_seeds(seeds)
    elif rank > 0:
        # 如果不是第一个进程，尝试连接到前一个进程
        prev_port = 8000 + (rank - 1)
        config = config.with_seeds([f"127.0.0.1:{prev_port}"])

    # 创建系统
    system = await create_actor_system(config)
    print(f"[Process {rank}] ActorSystem started at {system.addr}")

    # 等待集群稳定（大规模压测需要更长时间）
    # 默认故障检测超时是5秒，所以需要等待至少10秒让所有节点完成初始同步
    stabilize_timeout = args.stabilize_timeout
    print(
        f"[Process {rank}] Waiting for cluster to stabilize ({stabilize_timeout}s)..."
    )
    await asyncio.sleep(stabilize_timeout)

    # 获取集群成员
    members = await system.members()
    print(f"\n[Process {rank}] Cluster members: {len(members)}")
    if len(members) < world_size:
        print(f"  ⚠️  Warning: Only {len(members)}/{world_size} nodes found!")
        print("  This may indicate:")
        print("    1. Some processes haven't started yet")
        print("    2. Network partition (split-brain)")
        print("    3. Processes crashed or exited")
    for member in members:
        status = member.get("status", "unknown")
        print(f"  - Node {member['node_id']}: {member['addr']} [{status}]")

    # 创建5种worker（每个进程都创建）
    # 确保每个节点都有所有5种类型的worker
    worker_refs = {}
    worker_names = ["echo", "compute", "stream", "batch", "stateful"]
    required_workers = set(worker_names)
    spawned_workers = set()

    print(f"\n[Process {rank}] Spawning workers...")
    for worker_name in worker_names:
        try:
            worker_class = WORKER_TYPES[worker_name]
            worker_ref = await system.spawn(
                f"{worker_name}_{rank}",
                worker_class(),
                public=True,  # 注册为公共actor，可以被其他节点访问
            )
            worker_refs[worker_name] = worker_ref
            spawned_workers.add(worker_name)
            print(f"  ✓ Spawned {worker_name} worker: {worker_ref.actor_id}")
        except Exception as e:
            print(f"  ✗ Failed to spawn {worker_name} worker: {e}")
            # 继续创建其他worker

    # 验证所有必需的worker都已创建
    missing_workers = required_workers - spawned_workers
    if missing_workers:
        print(f"  ⚠️  WARNING: Missing workers on node {rank}: {missing_workers}")
        print("  This node does not have all required worker types!")
        print(f"  Expected: {required_workers}, Got: {spawned_workers}")
    else:
        print(
            f"  ✓ All {len(required_workers)} worker types spawned successfully on node {rank}"
        )

    # 等待所有进程启动worker并完成注册
    # 在大规模场景下，Gossip传播需要更多时间
    stabilize_timeout = args.stabilize_timeout
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
                # 尝试解析远程worker
                # 注意：spawn时已经添加了"actors/"前缀，所以这里直接使用worker名称
                # spawn创建的路径是: actors/{worker_name}_{rank}
                remote_ref = await system.resolve_named(
                    f"{worker_name}_{other_rank}",  # 去掉重复的actors/前缀
                    node_id=None,
                )
                worker_refs[f"{worker_name}_remote_{other_rank}"] = remote_ref
                resolved_count += 1
                if resolved_count <= 5:  # 只打印前5个，避免日志过多
                    print(f"  ✓ Resolved {worker_name}_{other_rank}")
            except Exception as e:
                if resolved_count < 5:  # 只打印前几个错误
                    print(f"  ✗ Failed to resolve {worker_name}_{other_rank}: {e}")

    if resolved_count > 5:
        print(f"  ... and {resolved_count - 5} more workers resolved")

    total_expected = (world_size - 1) * len(worker_names)
    if resolved_count < total_expected:
        print(
            f"  ⚠️  Warning: Only resolved {resolved_count}/{total_expected} remote workers"
        )
        print("  This may indicate cluster partition or nodes not fully registered")

    # 创建压测客户端
    stats = StressTestStats()
    client = StressTestClient(
        system,
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
    print(f"\n{'='*60}")
    print(f"Final Statistics - Process {rank}")
    print(f"{'='*60}")
    summary = stats.get_summary()
    print(json.dumps(summary, indent=2))

    # 保存统计到文件
    stats_dir = args.log_dir
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, f"stress_test_stats_rank_{rank}.json")
    with open(stats_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Process {rank}] Statistics saved to {stats_file}")

    # 关闭系统
    await system.shutdown()
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
        # 确保错误也写入日志
        import traceback

        print(f"\n[FATAL ERROR] Process failed: {e}")
        print(traceback.format_exc())
        sys.exit(1)
