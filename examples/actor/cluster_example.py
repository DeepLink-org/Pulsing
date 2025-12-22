"""
集群模式示例

演示如何使用 Pulsing Actor System 构建分布式集群。

使用方法:
    # 终端 1: 启动 Router 节点
    python cluster_example.py router --addr 0.0.0.0:8000
    
    # 终端 2: 启动 Worker 节点 1
    python cluster_example.py worker --addr 0.0.0.0:8001 --seeds 127.0.0.1:8000 --name worker1
    
    # 终端 3: 启动 Worker 节点 2
    python cluster_example.py worker --addr 0.0.0.0:8002 --seeds 127.0.0.1:8000 --name worker2
    
    # 终端 4: 启动 Client
    python cluster_example.py client --seeds 127.0.0.1:8000
"""

import argparse
import asyncio
import signal

from pulsing.actor import (
    ActorSystem,
    SystemConfig,
    Actor,
    Message,
    ActorId,
)


class EchoWorker(Actor):
    """Echo Worker - 简单的回显服务"""
    
    def __init__(self, name: str):
        self.name = name
        self.request_count = 0
    
    def on_start(self, actor_id: ActorId):
        print(f"[{self.name}] Started with ID: {actor_id}")
    
    def on_stop(self):
        print(f"[{self.name}] Stopping...")
    
    def metadata(self):
        return {
            "type": "echo_worker",
            "name": self.name,
        }
    
    async def receive(self, msg: Message) -> Message:
        self.request_count += 1
        data = msg.to_json()
        
        if msg.msg_type == "Echo":
            text = data.get("text", "")
            print(f"[{self.name}] Echo request #{self.request_count}: {text}")
            return Message.from_json("EchoResponse", {
                "worker": self.name,
                "echo": text,
                "request_count": self.request_count,
            })
        
        elif msg.msg_type == "Status":
            return Message.from_json("WorkerStatus", {
                "worker": self.name,
                "request_count": self.request_count,
                "status": "healthy",
            })
        
        return Message.from_json("Error", {
            "error": f"Unknown message type: {msg.msg_type}"
        })


class SimpleRouter(Actor):
    """
    简单的 RoundRobin Router
    
    在分布式环境中，Worker 会通过消息向 Router 注册
    """
    
    def __init__(self):
        self.workers = {}  # name -> worker_id
        self.worker_order = []  # 保持顺序的 worker 名称列表
        self.current_index = 0
    
    def on_start(self, actor_id: ActorId):
        print(f"[Router] Started with ID: {actor_id}")
    
    def metadata(self):
        return {
            "type": "router",
            "scheduler": "round_robin",
        }
    
    async def receive(self, msg: Message) -> Message:
        data = msg.to_json()
        
        if msg.msg_type == "RegisterWorker":
            name = data.get("name")
            worker_id = data.get("worker_id")
            
            if name and worker_id:
                if name not in self.workers:
                    self.workers[name] = worker_id
                    self.worker_order.append(name)
                else:
                    self.workers[name] = worker_id  # 更新
                
                print(f"[Router] Worker registered: {name} = {worker_id}")
                print(f"[Router] Total workers: {len(self.workers)}")
            
            return Message.from_json("Ok", {
                "message": "Worker registered",
                "name": name,
                "total_workers": len(self.workers),
            })
        
        elif msg.msg_type == "Route":
            # RoundRobin 选择 Worker
            if not self.worker_order:
                return Message.from_json("Error", {"error": "No workers available"})
            
            name = self.worker_order[self.current_index]
            worker_id = self.workers[name]
            self.current_index = (self.current_index + 1) % len(self.worker_order)
            
            print(f"[Router] Routing to: {name}")
            
            return Message.from_json("RouteResult", {
                "worker_name": name,
                "worker_id": worker_id,
            })
        
        elif msg.msg_type == "ListWorkers":
            return Message.from_json("WorkerList", {
                "workers": list(self.workers.keys()),
                "count": len(self.workers),
            })
        
        elif msg.msg_type == "Status":
            return Message.from_json("RouterStatus", {
                "total_workers": len(self.workers),
                "current_index": self.current_index,
                "workers": list(self.workers.keys()),
            })
        
        return Message.from_json("Error", {"error": f"Unknown type: {msg.msg_type}"})


async def run_router(addr: str):
    """运行 Router 节点"""
    print(f"Starting Router at {addr}")
    
    config = SystemConfig.with_addr(addr)
    system = await ActorSystem.create(config)
    print(f"Actor System created: {system.addr}")
    
    # Spawn Router (公开到集群)
    router_ref = await system.spawn("router", SimpleRouter(), public=True)
    print(f"Router actor spawned (public)")
    
    # 设置信号处理
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        print("\nReceived shutdown signal...")
        shutdown_event.set()
    
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
    
    print("Router ready. Press Ctrl+C to stop.")
    await shutdown_event.wait()
    
    await system.shutdown()
    print("Router stopped.")


async def run_worker(addr: str, seeds: list, name: str):
    """运行 Worker 节点"""
    print(f"Starting Worker '{name}' at {addr}")
    print(f"Seed nodes: {seeds}")
    
    config = SystemConfig.with_addr(addr).with_seeds(seeds)
    system = await ActorSystem.create(config)
    print(f"Actor System created: {system.addr}")
    
    # 等待加入集群
    await asyncio.sleep(1)
    
    # 查看集群成员
    members = await system.members()
    print(f"Cluster members: {len(members)}")
    for m in members:
        print(f"  - {m}")
    
    # Spawn Worker (公开到集群)
    worker_ref = await system.spawn(f"worker.{name}", EchoWorker(name), public=True)
    print(f"Worker actor spawned: worker.{name}")
    
    # 尝试向 Router 注册
    try:
        # 通过名称查找 Router
        # 注意：这里简化处理，实际需要服务发现机制
        print("Attempting to register with router...")
        # 注册逻辑会通过集群的 gossip 协议自动发现
    except Exception as e:
        print(f"Failed to register with router: {e}")
    
    # 设置信号处理
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        print("\nReceived shutdown signal...")
        shutdown_event.set()
    
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
    
    print(f"Worker '{name}' ready. Press Ctrl+C to stop.")
    await shutdown_event.wait()
    
    await system.shutdown()
    print(f"Worker '{name}' stopped.")


async def run_client(seeds: list):
    """运行 Client - 测试集群"""
    print("Starting Client")
    print(f"Seed nodes: {seeds}")
    
    config = SystemConfig.standalone().with_seeds(seeds)
    system = await ActorSystem.create(config)
    print(f"Actor System created: {system.addr}")
    
    # 等待加入集群
    await asyncio.sleep(2)
    
    # 查看集群成员
    members = await system.members()
    print(f"\nCluster members: {len(members)}")
    for m in members:
        print(f"  - {m}")
    
    # 列出本地 actors (只是测试)
    local_actors = system.local_actor_names()
    print(f"\nLocal actors: {local_actors}")
    
    print("\nClient test complete.")
    await system.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Pulsing Actor Cluster Example")
    parser.add_argument("mode", choices=["router", "worker", "client"],
                       help="Node mode: router, worker, or client")
    parser.add_argument("--addr", default="0.0.0.0:8000",
                       help="Bind address (default: 0.0.0.0:8000)")
    parser.add_argument("--seeds", default="",
                       help="Comma-separated seed node addresses")
    parser.add_argument("--name", default="worker1",
                       help="Worker name (for worker mode)")
    
    args = parser.parse_args()
    
    # 解析 seeds
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    
    if args.mode == "router":
        asyncio.run(run_router(args.addr))
    elif args.mode == "worker":
        if not seeds:
            print("Warning: No seed nodes specified. Worker may not be able to join cluster.")
        asyncio.run(run_worker(args.addr, seeds, args.name))
    elif args.mode == "client":
        if not seeds:
            print("Error: Seeds required for client mode")
            return
        asyncio.run(run_client(seeds))


if __name__ == "__main__":
    main()

