"""
简单的 Actor System 示例

演示如何使用 Pulsing Actor System 构建一个简单的 Router + Worker 系统。
"""

import asyncio
from pulsing.actor import (
    ActorSystem,
    SystemConfig,
    Actor,
    Message,
    ActorId,
)


class SimpleWorker(Actor):
    """简单的 Worker Actor"""
    
    def __init__(self, name: str):
        self.name = name
        self.request_count = 0
    
    def on_start(self, actor_id: ActorId):
        print(f"[{self.name}] Started: {actor_id}")
    
    def on_stop(self):
        print(f"[{self.name}] Stopped")
    
    async def receive(self, msg: Message) -> Message:
        self.request_count += 1
        data = msg.to_json()
        
        if msg.msg_type == "Echo":
            return Message.from_json("EchoResponse", {
                "worker": self.name,
                "echo": data.get("text", ""),
                "request_count": self.request_count,
            })
        elif msg.msg_type == "Status":
            return Message.from_json("StatusResponse", {
                "worker": self.name,
                "request_count": self.request_count,
            })
        else:
            return Message.from_json("Error", {
                "error": f"Unknown message type: {msg.msg_type}"
            })


class RoundRobinRouter(Actor):
    """
    简单的 RoundRobin Router
    
    维护一个 Worker 列表，按轮询方式分发请求
    """
    
    def __init__(self):
        self.workers = []  # Worker ActorId 列表
        self.current_index = 0
    
    def on_start(self, actor_id: ActorId):
        print(f"[Router] Started: {actor_id}")
    
    async def receive(self, msg: Message) -> Message:
        data = msg.to_json()
        
        if msg.msg_type == "RegisterWorker":
            # 注册新 Worker
            worker_id = data.get("worker_id")
            if worker_id and worker_id not in self.workers:
                self.workers.append(worker_id)
                print(f"[Router] Worker registered: {worker_id} (total: {len(self.workers)})")
            return Message.from_json("Ok", {
                "message": "Worker registered",
                "total_workers": len(self.workers),
            })
        
        elif msg.msg_type == "GetNextWorker":
            # 获取下一个 Worker (RoundRobin)
            if not self.workers:
                return Message.from_json("Error", {"error": "No workers available"})
            
            worker_id = self.workers[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.workers)
            
            return Message.from_json("WorkerSelected", {
                "worker_id": worker_id,
                "total_workers": len(self.workers),
            })
        
        elif msg.msg_type == "Status":
            return Message.from_json("RouterStatus", {
                "total_workers": len(self.workers),
                "current_index": self.current_index,
                "workers": self.workers,
            })
        
        return Message.from_json("Error", {"error": f"Unknown type: {msg.msg_type}"})


async def main():
    """主函数 - 演示单节点的 Actor 系统"""
    
    print("=" * 60)
    print("Pulsing Actor System - Simple Example")
    print("=" * 60)
    
    # 创建 Actor System (standalone 模式)
    config = SystemConfig.standalone()
    system = await ActorSystem.create(config)
    print(f"\nActor System created: {system.addr}")
    
    # 创建 Router
    router_ref = await system.spawn("router", RoundRobinRouter(), public=True)
    print(f"Router spawned: router")
    
    # 创建多个 Worker
    workers = []
    for i in range(3):
        name = f"worker_{i}"
        worker_ref = await system.spawn(name, SimpleWorker(name))
        workers.append((name, worker_ref))
        print(f"Worker spawned: {name}")
    
    # 向 Router 注册所有 Worker
    print("\n--- Registering workers to router ---")
    for name, _ in workers:
        response = await router_ref.ask_json("RegisterWorker", {"worker_id": name})
        print(f"Registered {name}: {response}")
    
    # 测试 RoundRobin 调度
    print("\n--- Testing RoundRobin scheduling ---")
    for i in range(6):
        response = await router_ref.ask_json("GetNextWorker", {})
        print(f"Request {i+1}: {response}")
    
    # 直接向 Worker 发送请求
    print("\n--- Direct worker requests ---")
    for name, worker_ref in workers:
        response = await worker_ref.ask_json("Echo", {"text": f"Hello from {name}!"})
        print(f"{name} response: {response}")
    
    # 检查 Worker 状态
    print("\n--- Worker status ---")
    for name, worker_ref in workers:
        response = await worker_ref.ask_json("Status", {})
        print(f"{name}: {response}")
    
    # 检查 Router 状态
    print("\n--- Router status ---")
    response = await router_ref.ask_json("Status", {})
    print(f"Router: {response}")
    
    # 清理
    print("\n--- Shutting down ---")
    await system.shutdown()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())

