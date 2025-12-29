"""StorageManager - 每个节点一个，负责管理本节点的所有 BucketStorage Actor"""

import asyncio
import hashlib
import logging
from typing import Any

from pulsing.actor import Actor, ActorId, ActorRef, ActorSystem, Message

from .storage import BucketStorage

logger = logging.getLogger(__name__)

# StorageManager 的固定服务名
STORAGE_MANAGER_NAME = "queue_storage_manager"


def _compute_owner(bucket_key: str, nodes: list[dict]) -> int | None:
    """根据 bucket key 计算 owner 节点 ID

    使用一致性哈希，确保：
    1. 同一个 bucket 总是由同一个节点负责
    2. 节点变化时影响最小
    """
    if not nodes:
        return None

    # 按 node_id 排序确保一致性（node_id 可能是字符串，统一转为 int）
    sorted_nodes = sorted(nodes, key=lambda n: int(n.get("node_id", 0)))
    hash_value = int(hashlib.md5(bucket_key.encode()).hexdigest(), 16)
    index = hash_value % len(sorted_nodes)
    # 返回 int 类型的 node_id
    node_id = sorted_nodes[index].get("node_id")
    return int(node_id) if node_id is not None else None


class StorageManager(Actor):
    """存储管理器 Actor

    每个节点一个实例，负责：
    1. 接收 GetBucket 请求
    2. 判断 bucket 是否属于本节点（一致性哈希）
    3. 属于本节点：创建/返回 BucketStorage Actor
    4. 不属于本节点：返回 Redirect 响应，指向正确节点
    """

    def __init__(self, system: ActorSystem, base_storage_path: str = "./queue_storage"):
        self.system = system
        self.base_storage_path = base_storage_path

        # 本节点管理的 bucket: {(topic, bucket_id): ActorRef}
        self._buckets: dict[tuple[str, int], ActorRef] = {}
        self._lock = asyncio.Lock()

        # 缓存的集群成员信息
        self._members: list[dict] = []
        self._members_updated_at: float = 0

    def on_start(self, actor_id: ActorId) -> None:
        logger.info(f"StorageManager started on node {self.system.node_id}")

    def on_stop(self) -> None:
        logger.info("StorageManager stopping")

    async def _refresh_members(self) -> list[dict]:
        """刷新集群成员列表（带缓存）"""
        import time

        now = time.time()
        if now - self._members_updated_at > 1.0:  # 1 秒缓存
            self._members = await self.system.members()
            self._members_updated_at = now
        return self._members

    def _bucket_key(self, topic: str, bucket_id: int) -> str:
        """生成 bucket 的唯一 key"""
        return f"{topic}:bucket_{bucket_id}"

    async def _get_or_create_bucket(
        self,
        topic: str,
        bucket_id: int,
        batch_size: int,
        storage_path: str | None = None,
    ) -> ActorRef:
        """获取或创建本地 BucketStorage Actor"""
        key = (topic, bucket_id)

        if key in self._buckets:
            return self._buckets[key]

        async with self._lock:
            if key in self._buckets:
                return self._buckets[key]

            # 创建 BucketStorage Actor
            actor_name = f"bucket_{topic}_{bucket_id}"
            # 使用传入的 storage_path 或默认路径
            if storage_path:
                bucket_storage_path = f"{storage_path}/bucket_{bucket_id}"
            else:
                bucket_storage_path = (
                    f"{self.base_storage_path}/{topic}/bucket_{bucket_id}"
                )

            try:
                # 尝试解析已存在的
                self._buckets[key] = await self.system.resolve_named(actor_name)
                logger.debug(f"Resolved existing bucket: {actor_name}")
            except Exception:
                # 创建新的
                storage = BucketStorage(
                    bucket_id=bucket_id,
                    storage_path=bucket_storage_path,
                    batch_size=batch_size,
                )
                self._buckets[key] = await self.system.spawn(
                    actor_name, storage, public=True
                )
                logger.info(f"Created bucket: {actor_name} at {bucket_storage_path}")

            return self._buckets[key]

    async def receive(self, msg: Message) -> Message | None:
        try:
            return await self._handle_message(msg)
        except Exception as e:
            logger.exception(f"Error handling message: {e}")
            return Message.from_json("Error", {"error": str(e)})

    async def _handle_message(self, msg: Message) -> Message | None:
        msg_type = msg.msg_type
        data = msg.to_json()

        if msg_type == "GetBucket":
            # 请求获取 bucket 引用
            topic = data.get("topic")
            bucket_id = data.get("bucket_id")
            batch_size = data.get("batch_size", 100)
            storage_path = data.get("storage_path")  # 可选的自定义存储路径

            if topic is None or bucket_id is None:
                return Message.from_json(
                    "Error", {"error": "Missing 'topic' or 'bucket_id'"}
                )

            # 计算 owner
            bucket_key = self._bucket_key(topic, bucket_id)
            members = await self._refresh_members()
            owner_node_id = _compute_owner(bucket_key, members)
            local_node_id = self.system.node_id.id

            # 判断是否属于本节点
            if owner_node_id is None or owner_node_id == local_node_id:
                # 本节点负责，创建/返回 bucket
                bucket_ref = await self._get_or_create_bucket(
                    topic, bucket_id, batch_size, storage_path
                )
                return Message.from_json(
                    "BucketReady",
                    {
                        "_type": "BucketReady",  # 备用：跨节点时 msg_type 可能丢失
                        "topic": topic,
                        "bucket_id": bucket_id,
                        "actor_id": bucket_ref.actor_id.local_id,
                        # 用十六进制字符串传输 node_id，避免 JSON 大整数精度丢失
                        "node_id_hex": hex(local_node_id),
                    },
                )
            else:
                # 不属于本节点，返回重定向
                # 找到 owner 节点的地址
                owner_addr = None
                for m in members:
                    # node_id 可能是字符串，统一转为 int 比较
                    m_node_id = m.get("node_id")
                    if m_node_id is not None and int(m_node_id) == owner_node_id:
                        owner_addr = m.get("addr")
                        break

                return Message.from_json(
                    "Redirect",
                    {
                        "_type": "Redirect",  # 备用：跨节点时 msg_type 可能丢失
                        "topic": topic,
                        "bucket_id": bucket_id,
                        # 用十六进制字符串传输 node_id，避免 JSON 大整数精度丢失
                        "owner_node_id_hex": hex(owner_node_id),
                        "owner_addr": owner_addr,
                    },
                )

        elif msg_type == "ListBuckets":
            # 列出本节点管理的所有 bucket
            buckets = [
                {"topic": topic, "bucket_id": bid}
                for (topic, bid) in self._buckets.keys()
            ]
            return Message.from_json("BucketList", {"buckets": buckets})

        elif msg_type == "GetStats":
            # 获取统计信息
            return Message.from_json(
                "Stats",
                {
                    "node_id": self.system.node_id.id,
                    "bucket_count": len(self._buckets),
                    "buckets": [
                        {"topic": t, "bucket_id": b} for (t, b) in self._buckets.keys()
                    ],
                },
            )

        else:
            return Message.from_json(
                "Error", {"error": f"Unknown message type: {msg_type}"}
            )


# 用于防止并发创建 StorageManager 的锁
_manager_lock = asyncio.Lock()


async def get_storage_manager(system: ActorSystem) -> ActorRef:
    """获取本节点的 StorageManager，如果不存在则创建"""
    local_node_id = system.node_id.id

    # 尝试解析本地节点的 StorageManager
    try:
        return await system.resolve_named(STORAGE_MANAGER_NAME, node_id=local_node_id)
    except Exception:
        pass

    async with _manager_lock:
        # 再次检查本地节点
        try:
            return await system.resolve_named(
                STORAGE_MANAGER_NAME, node_id=local_node_id
            )
        except Exception:
            pass

        # 创建新的 StorageManager
        try:
            manager = StorageManager(system)
            return await system.spawn(STORAGE_MANAGER_NAME, manager, public=True)
        except Exception as e:
            if "already exists" in str(e).lower():
                return await system.resolve_named(
                    STORAGE_MANAGER_NAME, node_id=local_node_id
                )
            raise


async def ensure_storage_managers(system: ActorSystem) -> None:
    """确保本地节点有 StorageManager

    每个节点需要有自己的 StorageManager 来处理 bucket 请求。
    此函数确保本地节点创建了 StorageManager。
    远程节点的 StorageManager 会在它们调用 write_queue 或 read_queue 时自动创建。
    """
    await get_storage_manager(system)
    logger.debug(f"Local StorageManager ensured on node {system.node_id.id}")


async def get_bucket_ref(
    system: ActorSystem,
    topic: str,
    bucket_id: int,
    batch_size: int = 100,
    storage_path: str | None = None,
    max_redirects: int = 3,
) -> ActorRef:
    """获取指定 bucket 的 ActorRef

    自动处理重定向，确保最终获取到正确节点上的 bucket。

    Args:
        system: Actor 系统
        topic: 队列主题
        bucket_id: bucket ID
        batch_size: 批处理大小
        storage_path: 自定义存储路径（可选）
        max_redirects: 最大重定向次数
    """
    from pulsing.actor import ActorId, NodeId

    # 先从本地 StorageManager 请求
    manager = await get_storage_manager(system)

    for redirect_count in range(max_redirects + 1):
        msg_data = {
            "topic": topic,
            "bucket_id": bucket_id,
            "batch_size": batch_size,
        }
        if storage_path:
            msg_data["storage_path"] = storage_path

        response = await manager.ask(Message.from_json("GetBucket", msg_data))

        resp_data = response.to_json()
        # 跨节点时 msg_type 可能丢失，使用 _type 字段作为备用
        msg_type = response.msg_type or resp_data.get("_type", "")

        if msg_type == "BucketReady":
            # 成功获取 bucket
            actor_id = resp_data["actor_id"]
            # node_id 用十六进制字符串传输，转为 int
            node_id = int(resp_data["node_id_hex"], 16)

            bucket_actor_id = ActorId(actor_id, NodeId(node_id))
            return await system.actor_ref(bucket_actor_id)

        elif msg_type == "Redirect":
            # 需要重定向到其他节点
            # owner_node_id 用十六进制字符串传输，转为 int
            hex_str = resp_data.get("owner_node_id_hex")
            owner_node_id = int(hex_str, 16)
            owner_addr = resp_data.get("owner_addr")

            logger.debug(
                f"Redirecting bucket {topic}:{bucket_id} to node {owner_node_id} @ {owner_addr}"
            )

            if redirect_count >= max_redirects:
                raise RuntimeError(f"Too many redirects for bucket {topic}:{bucket_id}")

            # 检查是否重定向到自己（避免无限循环）
            if owner_node_id == system.node_id.id:
                raise RuntimeError(
                    f"Redirect loop detected for bucket {topic}:{bucket_id}"
                )

            # 获取 owner 节点的 StorageManager（带重试，等待远程节点初始化）
            max_resolve_retries = 10
            for resolve_retry in range(max_resolve_retries):
                try:
                    manager = await system.resolve_named(
                        STORAGE_MANAGER_NAME, node_id=owner_node_id
                    )
                    break
                except Exception as e:
                    if resolve_retry < max_resolve_retries - 1:
                        logger.debug(
                            f"StorageManager not found on node {owner_node_id}, "
                            f"retry {resolve_retry + 1}/{max_resolve_retries}"
                        )
                        await asyncio.sleep(0.5)
                    else:
                        raise RuntimeError(
                            f"StorageManager not found on node {owner_node_id} after "
                            f"{max_resolve_retries} retries: {e}"
                        ) from e

        elif msg_type == "Error":
            raise RuntimeError(f"GetBucket failed: {resp_data.get('error')}")

        else:
            raise RuntimeError(f"Unexpected response: {msg_type}")

    raise RuntimeError(f"Failed to get bucket {topic}:{bucket_id}")
