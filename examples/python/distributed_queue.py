#!/usr/bin/env python3
"""分布式内存队列示例

演示如何使用 write_queue 和 read_queue 进行基本的数据读写操作。

架构特点：
- 每个 bucket 对应一个独立的 BucketStorage Actor
- 内存缓冲和持久化数据同时对消费者可见
"""

import asyncio
import logging

from pulsing.actor import SystemConfig, create_actor_system
from pulsing.queue import read_queue, write_queue

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """主函数"""
    logger.info("=== 分布式内存队列示例 ===\n")

    # 创建 Actor 系统
    system = await create_actor_system(SystemConfig.standalone())
    logger.info("✓ Actor 系统已启动\n")

    try:
        # 生产者：打开队列用于写入
        writer = await write_queue(
            system,
            topic="my_queue",
            bucket_column="user_id",  # 按照 user_id 进行分桶
            num_buckets=4,
            batch_size=10,
        )
        logger.info("✓ 队列已创建（每个 bucket 一个 Actor）\n")

        # 消费者：打开队列用于读取
        reader = await read_queue(system, topic="my_queue")
        logger.info("✓ 队列已打开\n")

        # 写入数据（数据立即对消费者可见，无需等待持久化）
        logger.info("--- 写入数据 ---")
        for i in range(20):
            record = {
                "user_id": f"user_{i % 5}",
                "message": f"Message {i}",
                "timestamp": i,
            }
            await writer.put(record)
        logger.info("✓ 已写入 20 条记录\n")

        # 读取数据（内存缓冲 + 持久化数据同时可见）
        logger.info("--- 读取数据（包含内存缓冲）---")
        records = await reader.get(limit=20)
        logger.info(f"✓ 读取到 {len(records)} 条记录")
        if records:
            logger.info(f"示例记录: {records[0]}\n")

        # 刷新缓冲区
        await writer.flush()
        logger.info("✓ 数据已持久化\n")

        logger.info("✓ 示例完成!")

    finally:
        await system.shutdown()
        logger.info("系统已关闭")


if __name__ == "__main__":
    asyncio.run(main())
