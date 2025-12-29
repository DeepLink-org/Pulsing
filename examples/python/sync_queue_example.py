#!/usr/bin/env python3
"""分布式内存队列示例（同步版本）

演示如何使用 .sync() 方法获取同步包装器进行数据读写。

与异步版本的区别：
- writer.sync().put() 替代 await writer.put()
- reader.sync().get() 替代 await reader.get()
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
    logger.info("=== 分布式内存队列示例（同步版本）===\n")

    # 创建 Actor 系统
    system = await create_actor_system(SystemConfig.standalone())
    logger.info("✓ Actor 系统已启动\n")

    try:
        # 生产者：打开队列用于写入，获取同步包装器
        writer = (
            await write_queue(
                system,
                topic="my_queue",
                bucket_column="user_id",  # 按照 user_id 进行分桶
                num_buckets=4,
                batch_size=10,
            )
        ).sync()
        logger.info("✓ 队列已创建（同步写入器）\n")

        # 消费者：打开队列用于读取，获取同步包装器
        reader = (await read_queue(system, topic="my_queue")).sync()
        logger.info("✓ 队列已打开（同步读取器）\n")

        # 同步写入数据
        logger.info("--- 写入数据（同步）---")
        for i in range(20):
            record = {
                "user_id": f"user_{i % 5}",
                "message": f"Message {i}",
                "timestamp": i,
            }
            writer.put(record)  # 同步调用，无需 await
        logger.info("✓ 已写入 20 条记录\n")

        # 同步读取数据
        logger.info("--- 读取数据（同步）---")
        records = reader.get(limit=20)  # 同步调用，无需 await
        logger.info(f"✓ 读取到 {len(records)} 条记录")
        if records:
            logger.info(f"示例记录: {records[0]}\n")

        # 同步刷新缓冲区
        writer.flush()  # 同步调用
        logger.info("✓ 数据已持久化\n")

        logger.info("✓ 示例完成!")

    finally:
        await system.shutdown()
        logger.info("系统已关闭")


if __name__ == "__main__":
    asyncio.run(main())
