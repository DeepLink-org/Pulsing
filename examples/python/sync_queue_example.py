#!/usr/bin/env python3
"""Distributed memory queue example (synchronous version)

Demonstrates how to use .sync() method to get synchronous wrapper for data read/write.

Differences from async version:
- writer.sync().put() instead of await writer.put()
- reader.sync().get() instead of await reader.get()
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
    """Main function"""
    logger.info("=== Distributed Memory Queue Example (Synchronous Version) ===\n")

    # Create Actor system
    system = await create_actor_system(SystemConfig.standalone())
    logger.info("✓ Actor system started\n")

    try:
        # Producer: open queue for writing, get synchronous wrapper
        writer = (
            await write_queue(
                system,
                topic="my_queue",
                bucket_column="user_id",  # Bucket by user_id
                num_buckets=4,
                batch_size=10,
            )
        ).sync()
        logger.info("✓ Queue created (synchronous writer)\n")

        # Consumer: open queue for reading, get synchronous wrapper
        reader = (await read_queue(system, topic="my_queue")).sync()
        logger.info("✓ Queue opened (synchronous reader)\n")

        # Synchronously write data
        logger.info("--- Writing data (synchronous) ---")
        for i in range(20):
            record = {
                "user_id": f"user_{i % 5}",
                "message": f"Message {i}",
                "timestamp": i,
            }
            writer.put(record)  # Synchronous call, no await needed
        logger.info("✓ Wrote 20 records\n")

        # Synchronously read data
        logger.info("--- Reading data (synchronous) ---")
        records = reader.get(limit=20)  # Synchronous call, no await needed
        logger.info(f"✓ Read {len(records)} records")
        if records:
            logger.info(f"Sample record: {records[0]}\n")

        # Synchronously flush buffer
        writer.flush()  # Synchronous call
        logger.info("✓ Data persisted\n")

        logger.info("✓ Example completed!")

    finally:
        await system.shutdown()
        logger.info("System shutdown")


if __name__ == "__main__":
    asyncio.run(main())
