#!/usr/bin/env python3
"""Distributed memory queue example

Demonstrates how to use system.queue.write/read for basic data read/write operations.

Architecture features:
- Each bucket corresponds to an independent BucketStorage Actor
- Memory buffer and persisted data are both visible to consumers simultaneously
"""

import asyncio
import logging

import pulsing as pul

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Main function"""
    logger.info("=== Distributed Memory Queue Example ===\n")

    # Create Actor system
    system = await pul.actor_system()
    logger.info("✓ Actor system started\n")

    try:
        # Producer: open queue for writing
        writer = await system.queue.write(
            "my_queue",
            bucket_column="user_id",  # Bucket by user_id
            num_buckets=4,
            batch_size=10,
        )
        logger.info("✓ Queue created (one Actor per bucket)\n")

        # Consumer: open queue for reading
        reader = await system.queue.read("my_queue")
        logger.info("✓ Queue opened\n")

        # Write data (data immediately visible to consumers, no need to wait for persistence)
        logger.info("--- Writing data ---")
        for i in range(20):
            record = {
                "user_id": f"user_{i % 5}",
                "message": f"Message {i}",
                "timestamp": i,
            }
            await writer.put(record)
        logger.info("✓ Wrote 20 records\n")

        # Read data (memory buffer + persisted data both visible)
        logger.info("--- Reading data (including memory buffer) ---")
        records = await reader.get(limit=20)
        logger.info(f"✓ Read {len(records)} records")
        if records:
            logger.info(f"Sample record: {records[0]}\n")

        # Flush buffer
        await writer.flush()
        logger.info("✓ Data persisted\n")

        logger.info("✓ Example completed!")

    finally:
        await system.shutdown()
        logger.info("System shutdown")


if __name__ == "__main__":
    asyncio.run(main())
