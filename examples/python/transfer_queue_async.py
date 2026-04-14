#!/usr/bin/env python3
"""Transfer Queue async example - exact reads from fixed-capacity buckets.

Demonstrates:
1. Explicit bucket selection on write and read
2. Incremental field merge for one sample
3. Timeout-based exact reads
4. Ring-buffer overwrite when a bucket exceeds capacity

Usage:
    python examples/python/transfer_queue_async.py
"""

import asyncio
import logging

import pulsing as pul

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    logger.info("=== Transfer Queue Async Example ===\n")

    try:
        client = await pul.transfer_queue.get_async_client(
            topic="demo_async",
            num_buckets=2,
            bucket_capacity=2,
        )
        logger.info("Transfer queue client created (2 buckets, bucket_capacity=2)\n")

        logger.info("--- Phase 1: Incremental writes into bucket 1 ---")
        meta = await client.async_put(
            sample_idx=0,
            data={"prompt": "Question 0"},
            bucket_id=1,
        )
        logger.info("sample 0 prompt write: %s", meta)

        missing = await client.async_get(
            data_fields=["prompt", "response"],
            sample_idx=0,
            bucket_id=1,
            timeout=0.1,
        )
        logger.info("before response arrives: %s", missing)

        meta = await client.async_put(
            sample_idx=0,
            data={"response": "Answer 0"},
            bucket_id=1,
        )
        logger.info("sample 0 response write: %s", meta)

        row = await client.async_get(
            data_fields=["prompt", "response"],
            sample_idx=0,
            bucket_id=1,
            timeout=1.0,
        )
        logger.info("exact read from bucket 1: %s", row)

        logger.info("\n--- Phase 2: Reads are bucket-local ---")
        wrong_bucket = await client.async_get(
            data_fields=["prompt", "response"],
            sample_idx=0,
            bucket_id=0,
            timeout=0.1,
        )
        logger.info("same sample_idx from bucket 0: %s", wrong_bucket)

        logger.info("\n--- Phase 3: Ring buffer overwrite ---")
        await client.async_put(sample_idx=10, data={"value": "oldest"}, bucket_id=0)
        await client.async_put(sample_idx=11, data={"value": "middle"}, bucket_id=0)
        await client.async_put(sample_idx=12, data={"value": "newest"}, bucket_id=0)

        evicted = await client.async_get(
            data_fields=["value"],
            sample_idx=10,
            bucket_id=0,
            timeout=0.1,
        )
        newest = await client.async_get(
            data_fields=["value"],
            sample_idx=12,
            bucket_id=0,
            timeout=0.1,
        )
        logger.info("evicted sample 10: %s", evicted)
        logger.info("newest sample 12: %s", newest)

        await client.async_clear()
        logger.info("\nCleared all data")
    finally:
        logger.info("Example finished (transfer_queue runtime cleanup is automatic)")


if __name__ == "__main__":
    asyncio.run(main())
