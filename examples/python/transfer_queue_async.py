#!/usr/bin/env python3
"""Transfer Queue example — incremental field-by-field sample writing.

Demonstrates the transfer queue for training-inference pipelines:
1. Samples are written incrementally (prompt first, response later)
2. Complete samples are consumed in batches once all required fields are ready
3. Consumption is tracked per task_name so multiple consumers can read independently

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
    logger.info("=== Transfer Queue Example ===\n")

    try:
        # Initialize pulsing (user must call this explicitly)
        await pul.init()

        # Get an async client for partition "demo"
        client = pul.transfer_queue.get_async_client(
            partition_id="demo", num_buckets=2, batch_size=4
        )
        logger.info("Transfer queue client created (2 buckets, batch_size=4)\n")

        # --- Phase 1: Simulate inference producing prompts ---
        logger.info("--- Phase 1: Writing prompts ---")
        for i in range(6):
            meta = await client.async_put(
                sample_idx=i, data={"prompt": f"Question {i}"}
            )
            logger.info(f"  sample {i}: wrote prompt, fields={meta['fields']}")

        # Try to read — samples are incomplete (no response yet)
        incomplete = await client.async_get(
            data_fields=["prompt", "response"], batch_size=10, task_name="train"
        )
        logger.info(
            f"\nAfter prompts only: {len(incomplete)} complete samples (expected 0)\n"
        )

        # --- Phase 2: Simulate inference producing responses ---
        logger.info("--- Phase 2: Writing responses ---")
        for i in range(6):
            meta = await client.async_put(
                sample_idx=i, data={"response": f"Answer {i}"}
            )
            logger.info(f"  sample {i}: wrote response, fields={meta['fields']}")

        # --- Phase 3: Consume complete samples ---
        logger.info("\n--- Phase 3: Consuming complete samples ---")
        batch1 = await client.async_get(
            data_fields=["prompt", "response"], batch_size=4, task_name="train"
        )
        logger.info(f"Batch 1 ({len(batch1)} samples):")
        for s in batch1:
            logger.info(f"  prompt={s['prompt']!r}, response={s['response']!r}")

        batch2 = await client.async_get(
            data_fields=["prompt", "response"], batch_size=4, task_name="train"
        )
        logger.info(f"Batch 2 ({len(batch2)} samples):")
        for s in batch2:
            logger.info(f"  prompt={s['prompt']!r}, response={s['response']!r}")

        # --- Phase 4: Independent consumer reads the same data ---
        logger.info("\n--- Phase 4: Independent consumer (different task_name) ---")
        eval_batch = await client.async_get(
            data_fields=["prompt", "response"], batch_size=10, task_name="eval"
        )
        logger.info(
            f"Eval consumer got {len(eval_batch)} samples "
            f"(same data, independent tracking)"
        )

        # --- Cleanup ---
        await client.async_clear()
        logger.info("\nCleared all data")
        logger.info("Example completed!")

    finally:
        await pul.shutdown()
        logger.info("System shutdown")


if __name__ == "__main__":
    asyncio.run(main())